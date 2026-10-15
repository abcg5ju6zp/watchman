package index

import (
	"context"
	"fmt"
	"runtime"
	"runtime/debug"
	"sync"

	"github.com/moov-io/watchman"
	"github.com/moov-io/watchman/internal/download"
	"github.com/moov-io/watchman/internal/ingest"
	"github.com/moov-io/watchman/internal/tfidf"
	"github.com/moov-io/watchman/pkg/search"
)

type Lists interface {
	GetEntities(ctx context.Context, source search.SourceList) ([]search.Entity[search.Value], error)
	// SelectCandidates returns the subset of entities that should be scored for the query.
	// It applies source/type partitioning, name/crypto inverted indexes, and hashed
	// government-ID / address blocking keys, with safe fallbacks that never reduce
	// recall below a full partition scan.
	SelectCandidates(ctx context.Context, query search.Entity[search.Value]) (Candidates, error)

	// Update builds and publishes a new generation in one step. Refreshes use
	// PrepareCandidate/PublishCandidate instead so a failed embedding build can
	// never leave the online view on a newer text generation.
	Update(latest download.Stats)

	// PrepareCandidate builds the text corpus and TF-IDF structures for latest
	// without touching the online view. PublishCandidate swaps them in.
	PrepareCandidate(latest download.Stats) download.TextCandidate
	PublishCandidate(candidate download.TextCandidate)

	// PriorSource returns the currently published generation's data for one
	// source so an allowed-to-fail refresh can carry it over.
	PriorSource(source search.SourceList) (download.PriorSource, bool)

	LatestStats() download.Stats
	GetTFIDFIndex() *tfidf.Index
	// RefreshIngest rebuilds the ingested-file corpus from the repository.
	RefreshIngest(ctx context.Context) error
}

// Candidate is a prepared but not yet published text generation.
type Candidate struct {
	corpus *corpus
	meta   download.Stats
}

// IsRefreshCandidate marks Candidate as a download.TextCandidate.
func (c *Candidate) IsRefreshCandidate() {}

func NewLists(ingestRepository ingest.Repository) Lists {
	return &lists{
		ingestRepository: ingestRepository,
	}
}

type lists struct {
	mu          sync.RWMutex
	latestStats download.Stats
	corpus      *corpus

	ingestRepository  ingest.Repository
	ingestMu          sync.Mutex // serializes ingest corpus rebuilds
	ingestCorpus      *corpus
	ingestFingerprint string
	ingestCounts      map[string]int
	ingestHashes      map[string]string
}

func (l *lists) downloadedSource(source search.SourceList) bool {
	if source.IsRequestType() {
		return true
	}
	src := string(source)
	if src == "" {
		return true
	}
	_, ok := l.latestStats.Lists[src]
	return ok
}

func (l *lists) GetEntities(ctx context.Context, source search.SourceList) ([]search.Entity[search.Value], error) {
	if err := l.prepareIngest(ctx, source); err != nil {
		return nil, err
	}

	l.mu.RLock()
	defer l.mu.RUnlock()

	src := string(source)
	downloaded := l.downloadedSource(source)

	if src == "" {
		var out []search.Entity[search.Value]
		if l.corpus != nil {
			out = append(out, l.corpus.entities...)
		}
		if l.ingestCorpus != nil {
			out = append(out, l.ingestCorpus.entities...)
		}
		if out != nil {
			return out, nil
		}
		return l.latestStats.Entities, nil
	}

	if downloaded && !source.IsRequestType() && l.corpus != nil {
		idxs, _ := l.corpus.partitionIndices(source, "")
		return l.corpus.materialize(idxs), nil
	}
	if source.IsRequestType() && l.corpus != nil {
		return l.corpus.entities, nil
	}
	if l.ingestLoaded(source) {
		idxs, _ := l.ingestCorpus.partitionIndices(source, "")
		return l.ingestCorpus.materialize(idxs), nil
	}
	if downloaded && l.corpus != nil {
		return l.corpus.entities, nil
	}
	if l.ingestRepository != nil {
		return nil, nil
	}

	return nil, fmt.Errorf("source %s not found", source)
}

func (l *lists) SelectCandidates(ctx context.Context, query search.Entity[search.Value]) (Candidates, error) {
	source := query.Source
	if err := l.prepareIngest(ctx, source); err != nil {
		return Candidates{}, err
	}

	l.mu.RLock()
	defer l.mu.RUnlock()

	downloaded := l.downloadedSource(source)
	ingested := l.ingestLoaded(source)

	if string(source) == "" {
		var dl, ing Candidates
		if l.corpus != nil {
			dl = l.corpus.selectCandidates(query, CandidateOpts{})
		}
		if l.ingestCorpus != nil && l.ingestFingerprint != "" {
			ing = l.ingestCorpus.selectCandidates(query, CandidateOpts{})
		}
		merged := mergeCandidates(dl, ing)
		if merged.Len() > 0 || l.corpus != nil || l.ingestCorpus != nil {
			return merged, nil
		}
		return candidatesFromEntities(l.latestStats.Entities, l.latestStats.TFIDFIndex), nil
	}

	if downloaded && l.corpus != nil {
		return l.corpus.selectCandidates(query, CandidateOpts{}), nil
	}
	if ingested {
		return l.ingestCorpus.selectCandidates(query, CandidateOpts{}), nil
	}
	if downloaded {
		return candidatesFromEntities(l.latestStats.Entities, l.latestStats.TFIDFIndex), nil
	}
	if l.ingestRepository != nil {
		return Candidates{}, nil
	}

	return Candidates{}, fmt.Errorf("source %s not found", source)
}

func (l *lists) prepareIngest(ctx context.Context, source search.SourceList) error {
	if l.ingestRepository == nil {
		return nil
	}
	src := string(source)
	if src != "" && !source.IsRequestType() {
		l.mu.RLock()
		_, downloaded := l.latestStats.Lists[src]
		l.mu.RUnlock()
		if downloaded {
			return nil
		}
	}
	return l.ensureIngest(ctx)
}

func (l *lists) LatestStats() download.Stats {
	l.mu.RLock()
	defer l.mu.RUnlock()

	lists := l.latestStats.Lists
	hashes := l.latestStats.ListHashes
	if len(l.ingestCounts) > 0 {
		lists = copyStringInt(l.latestStats.Lists)
		hashes = copyStringString(l.latestStats.ListHashes)
		if lists == nil {
			lists = make(map[string]int, len(l.ingestCounts))
		}
		if hashes == nil {
			hashes = make(map[string]string, len(l.ingestHashes))
		}
		for src, n := range l.ingestCounts {
			lists[src] = n
			hashes[src] = l.ingestHashes[src]
		}
	}

	results := l.latestStats.SourceResults
	if len(results) > 0 {
		results = make(map[string]download.SourceResult, len(l.latestStats.SourceResults))
		for k, v := range l.latestStats.SourceResults {
			results[k] = v
		}
	}

	return download.Stats{
		Lists:         lists,
		ListHashes:    hashes,
		SourceResults: results,
		StartedAt:     l.latestStats.StartedAt,
		EndedAt:       l.latestStats.EndedAt,
		Version:       watchman.Version,
		GenerationID:  l.latestStats.GenerationID,
	}
}

// Update builds and immediately publishes a new generation. The coordinated
// refresh path uses PrepareCandidate and PublishCandidate separately so the
// online view only advances once every enabled capability is ready.
//
// Memory ownership after Update:
//   - The entity slice and TF-IDF index live only on the in-memory corpus.
//   - latestStats keeps list metadata (counts, hashes, timestamps) only.
//
// Callers should drop their own references to stats.Entities after Update so the
// previous generation (and download temps) can be collected. Update runs a GC
// and FreeOSMemory after the swap to reduce refresh peak RSS / OOM risk when the
// old and new corpora would otherwise coexist until the next natural GC cycle.
func (l *lists) Update(latest download.Stats) {
	l.PublishCandidate(l.PrepareCandidate(latest))
}

// PrepareCandidate builds the search corpus and metadata for latest without
// changing the online view. The returned candidate is immutable and safe to
// discard if a later refresh stage (e.g. the embedding build) fails.
func (l *lists) PrepareCandidate(latest download.Stats) download.TextCandidate {
	// Build search corpus outside any lock (CPU-heavy).
	// corpus.entities aliases latest.Entities (no extra copy of the slice).
	c := buildCorpus(latest.Entities, latest.TFIDFIndex)

	// Metadata only — do not retain a second root to the entity slice / TF-IDF
	// index via latestStats.
	meta := latest
	meta.Entities = nil
	meta.TFIDFIndex = nil

	return &Candidate{corpus: c, meta: meta}
}

// PublishCandidate atomically swaps the online view onto candidate. It must be
// called only after every enabled capability for the generation is prepared.
func (l *lists) PublishCandidate(candidate download.TextCandidate) {
	c, ok := candidate.(*Candidate)
	if !ok || c == nil {
		return
	}

	l.mu.Lock()
	// Overwriting l.corpus drops the only lists-owned root to the previous
	// generation; nothing else in this frame retains it.
	l.latestStats = c.meta
	l.corpus = c.corpus
	l.mu.Unlock()

	// Refresh is infrequent; reclaim the prior generation promptly so cgroup
	// peaks are closer to one corpus than two. FreeOSMemory returns idle heap
	// pages to the OS (important after large refreshes).
	runtime.GC()
	debug.FreeOSMemory()
}

// PriorSource returns the published generation's entities and metadata for one
// source. The returned entities are value copies safe to reuse while the live
// corpus keeps serving readers.
func (l *lists) PriorSource(source search.SourceList) (download.PriorSource, bool) {
	l.mu.RLock()
	defer l.mu.RUnlock()

	if l.corpus == nil {
		return download.PriorSource{}, false
	}
	idxs, ok := l.corpus.partitionIndices(source, "")
	if !ok || len(idxs) == 0 {
		return download.PriorSource{}, false
	}
	return download.PriorSource{
		Entities:     l.corpus.materialize(idxs),
		Hash:         l.latestStats.ListHashes[string(source)],
		GenerationID: l.latestStats.GenerationID,
	}, true
}

func (l *lists) GetTFIDFIndex() *tfidf.Index {
	l.mu.RLock()
	defer l.mu.RUnlock()

	if l.corpus != nil {
		return l.corpus.tfidf
	}
	return l.latestStats.TFIDFIndex
}
