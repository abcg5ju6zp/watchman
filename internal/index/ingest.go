package index

import (
	"context"
	"fmt"

	"github.com/moov-io/base/telemetry"
	"github.com/moov-io/watchman/internal/ingest"
	"github.com/moov-io/watchman/internal/tfidf"
	"github.com/moov-io/watchman/pkg/search"

	"go.opentelemetry.io/otel/attribute"
)

// ingestSnapshotLoadAttempts bounds how often rebuild re-reads summaries when
// the database generation changes while entities are being listed.
const ingestSnapshotLoadAttempts = 3

// RefreshIngest rebuilds the in-memory ingest corpus from the repository.
// POST /v2/ingest calls this so the next search sees the new rows without waiting
// for a checksum miss. Startup also calls it to warm the cache (and to finish
// publishing any revision left pending by a previous process).
//
// The rebuild is conditional: it never publishes a snapshot that is older per
// source than the generation already live, so an out-of-order refresh for an
// older upload cannot cover a newer successful version.
func (l *lists) RefreshIngest(ctx context.Context) error {
	if l.ingestRepository == nil {
		return nil
	}

	l.ingestMu.Lock()
	defer l.ingestMu.Unlock()
	return l.rebuildIngestLocked(ctx)
}

// IngestVersion reports the source revision and content version currently
// served by the live ingest generation. Callers use it to confirm that summary,
// corpus, TFIDF and online search all agree on the version an upload staged.
func (l *lists) IngestVersion(source string) (ingest.SourceChecksum, bool) {
	l.mu.RLock()
	defer l.mu.RUnlock()

	if l.ingestCorpus == nil {
		return ingest.SourceChecksum{}, false
	}
	revision, ok := l.ingestRevisions[source]
	if !ok {
		return ingest.SourceChecksum{}, false
	}
	return ingest.SourceChecksum{
		Source:      source,
		Revision:    revision,
		Version:     l.ingestVersions[source],
		EntityCount: l.ingestCounts[source],
		Checksum:    l.ingestHashes[source],
		Published:   true,
	}, true
}

func (l *lists) ensureIngest(ctx context.Context) error {
	if l.ingestRepository == nil {
		return nil
	}

	sums, err := l.ingestRepository.Checksums(ctx)
	if err != nil {
		return fmt.Errorf("ingest checksums: %w", err)
	}
	fp := ingest.Fingerprint(sums)

	l.mu.RLock()
	hit := l.ingestCorpus != nil && l.ingestFingerprint == fp
	l.mu.RUnlock()
	if hit {
		return nil
	}

	l.ingestMu.Lock()
	defer l.ingestMu.Unlock()

	l.mu.RLock()
	hit = l.ingestCorpus != nil && l.ingestFingerprint == fp
	l.mu.RUnlock()
	if hit {
		return nil
	}

	return l.rebuildIngestLocked(ctx)
}

type ingestSnapshot struct {
	sums     []ingest.SourceChecksum
	entities []search.Entity[search.Value]
}

// loadIngestSnapshot reads summary rows and entities as one generation. The
// two reads are re-checked against each other so a replacement landing between
// them cannot yield entities from one version described by another version's
// summary; the load retries instead.
func (l *lists) loadIngestSnapshot(ctx context.Context) (ingestSnapshot, string, error) {
	var last ingestSnapshot

	for attempt := 0; attempt < ingestSnapshotLoadAttempts; attempt++ {
		sums, err := l.ingestRepository.Checksums(ctx)
		if err != nil {
			return ingestSnapshot{}, "", fmt.Errorf("ingest checksums: %w", err)
		}
		fp := ingest.Fingerprint(sums)

		var entities []search.Entity[search.Value]
		if fp != "" {
			entities, err = l.ingestRepository.ListAll(ctx)
			if err != nil {
				return ingestSnapshot{}, "", fmt.Errorf("listing ingested entities: %w", err)
			}
		}

		// Re-read the summaries and require the same generation plus matching
		// totals before accepting the snapshot.
		sumsAfter, err := l.ingestRepository.Checksums(ctx)
		if err != nil {
			return ingestSnapshot{}, "", fmt.Errorf("ingest checksums: %w", err)
		}
		if ingest.Fingerprint(sumsAfter) != fp {
			last = ingestSnapshot{sums: sumsAfter}
			continue
		}

		total := 0
		for _, s := range sumsAfter {
			total += s.EntityCount
		}
		if total != len(entities) {
			last = ingestSnapshot{sums: sumsAfter}
			continue
		}

		return ingestSnapshot{sums: sumsAfter, entities: entities}, fp, nil
	}

	return last, "", fmt.Errorf("ingest snapshot changed while loading; retry the publish")
}

// wouldDowngradeIngest reports whether any source in the freshly loaded
// snapshot is older than the revision already live in memory.
func (l *lists) wouldDowngradeIngest(sums []ingest.SourceChecksum) bool {
	l.mu.RLock()
	defer l.mu.RUnlock()

	for _, s := range sums {
		if current, ok := l.ingestRevisions[s.Source]; ok && s.Revision < current {
			return true
		}
	}
	return false
}

func (l *lists) rebuildIngestLocked(ctx context.Context) error {
	ctx, span := telemetry.StartSpan(ctx, "ingest-rebuild-corpus")
	defer span.End()

	snapshot, fp, err := l.loadIngestSnapshot(ctx)
	if err != nil {
		return err
	}
	sums := snapshot.sums

	// Conditional publish: an older refresh losing the race to a newer one
	// must leave the newer in-memory generation untouched.
	if l.wouldDowngradeIngest(sums) {
		span.SetAttributes(attribute.Bool("ingest.skipped_stale", true))
		return nil
	}

	entities := snapshot.entities

	// Corpus and TFIDF are both built from this single entity generation, so
	// term weights, partitions and the summary can never disagree on versions.
	tfidfIndex := buildIngestTFIDFIndex(entities)
	c := buildCorpus(entities, tfidfIndex)

	counts := make(map[string]int, len(sums))
	hashes := make(map[string]string, len(sums))
	revisions := make(map[string]int64, len(sums))
	versions := make(map[string]string, len(sums))
	pending := make([]ingest.SourceChecksum, 0, len(sums))
	total := 0
	for _, s := range sums {
		counts[s.Source] = s.EntityCount
		hashes[s.Source] = s.Checksum
		revisions[s.Source] = s.Revision
		versions[s.Source] = s.Version
		total += s.EntityCount
		if !s.Published {
			pending = append(pending, s)
		}
	}

	span.SetAttributes(
		attribute.Int("ingest.source_count", len(sums)),
		attribute.Int("ingest.entity_count", total),
		attribute.String("ingest.fingerprint", fp),
	)

	// One pointer swap publishes corpus, TFIDF and per-source versions
	// together; in-flight searches hold the previous generation and new
	// searches see the new one — no mixed reads.
	l.mu.Lock()
	l.ingestCorpus = c
	l.ingestFingerprint = fp
	l.ingestCounts = counts
	l.ingestHashes = hashes
	l.ingestRevisions = revisions
	l.ingestVersions = versions
	l.mu.Unlock()

	// The rows are now live in this process' online search path. Confirm the
	// persisted publish stage so a crash immediately after the swap cannot
	// leave the revision pending forever. Conditional on revision: revisions
	// replaced concurrently report ok=false and are simply skipped.
	for _, s := range pending {
		ok, err := l.ingestRepository.MarkPublished(ctx, s.Source, s.Revision)
		if err != nil {
			return fmt.Errorf("confirming %s revision %d published: %w", s.Source, s.Revision, err)
		}
		if !ok {
			continue
		}
	}
	return nil
}

// buildIngestTFIDFIndex builds TF-IDF weights from the same entity generation
// the corpus is about to be built from. Indexing is opt-in via TFIDF_ENABLED,
// matching the downloaded-lists path; a disabled index is still returned so
// callers have a stable non-nil handle.
func buildIngestTFIDFIndex(entities []search.Entity[search.Value]) *tfidf.Index {
	cfg := tfidf.ConfigFromEnvironment()
	idx := tfidf.NewIndex(cfg)
	if !cfg.Enabled {
		return idx
	}

	var documents [][]string
	for i := range entities {
		if len(entities[i].PreparedFields.NameFields) > 0 {
			documents = append(documents, entities[i].PreparedFields.NameFields)
		}
		for _, altFields := range entities[i].PreparedFields.AltNameFields {
			if len(altFields) > 0 {
				documents = append(documents, altFields)
			}
		}
	}
	idx.Build(documents)
	return idx
}

func (l *lists) ingestLoaded(source search.SourceList) bool {
	if l.ingestCorpus == nil {
		return false
	}
	src := string(source)
	if src == "" {
		return l.ingestFingerprint != ""
	}
	_, ok := l.ingestCounts[src]
	return ok
}

func mergeCandidates(a, b Candidates) Candidates {
	if a.Len() == 0 {
		return b
	}
	if b.Len() == 0 {
		return a
	}

	n := a.Len() + b.Len()
	ents := make([]search.Entity[search.Value], 0, n)
	idxs := make([]int, n)
	for i := 0; i < a.Len(); i++ {
		ents = append(ents, a.At(i))
		idxs[i] = i
	}
	base := len(ents)
	for i := 0; i < b.Len(); i++ {
		ents = append(ents, b.At(i))
		idxs[base+i] = base + i
	}

	tfidf := a.TFIDF
	if tfidf == nil {
		tfidf = b.TFIDF
	}
	return Candidates{Entities: ents, Indices: idxs, TFIDF: tfidf}
}

func copyStringInt(in map[string]int) map[string]int {
	if in == nil {
		return nil
	}
	out := make(map[string]int, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func copyStringString(in map[string]string) map[string]string {
	if in == nil {
		return nil
	}
	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}
