package download

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"sync"
	"time"

	"github.com/moov-io/base/log"
	"github.com/moov-io/base/telemetry"
	"github.com/moov-io/watchman/internal/embeddings"
	"github.com/moov-io/watchman/pkg/search"
)

// Trigger labels describe what initiated a refresh.
const (
	TriggerStartup   = "startup"
	TriggerScheduled = "scheduled"
	TriggerManual    = "manual"
)

// ErrAlreadyRunning is returned when a refresh is requested while another is in progress.
var ErrAlreadyRunning = errors.New("a data refresh is already running")

// State describes the lifecycle of the current or most recent refresh.
type State string

const (
	StateIdle      State = "idle"
	StateRunning   State = "running"
	StateSucceeded State = "succeeded"
	StateFailed    State = "failed"
)

// SourceStatus is the API-facing provenance of one source in a generation.
type SourceStatus struct {
	Source   string `json:"source"`
	Entities int    `json:"entities"`
	Hash     string `json:"hash,omitempty"`

	// Fresh is true when the source was downloaded for this generation.
	Fresh bool `json:"fresh"`
	// Stale is true when the source could not be refreshed and the previous
	// generation's data is being served for it.
	Stale bool `json:"stale,omitempty"`
	// PriorGenerationID identifies the generation the stale data came from.
	PriorGenerationID string `json:"priorGenerationId,omitempty"`
	// Error is set when the source failed to refresh.
	Error string `json:"error,omitempty"`
}

// Status is a snapshot of the current or most recent data refresh.
type Status struct {
	State   State  `json:"state"`
	Trigger string `json:"trigger,omitempty"`

	StartedAt  *time.Time `json:"startedAt,omitempty"`
	FinishedAt *time.Time `json:"finishedAt,omitempty"`
	Duration   string     `json:"duration,omitempty"`
	LastError  string     `json:"lastError,omitempty"`

	// GenerationID identifies the published generation currently online.
	GenerationID string `json:"generationId,omitempty"`
	// Sources reports per-source provenance of the published generation.
	Sources []SourceStatus `json:"sources,omitempty"`
}

// TextCandidate is an opaque prepared text generation. Implementations live in
// internal/index; the download package only passes the handle from
// PrepareCandidate to PublishCandidate.
type TextCandidate interface {
	IsRefreshCandidate()
}

// publishedLists is the online text view a refresh atomically swaps.
type publishedLists interface {
	Update(Stats)
	PrepareCandidate(Stats) TextCandidate
	PublishCandidate(TextCandidate)
	PriorSource(source search.SourceList) (PriorSource, bool)
}

// embeddingCandidates is the optional vector index capability.
type embeddingCandidates interface {
	EmbeddingsEnabled() bool
	PrepareEmbeddingIndex(ctx context.Context, names, ids []string) (embeddings.IndexCandidate, error)
	PublishEmbeddingIndex(candidate embeddings.IndexCandidate)
}

// generationPublisher publishes the text corpus and vector index of one
// generation atomically relative to online searches.
type generationPublisher interface {
	PublishGeneration(text TextCandidate, vectors embeddings.IndexCandidate)
}

// Option configures a Refresher.
type Option func(*Refresher)

// WithGenerationStore enables durable candidate staging and startup recovery.
func WithGenerationStore(store *generationStore) Option {
	return func(r *Refresher) {
		r.store = store
	}
}

// Refresher coordinates full data refreshes with single-flight semantics,
// generation-pinned candidates, atomic publication and status tracking.
type Refresher struct {
	baseCtx context.Context
	logger  log.Logger

	dl            Downloader
	indexedLists  publishedLists
	searchService embeddingCandidates
	store         *generationStore

	mu      sync.Mutex
	running bool
	status  Status

	// resumeWS is a leftover prepared candidate found by Recover and consumed
	// by the next refresh (the startup load).
	resumeWS *GenerationWorkspace
}

func NewRefresher(baseCtx context.Context, logger log.Logger, dl Downloader, su publishedLists, es embeddingCandidates, opts ...Option) *Refresher {
	r := &Refresher{
		baseCtx:       baseCtx,
		logger:        logger,
		dl:            dl,
		indexedLists:  su,
		searchService: es,
		status:        Status{State: StateIdle},
	}
	for _, opt := range opts {
		opt(r)
	}
	return r
}

// RefreshNow runs a full refresh synchronously. Returns ErrAlreadyRunning if one is in progress.
func (r *Refresher) RefreshNow(ctx context.Context, trigger string) error {
	start, ok := r.start(trigger)
	if !ok {
		return ErrAlreadyRunning
	}
	err := r.run(ctx, trigger)
	r.finish(start, err)
	return err
}

// TriggerAsync starts a refresh in the background. Returns false if one is already running.
func (r *Refresher) TriggerAsync(trigger string) bool {
	start, ok := r.start(trigger)
	if !ok {
		return false
	}
	go func() {
		err := r.run(r.baseCtx, trigger)
		r.finish(start, err)
		if err != nil {
			r.logger.Error().LogErrorf("background %s data refresh failed: %v", trigger, err)
		}
	}()
	return true
}

// Status returns a snapshot of the current or most recent refresh.
func (r *Refresher) Status() Status {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.status
}

// Recover resolves candidates left on disk by a process exit during publish.
// It must be called once at startup, before the initial RefreshNow: a complete
// candidate is resumed by that refresh using its pinned files, while every
// incomplete candidate is discarded before any new download can start.
func (r *Refresher) Recover(ctx context.Context) error {
	if r.store == nil {
		return nil
	}
	ws, err := r.store.Recover()
	if err != nil {
		return fmt.Errorf("recovering refresh generation: %w", err)
	}
	r.resumeWS = ws
	return nil
}

func (r *Refresher) start(trigger string) (time.Time, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.running {
		return time.Time{}, false
	}
	r.running = true
	now := time.Now().UTC()
	r.status = Status{
		State:        StateRunning,
		Trigger:      trigger,
		StartedAt:    &now,
		GenerationID: r.status.GenerationID,
		Sources:      r.status.Sources,
	}
	return now, true
}

func (r *Refresher) finish(start time.Time, err error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	end := time.Now().UTC()
	r.running = false
	r.status.FinishedAt = &end
	r.status.Duration = end.Sub(start).String()
	if err != nil {
		r.status.State = StateFailed
		r.status.LastError = err.Error()
		return
	}
	r.status.State = StateSucceeded
	r.status.LastError = ""
}

func (r *Refresher) embeddingsEnabled() bool {
	return r.searchService != nil && r.searchService.EmbeddingsEnabled()
}

// run executes one refresh generation. A resumed startup candidate that cannot
// be completed as-is is discarded and replaced by a single fresh generation —
// never by a mixture of leftover and freshly downloaded files.
func (r *Refresher) run(ctx context.Context, trigger string) error {
	ctx, span := telemetry.StartSpan(ctx, "refresh-all-sources")
	defer span.End()

	ws := r.resumeWS
	r.resumeWS = nil
	resumed := ws != nil
	if ws == nil {
		ws = r.createWorkspace()
	}

	err := r.buildAndPublish(ctx, ws, resumed)
	if err == nil {
		return nil
	}
	if ws != nil {
		ws.Discard()
	}

	// Startup recovery is binary: either complete the same candidate or drop it
	// entirely. After discarding an uncompletable candidate, load one coherent
	// fresh generation. Manual/scheduled failures just keep the prior view.
	if resumed && trigger == TriggerStartup {
		r.logger.Warn().Logf("discarding interrupted refresh candidate and starting a fresh generation: %v", err)
		fresh := r.createWorkspace()
		freshErr := r.buildAndPublish(ctx, fresh, false)
		if freshErr != nil && fresh != nil {
			fresh.Discard()
		}
		return freshErr
	}
	return err
}

func (r *Refresher) createWorkspace() *GenerationWorkspace {
	if r.store == nil {
		return nil
	}
	ws, err := r.store.Create(r.embeddingsEnabled())
	if err != nil {
		r.logger.Warn().Logf("refresh generation will run without durable staging: %v", err)
		return nil
	}
	return ws
}

func (r *Refresher) buildAndPublish(ctx context.Context, ws *GenerationWorkspace, resumed bool) error {
	generationID := ws.ID()
	if generationID == "" {
		generationID = newGenerationID()
	}
	if resumed {
		r.logger.Info().Logf("completing interrupted refresh generation %s from pinned files", generationID)
	} else {
		r.logger.Info().Logf("starting refresh generation %s", generationID)
	}

	// Bind downloads to the isolated candidate directory and the live prior
	// generation for the duration of this RefreshAll call.
	var prior PriorData
	if r.indexedLists != nil {
		prior = r.indexedLists
	}
	if setter, ok := r.dl.(refreshRuntimeSetter); ok {
		setter.SetRefreshRuntime(&refreshRuntime{
			workingDir:   ws.DataDir(),
			generationID: generationID,
			prior:        prior,
			workspace:    ws,
		})
		defer setter.SetRefreshRuntime(nil)
	}

	stats, err := r.dl.RefreshAll(ctx)

	entityCount := len(stats.Entities)

	if err != nil {
		// A required source failed (or validation failed): no candidate is
		// published and the complete previous generation keeps serving.
		return fmt.Errorf("loading refresh generation %s: %w", generationID, err)
	}

	// The Refresher owns the generation ID; make sure Stats carry it even when
	// the Downloader implementation did not stamp it.
	stats.GenerationID = generationID

	if resumed {
		// Guarantee the reload used the pinned bytes; a mismatch means this
		// candidate cannot be completed and must be discarded.
		if err := ws.Matches(stats); err != nil {
			return err
		}
		ws.ApplyProvenance(&stats)
	}

	// Extract vector inputs before text indexing takes ownership of entities.
	var names, ids []string
	if r.embeddingsEnabled() && entityCount > 0 {
		names = make([]string, entityCount)
		ids = make([]string, entityCount)
		for i := range stats.Entities {
			names[i] = stats.Entities[i].Name
			ids[i] = stats.Entities[i].SourceID
		}
	}

	// 1. Build the text candidate (corpus + TF-IDF) without publishing it.
	var textCandidate TextCandidate
	if r.indexedLists != nil {
		textCandidate = r.indexedLists.PrepareCandidate(stats)
	}

	// 2. Build the optional vector candidate without publishing it. A provider
	//    or cache failure here leaves the online view entirely on the previous
	//    generation — text is never swapped before vectors are ready.
	var embeddingCandidate embeddings.IndexCandidate
	if r.embeddingsEnabled() {
		if entityCount == 0 {
			r.logger.Info().Log("embeddings: no entities to index for generation")
		} else {
			candidate, buildErr := r.searchService.PrepareEmbeddingIndex(ctx, names, ids)
			if buildErr != nil {
				return fmt.Errorf("building embedding index for generation %s: %w", generationID, buildErr)
			}
			embeddingCandidate = candidate
		}
	}

	// 3. Durable barrier: once this manifest exists, a crash during publish is
	//    recoverable by completing exactly this candidate.
	if ws != nil {
		if err := ws.MarkPrepared(stats, r.embeddingsEnabled()); err != nil {
			// Without the prepared marker recovery could not validate the
			// candidate; drop the staged files and publish in memory only.
			r.logger.Warn().Logf("could not stage refresh generation %s for recovery: %v", generationID, err)
			ws.Discard()
			ws = nil
		}
	}

	// 4. Publish the online views only after every enabled capability is ready.
	//    Text and vectors swap together under one publication lock so requests
	//    cannot straddle two generations.
	if pub, ok := r.searchService.(generationPublisher); ok && r.indexedLists != nil {
		pub.PublishGeneration(textCandidate, embeddingCandidate)
		// Index takes ownership of the entity slice and TF-IDF index.
		stats.Entities = nil
		stats.TFIDFIndex = nil
	} else {
		if r.indexedLists != nil {
			r.indexedLists.PublishCandidate(textCandidate)
			stats.Entities = nil
			stats.TFIDFIndex = nil
		}
		if r.embeddingsEnabled() {
			r.searchService.PublishEmbeddingIndex(embeddingCandidate)
		}
	}

	// 5. Promote pinned files and close out the generation. A failure here is
	//    logged: the in-memory generation is live and the prepared candidate
	//    remains on disk for startup recovery to complete with the same bytes.
	if ws != nil {
		if err := ws.Publish(stats); err != nil {
			r.logger.Warn().Logf("finalizing refresh generation %s on disk: %v", generationID, err)
		}
	}

	r.setPublished(generationID, stats)

	r.logger.Info().Logf("search index swapped to generation %s (%d entities, heap %.2fGB)",
		generationID, entityCount, getCurrentMemoryUsed())
	return nil
}

func (r *Refresher) setPublished(generationID string, stats Stats) {
	sources := sourceStatuses(stats)

	r.mu.Lock()
	defer r.mu.Unlock()
	r.status.GenerationID = generationID
	r.status.Sources = sources
}

func sourceStatuses(stats Stats) []SourceStatus {
	if len(stats.SourceResults) == 0 {
		return nil
	}
	names := make([]string, 0, len(stats.SourceResults))
	for name := range stats.SourceResults {
		names = append(names, name)
	}
	// Deterministic API output.
	for i := 0; i < len(names); i++ {
		for j := i + 1; j < len(names); j++ {
			if names[j] < names[i] {
				names[i], names[j] = names[j], names[i]
			}
		}
	}

	out := make([]SourceStatus, 0, len(names))
	for _, name := range names {
		res := stats.SourceResults[name]
		out = append(out, SourceStatus{
			Source:            name,
			Entities:          res.Entities,
			Hash:              res.Hash,
			Fresh:             res.Fresh,
			Stale:             res.Stale,
			PriorGenerationID: res.PriorGenerationID,
			Error:             res.Error,
		})
	}
	return out
}

func getCurrentMemoryUsed() float64 {
	var mem runtime.MemStats
	runtime.ReadMemStats(&mem)
	return float64(mem.Alloc) / 1024 / 1024 / 1024
}
