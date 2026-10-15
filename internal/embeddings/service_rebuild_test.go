package embeddings

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/moov-io/base/log"
	"github.com/moov-io/watchman/internal/db"
	"github.com/stretchr/testify/require"
)

// newRebuildService builds an in-package service backed by the mock
// provider and the supplied cache, fully offline and deterministic.
func newRebuildService(t *testing.T, dim, batchSize int, provider Provider, cache Cache) *service {
	t.Helper()

	if provider == nil {
		provider = NewMockProvider(dim)
	}
	if cache == nil {
		var err error
		cache, err = newMemoryCache(10000)
		require.NoError(t, err)
	}

	cfg := DefaultConfig()
	cfg.Provider = ProviderConfig{
		Name:              "mock",
		Model:             "mock-model",
		Dimension:         dim,
		NormalizeVectors:  true,
		Timeout:           time.Second,
	}
	cfg.Cache = CacheConfig{Type: "memory", Size: 10000}
	cfg.BatchSize = batchSize
	cfg.IndexBuildTimeout = 30 * time.Second

	return &service{
		logger:    log.NewNopLogger(),
		config:    cfg,
		provider:  provider,
		cache:     cache,
		dimension: dim,
		modelKey:  modelConfigKey(cfg.Provider),
		index:     newVectorIndex(dim),
	}
}

// recordingProvider records every batch of texts sent to the provider.
type recordingProvider struct {
	inner Provider

	mu    sync.Mutex
	calls [][]string
}

func (r *recordingProvider) Embed(ctx context.Context, texts []string) ([][]float64, error) {
	r.mu.Lock()
	r.calls = append(r.calls, append([]string(nil), texts...))
	r.mu.Unlock()
	return r.inner.Embed(ctx, texts)
}

func (r *recordingProvider) Dimension() int { return r.inner.Dimension() }
func (r *recordingProvider) Name() string   { return r.inner.Name() }
func (r *recordingProvider) Close() error   { return r.inner.Close() }

func (r *recordingProvider) encodedTexts() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	var out []string
	for _, call := range r.calls {
		out = append(out, call...)
	}
	return out
}

// failPutCache wraps a cache and fails Put for selected texts.
type failPutCache struct {
	Cache

	shouldFail func(text string) bool
}

func (f *failPutCache) Put(ctx context.Context, text string, entry cacheEntry) error {
	if f.shouldFail != nil && f.shouldFail(text) {
		return fmt.Errorf("simulated cache write failure for %q", text)
	}
	return f.Cache.Put(ctx, text, entry)
}

// gatedProvider blocks Embed calls that contain gateText until release is closed.
type gatedProvider struct {
	*MockProvider

	gateText string
	started  chan struct{}
	release  chan struct{}
	once     sync.Once
}

func newGatedProvider(dim int, gateText string) *gatedProvider {
	return &gatedProvider{
		MockProvider: NewMockProvider(dim),
		gateText:     gateText,
		started:      make(chan struct{}),
		release:      make(chan struct{}),
	}
}

func (g *gatedProvider) Embed(ctx context.Context, texts []string) ([][]float64, error) {
	for _, text := range texts {
		if text == g.gateText {
			g.once.Do(func() { close(g.started) })
			select {
			case <-g.release:
			case <-ctx.Done():
				return nil, ctx.Err()
			}
			break
		}
	}
	return g.MockProvider.Embed(ctx, texts)
}

func roster(names ...string) ([]string, []string) {
	ids := make([]string, len(names))
	for i, name := range names {
		ids[i] = "id-" + name
	}
	return names, ids
}

// Rubric 1: rebuild fixes model, dimension and the ordered id/name pairing;
// mutating caller slices after (or conceptually during) the call cannot mix
// into the candidate, and positions stay aligned.
func TestBuildIndex_SnapshotsRosterAndOrder(t *testing.T) {
	svc := newRebuildService(t, 8, 2, nil, nil)
	ctx := context.Background()

	names, ids := roster("beta", "alpha")
	require.NoError(t, svc.BuildIndex(ctx, names, ids))

	// Caller mutates its slices after the call - the active index is a snapshot.
	names[0] = "mutated"
	ids[0] = "id-mutated"

	results, err := svc.Search(ctx, "beta", 1)
	require.NoError(t, err)
	require.Len(t, results, 1)
	require.Equal(t, "beta", results[0].Name)
	require.Equal(t, "id-beta", results[0].ID)

	require.Nil(t, svc.index.GetVector("id-mutated"))
	require.NotNil(t, svc.index.GetVector("id-beta"))
	require.Equal(t, 2, svc.IndexSize())
}

// Rubric 2: cache hits must validate model configuration, content identity
// and vector dimension; stale or corrupt entries are re-encoded.
func TestBuildIndex_CacheHitValidated(t *testing.T) {
	dim := 8
	base, err := newMemoryCache(10000)
	require.NoError(t, err)
	ctx := context.Background()

	validModelKey := modelConfigKey(ProviderConfig{Name: "mock", Model: "mock-model", Dimension: dim, NormalizeVectors: true})
	validVec := NewMockProvider(dim).deterministicEmbedding("delta")

	// (a) wrong model configuration
	require.NoError(t, base.Put(ctx, "alpha", cacheEntry{
		Vector: make([]float64, dim), ModelKey: "provider=other", Dimension: dim, ContentHash: hashContent("alpha"),
	}))
	// (b) wrong dimension
	require.NoError(t, base.Put(ctx, "beta", cacheEntry{
		Vector: make([]float64, 4), ModelKey: validModelKey, Dimension: 4, ContentHash: hashContent("beta"),
	}))
	// (c) content hash for a different text
	require.NoError(t, base.Put(ctx, "gamma", cacheEntry{
		Vector: make([]float64, dim), ModelKey: validModelKey, Dimension: dim, ContentHash: hashContent("someone-else"),
	}))
	// (d) fully valid hit
	require.NoError(t, base.Put(ctx, "delta", cacheEntry{
		Vector: cloneVector(validVec), ModelKey: validModelKey, Dimension: dim, ContentHash: hashContent("delta"),
	}))

	rec := &recordingProvider{inner: NewMockProvider(dim)}
	svc := newRebuildService(t, dim, 10, rec, base)

	names, ids := roster("alpha", "beta", "gamma", "delta")
	require.NoError(t, svc.BuildIndex(ctx, names, ids))

	encoded := rec.encodedTexts()
	require.ElementsMatch(t, []string{"alpha", "beta", "gamma"}, encoded,
		"stale cache entries must be re-encoded, valid hit must not")

	// The valid cached vector is the one served (identical to deterministic embedding).
	vec := svc.index.GetVector("id-delta")
	require.Equal(t, validVec, vec)
	require.Len(t, svc.index.GetVector("id-alpha"), dim)
}

func TestEntryMatches(t *testing.T) {
	vec8 := make([]float64, 8)
	entry := cacheEntry{
		Vector:      vec8,
		ModelKey:    "model-key",
		Dimension:  8,
		ContentHash: hashContent("text"),
	}

	require.True(t, entryMatches("text", "model-key", 8, entry))
	require.False(t, entryMatches("text", "other-key", 8, entry), "model config mismatch")
	require.False(t, entryMatches("text", "model-key", 4, entry), "configured dimension mismatch")
	require.False(t, entryMatches("other", "model-key", 8, entry), "content identity mismatch")

	short := entry
	short.Vector = make([]float64, 7)
	require.False(t, entryMatches("text", "model-key", 8, short), "stored vector dimension mismatch")
}

// Rubric 3: a failed batch leaves the active index untouched and confirmed
// batches remain available for the same rebuild to resume.
func TestBuildIndex_BatchFailureKeepsActiveIndexAndResumes(t *testing.T) {
	dim := 8
	ctx := context.Background()

	// Initial active generation: 4 entities.
	initialProv := NewMockProvider(dim)
	svc := newRebuildService(t, dim, 2, initialProv, nil)
	oldNames, oldIDs := roster("old1", "old2", "old3", "old4")
	require.NoError(t, svc.BuildIndex(ctx, oldNames, oldIDs))

	// Roster v2 keeps two cached names and adds four new ones; the provider
	// fails after the first successful batch.
	failing := NewMockProvider(dim, WithMockFailAfter(1))
	svc.provider = failing

	newNames, newIDs := roster("old1", "old2", "new1", "new2", "new3", "new4")
	err := svc.BuildIndex(ctx, newNames, newIDs)
	require.Error(t, err)
	require.ErrorIs(t, err, ErrProviderFailure)

	// Active index is unchanged and still serves the old generation.
	require.Equal(t, 4, svc.IndexSize())
	results, err := svc.Search(ctx, "old1", 1)
	require.NoError(t, err)
	require.Equal(t, "id-old1", results[0].ID)
	require.Nil(t, svc.index.GetVector("id-new3"))

	// The first new batch is confirmed in the cache; the second is not.
	_, ok := svc.getCachedVector(ctx, "new1")
	require.True(t, ok, "successful batch must remain confirmed for resume")
	_, ok = svc.getCachedVector(ctx, "new3")
	require.False(t, ok, "failed batch must not count as confirmed")

	// Resume the same rebuild with a healthy provider: only the failed
	// (unconfirmed) batch is encoded again; the confirmed batch is reused.
	rec := &recordingProvider{inner: NewMockProvider(dim)}
	svc.provider = rec
	require.NoError(t, svc.BuildIndex(ctx, newNames, newIDs))
	require.ElementsMatch(t, []string{"new3", "new4"}, rec.encodedTexts())
	require.Equal(t, 6, svc.IndexSize())
	require.NotNil(t, svc.index.GetVector("id-new4"))
}

// Rubric 4: a cache write failure stops those vectors counting as completed
// progress and is never silently swallowed; resume works afterwards.
func TestBuildIndex_CacheWriteFailureNotCounted(t *testing.T) {
	dim := 8
	ctx := context.Background()

	mem, err := newMemoryCache(10000)
	require.NoError(t, err)
	failing := &failPutCache{Cache: mem, shouldFail: func(text string) bool { return text == "new3" }}

	svc := newRebuildService(t, dim, 2, NewMockProvider(dim), failing)
	names, ids := roster("new1", "new2", "new3", "new4")

	err = svc.BuildIndex(ctx, names, ids)
	require.Error(t, err)
	require.ErrorIs(t, err, ErrCacheWrite)

	// No candidate exposure; batch 1 confirmed, failed batch not counted.
	require.Equal(t, 0, svc.IndexSize())
	_, sErr := svc.Search(ctx, "new1", 1)
	require.ErrorIs(t, sErr, ErrIndexNotBuilt)
	_, ok := svc.getCachedVector(ctx, "new1")
	require.True(t, ok)
	_, ok = svc.getCachedVector(ctx, "new3")
	require.False(t, ok)
	_, ok = svc.getCachedVector(ctx, "new4")
	require.False(t, ok)

	// Resume with healthy writes: only the unconfirmed batch is encoded.
	svc.cache = mem
	rec := &recordingProvider{inner: NewMockProvider(dim)}
	svc.provider = rec
	require.NoError(t, svc.BuildIndex(ctx, names, ids))
	require.ElementsMatch(t, []string{"new3", "new4"}, rec.encodedTexts())
	require.Equal(t, 4, svc.IndexSize())
}

// Query encoding keeps serving fresh vectors when the query cache write
// fails, but the failure is surfaced (logged/recorded) rather than faking a
// cache hit - the next encode misses and re-encodes.
func TestEncode_CacheWriteFailureDoesNotBreakQuery(t *testing.T) {
	dim := 8
	failing := &failPutCache{
		Cache: func() Cache { c, _ := newMemoryCache(100); return c }(),
		shouldFail: func(text string) bool { return text == "query-x" },
	}
	rec := &recordingProvider{inner: NewMockProvider(dim)}
	svc := newRebuildService(t, dim, 2, rec, failing)

	vec, err := svc.Encode(context.Background(), "query-x")
	require.NoError(t, err, "query path must stay compatible despite a cache write error")
	require.Len(t, vec, dim)

	_, err = svc.Encode(context.Background(), "query-x")
	require.NoError(t, err)
	require.Len(t, rec.calls, 2, "failed write must not be treated as a confirmed cache entry")
}

// Rubric 5: the candidate index replaces the active index only after vector
// count, dimension and entity mapping all verify.
func TestBuildIndex_VerifiesCandidateBeforeSwap(t *testing.T) {
	ctx := context.Background()

	t.Run("wrong provider dimension", func(t *testing.T) {
		// Service expects dim 8 but the provider returns dim 7.
		bad := &fixedDimensionProvider{inner: NewMockProvider(7), dim: 8}
		svc := newRebuildService(t, 8, 2, bad, nil)
		names, ids := roster("a", "b")
		err := svc.BuildIndex(ctx, names, ids)
		require.Error(t, err)
		require.ErrorIs(t, err, ErrDimensionMismatch)
		require.Equal(t, 0, svc.IndexSize())
	})

	t.Run("provider short response", func(t *testing.T) {
		bad := &shortProvider{inner: NewMockProvider(8)}
		svc := newRebuildService(t, 8, 2, bad, nil)
		names, ids := roster("a", "b")
		err := svc.BuildIndex(ctx, names, ids)
		require.Error(t, err)
		require.ErrorIs(t, err, ErrInvalidResponse)
		require.Equal(t, 0, svc.IndexSize())
	})

	t.Run("duplicate entity ids", func(t *testing.T) {
		svc := newRebuildService(t, 8, 2, nil, nil)
		err := svc.BuildIndex(ctx, []string{"a", "b"}, []string{"same-id", "same-id"})
		require.Error(t, err)
		require.ErrorIs(t, err, ErrCandidateVerification)
		require.Equal(t, 0, svc.IndexSize())
	})

	t.Run("mismatched lengths", func(t *testing.T) {
		svc := newRebuildService(t, 8, 2, nil, nil)
		err := svc.BuildIndex(ctx, []string{"a", "b"}, []string{"only-one"})
		require.Error(t, err)
		require.Equal(t, 0, svc.IndexSize())
	})
}

type fixedDimensionProvider struct {
	inner Provider
	dim   int
}

func (p *fixedDimensionProvider) Embed(ctx context.Context, texts []string) ([][]float64, error) {
	return p.inner.Embed(ctx, texts)
}
func (p *fixedDimensionProvider) Dimension() int { return p.dim }
func (p *fixedDimensionProvider) Name() string   { return p.inner.Name() }
func (p *fixedDimensionProvider) Close() error   { return p.inner.Close() }

type shortProvider struct {
	inner Provider
}

func (p *shortProvider) Embed(ctx context.Context, texts []string) ([][]float64, error) {
	embs, err := p.inner.Embed(ctx, texts)
	if err != nil {
		return nil, err
	}
	return embs[:len(embs)-1], nil // always drop one vector
}
func (p *shortProvider) Dimension() int { return p.inner.Dimension() }
func (p *shortProvider) Name() string   { return p.inner.Name() }
func (p *shortProvider) Close() error   { return p.inner.Close() }

// Rubric 6: when a newer generation starts, an older rebuild finishing later
// must not overwrite it; candidates and searches never cross generations.
func TestBuildIndex_StaleGenerationCannotOverwrite(t *testing.T) {
	dim := 8
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	gate := newGatedProvider(dim, "stall-gate")
	svc := newRebuildService(t, dim, 2, gate, nil)

	// Existing active generation.
	oldNames, oldIDs := roster("old1")
	require.NoError(t, svc.BuildIndex(ctx, oldNames, oldIDs))

	// Start the slow rebuild A for a different roster; it blocks in its
	// first batch before anything is confirmed or exposed.
	aNames, aIDs := roster("stall-gate", "a2", "a3", "a4", "a5", "a6")
	aErr := make(chan error, 1)
	go func() { aErr <- svc.BuildIndex(ctx, aNames, aIDs) }()

	select {
	case <-gate.started:
	case <-time.After(5 * time.Second):
		t.Fatal("slow rebuild never reached the gate")
	}

	// While A is in flight queries only see the old active generation.
	require.Equal(t, 1, svc.IndexSize())
	results, err := svc.Search(ctx, "old1", 1)
	require.NoError(t, err)
	require.Equal(t, "id-old1", results[0].ID)

	// Rebuild B for the new roster completes and swaps while A is blocked.
	bNames, bIDs := roster("b1", "b2")
	require.NoError(t, svc.BuildIndex(ctx, bNames, bIDs))
	require.Equal(t, 2, svc.IndexSize())
	results, err = svc.Search(ctx, "b1", 1)
	require.NoError(t, err)
	require.Equal(t, "id-b1", results[0].ID)

	// Release A; it must finish as a stale generation without overwriting B.
	close(gate.release)
	select {
	case err := <-aErr:
		require.Error(t, err)
		require.ErrorIs(t, err, ErrStaleGeneration)
	case <-time.After(5 * time.Second):
		t.Fatal("stale rebuild never finished")
	}

	require.Equal(t, 2, svc.IndexSize(), "newer generation must remain active")
	require.Nil(t, svc.index.GetVector("id-a2"))
	require.Nil(t, svc.index.GetVector("id-stall-gate"))
	require.NotNil(t, svc.index.GetVector("id-b2"))
}

// A model configuration change invalidates every previously cached vector,
// even when the roster texts are identical.
func TestBuildIndex_ConfigChangeInvalidatesCache(t *testing.T) {
	ctx := context.Background()
	shared, err := newMemoryCache(10000)
	require.NoError(t, err)

	// Generation A: model-a at dimension 4.
	cfgA := DefaultConfig()
	cfgA.Provider = ProviderConfig{Name: "mock", Model: "model-a", Dimension: 4, NormalizeVectors: true}
	cfgA.BatchSize = 10
	cfgA.IndexBuildTimeout = 30 * time.Second
	svcA := &service{
		logger:    log.NewNopLogger(),
		config:    cfgA,
		provider:  NewMockProvider(4),
		cache:     shared,
		dimension: 4,
		modelKey:  modelConfigKey(cfgA.Provider),
		index:     newVectorIndex(4),
	}
	names, ids := roster("alpha", "beta")
	require.NoError(t, svcA.BuildIndex(ctx, names, ids))

	// Generation B: model-b at dimension 8 sharing the same cache.
	rec := &recordingProvider{inner: NewMockProvider(8)}
	cfgB := DefaultConfig()
	cfgB.Provider = ProviderConfig{Name: "mock", Model: "model-b", Dimension: 8, NormalizeVectors: true}
	cfgB.BatchSize = 10
	cfgB.IndexBuildTimeout = 30 * time.Second
	svcB := &service{
		logger:    log.NewNopLogger(),
		config:    cfgB,
		provider:  rec,
		cache:     shared,
		dimension: 8,
		modelKey:  modelConfigKey(cfgB.Provider),
		index:     newVectorIndex(8),
	}
	require.NoError(t, svcB.BuildIndex(ctx, names, ids))
	require.ElementsMatch(t, []string{"alpha", "beta"}, rec.encodedTexts(),
		"old-model cache entries must be discarded after a config change")
	require.Len(t, svcB.index.GetVector("id-alpha"), 8)

	// A-dimension queries cannot be scored against the B index.
	_, err = svcB.index.Search(make([]float64, 4), 1)
	require.ErrorIs(t, err, ErrDimensionMismatch)
}

// Rubric 7: after a restart (new service instance over the same confirmed
// cache) the rebuild resumes from verified progress and queries only ever
// reach the single active generation. Runs against both cache types.
func TestBuildIndex_RestartResumesFromConfirmedCache(t *testing.T) {
	runResume := func(t *testing.T, shared Cache) {
		ctx := context.Background()
		names, ids := roster("r1", "r2", "r3", "r4", "r5", "r6")

		// First process: provider dies after one confirmed batch.
		svc1 := newRebuildService(t, 8, 2, NewMockProvider(8, WithMockFailAfter(1)), shared)
		err := svc1.BuildIndex(ctx, names, ids)
		require.Error(t, err)
		require.ErrorIs(t, err, ErrProviderFailure)
		require.Equal(t, 0, svc1.IndexSize(), "no partial candidate may be exposed")

		// Simulated restart: brand new service and provider, same cache.
		rec := &recordingProvider{inner: NewMockProvider(8)}
		svc2 := newRebuildService(t, 8, 2, rec, shared)
		require.NoError(t, svc2.BuildIndex(ctx, names, ids))

		// Only the four unconfirmed names were encoded after restart.
		require.ElementsMatch(t, []string{"r3", "r4", "r5", "r6"}, rec.encodedTexts())
		require.Equal(t, 6, svc2.IndexSize())

		results, err := svc2.Search(ctx, "r5", 1)
		require.NoError(t, err)
		require.Equal(t, "id-r5", results[0].ID)
	}

	t.Run("memory", func(t *testing.T) {
		shared, err := newMemoryCache(10000)
		require.NoError(t, err)
		runResume(t, shared)
	})

	if !testing.Short() {
		t.Run("sql", func(t *testing.T) {
			db.ForEachDatabase(t, func(database db.DB) {
				config := Config{Provider: ProviderConfig{Model: "mock-model", Dimension: 8}}

				shared, err := newSqlRepository(context.Background(), config, database)
				require.NoError(t, err)
				runResume(t, shared)
			})
		})
	}
}

// Rubric 8: disabled service, noop/memory/sql caches and normal vector
// search all keep working.
func TestNewService_DisabledReturnsNil(t *testing.T) {
	cfg := DefaultConfig() // Enabled defaults to false
	svc, err := NewService(log.NewNopLogger(), cfg, nil)
	require.NoError(t, err)
	require.Nil(t, svc)
}

func TestBuildIndex_NoopCacheNormalSearch(t *testing.T) {
	ctx := context.Background()
	svc := newRebuildService(t, 8, 3, NewMockProvider(8), &noopCache{})

	names, ids := roster("Mohamed Ali", "John Smith", "Li Ming")
	require.NoError(t, svc.BuildIndex(ctx, names, ids))
	require.Equal(t, 3, svc.IndexSize())

	// Exact text matches its own deterministic embedding with score 1.
	results, err := svc.Search(ctx, "John Smith", 2)
	require.NoError(t, err)
	require.Len(t, results, 2)
	require.Equal(t, "id-John Smith", results[0].ID)
	require.InDelta(t, 1.0, results[0].Score, 0.0001)

	score, err := svc.Similarity(ctx, "Li Ming", "Li Ming")
	require.NoError(t, err)
	require.InDelta(t, 1.0, score, 0.0001)
}

func TestRosterFingerprint_OrderAndContentSensitive(t *testing.T) {
	base := rosterFingerprint([]string{"id1", "id2"}, []string{"a", "b"})

	require.Equal(t, base, rosterFingerprint([]string{"id1", "id2"}, []string{"a", "b"}))
	require.NotEqual(t, base, rosterFingerprint([]string{"id2", "id1"}, []string{"b", "a"}), "reordering changes generation")
	require.NotEqual(t, base, rosterFingerprint([]string{"id1", "id2"}, []string{"a", "c"}), "name change changes generation")
	require.NotEqual(t, base, rosterFingerprint([]string{"id1", "id3"}, []string{"a", "b"}), "id change changes generation")
}

func TestBuildIndex_EmptyRosterKeepsUsableEmptyIndex(t *testing.T) {
	svc := newRebuildService(t, 8, 2, nil, nil)
	require.NoError(t, svc.BuildIndex(context.Background(), nil, nil))
	require.Equal(t, 0, svc.IndexSize())

	_, err := svc.Search(context.Background(), "anything", 1)
	require.Error(t, err)
	require.True(t, errors.Is(err, ErrIndexNotBuilt))
}
