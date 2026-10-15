package embeddings

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"sync"

	"github.com/moov-io/watchman/internal/db"

	"github.com/moov-io/base/log"
	"github.com/moov-io/base/telemetry"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

// SearchResult represents a single search result from embedding similarity search.
type SearchResult struct {
	ID    string  // Entity ID from the sanctions list
	Name  string  // Original name that was indexed
	Score float64 // Cosine similarity score (0.0 to 1.0)
}

// Service provides semantic embedding-based name matching.
// It uses neural network embeddings to find similar names across different scripts
// (Arabic, Cyrillic, Chinese, etc.) and Latin text.
type Service interface {
	// Encode converts text to a normalized embedding vector.
	// The returned vector has dimensions matching the configured provider.
	Encode(ctx context.Context, text string) ([]float64, error)

	// EncodeBatch encodes multiple texts efficiently in a single API call.
	EncodeBatch(ctx context.Context, texts []string) ([][]float64, error)

	// BuildIndex creates a searchable index from entity names.
	// This must be called before Search() can be used.
	// ids and names must have the same length.
	BuildIndex(ctx context.Context, names []string, ids []string) error

	// Search finds similar names using vector similarity.
	// Returns up to k results sorted by similarity score (highest first).
	Search(ctx context.Context, query string, k int) ([]SearchResult, error)

	// Similarity computes cosine similarity between two texts.
	// Returns a value between 0.0 (dissimilar) and 1.0 (identical).
	Similarity(ctx context.Context, text1, text2 string) (float64, error)

	// ShouldUseEmbeddings determines if a query should use embeddings vs Jaro-Winkler.
	// When CrossScriptOnly is enabled, returns true only for non-Latin queries.
	ShouldUseEmbeddings(query string) bool

	// IndexSize returns the number of items in the search index.
	IndexSize() int

	// Shutdown releases resources held by the service.
	Shutdown()
}

// generation identifies one fixed roster of entities encoded with one fixed
// model configuration. Queries and the active index always belong to exactly
// one generation; a candidate built for any other generation can never be
// searched or swapped in after a newer generation has started.
type generation struct {
	// rosterHash fingerprints the ordered (id, name) pairs of the roster.
	rosterHash string

	// modelKey fingerprints the provider configuration (model, dimension, ...).
	modelKey string

	// dimension is the single vector dimension valid for this generation.
	dimension int
}

// service implements the Service interface.
type service struct {
	logger log.Logger
	config Config

	provider Provider
	cache    Cache

	// dimension and modelKey are fixed for the lifetime of the service and
	// validated on every cache hit, so vectors produced under a different
	// configuration can never enter an index.
	dimension int
	modelKey  string

	mu        sync.RWMutex // guards index, activeGen and buildSeq
	index     *vectorIndex // exactly one active generation, swapped atomically
	activeGen generation
	buildSeq  uint64 // monotonic rebuild token; only the newest build may swap
}

// NewService creates a new embeddings service.
// Returns nil if config.Enabled is false.
func NewService(logger log.Logger, config Config, database db.DB) (Service, error) {
	// Apply environment variable overrides
	config.LoadFromEnv()

	if !config.Enabled {
		return nil, nil
	}

	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("embeddings: invalid config: %w", err)
	}

	ctx, span := telemetry.StartSpan(context.Background(), "embeddings-setup", trace.WithAttributes(
		attribute.String("provider", config.Provider.Name),
		attribute.String("model", config.Provider.Model),
		attribute.Int("dimension", config.Provider.Dimension),
		attribute.String("cache_type", config.Cache.Type),
		attribute.Int("cache_size", config.Cache.Size),
		attribute.Bool("cross_script_only", config.CrossScriptOnly),
	))
	defer span.End()

	// Create provider based on configuration
	provider, err := createProvider(config.Provider)
	if err != nil {
		return nil, fmt.Errorf("embeddings: failed to create provider: %w", err)
	}

	logger.Info().Logf("embeddings: using %s provider", provider.Name())

	// Create cache
	cache, err := NewCache(ctx, config, database)
	if err != nil {
		return nil, fmt.Errorf("embeddings: failed to create cache: %w", err)
	}

	logger.Info().Logf("embeddings: service initialized (provider=%s, dimension=%d, %s_cache_size=%d, cross_script_only=%v)",
		provider.Name(), provider.Dimension(), config.Cache.Type, config.Cache.Size, config.CrossScriptOnly)

	return &service{
		logger:    logger,
		config:    config,
		provider:  provider,
		cache:     cache,
		dimension: provider.Dimension(),
		modelKey:  modelConfigKey(config.Provider),
		index:     newVectorIndex(provider.Dimension()),
	}, nil
}

// getCachedVector returns a vector only when the cache entry verifies against
// the fixed model configuration, the requested content and the dimension.
func (s *service) getCachedVector(ctx context.Context, text string) ([]float64, bool) {
	entry, ok := s.cache.Get(ctx, text)
	if !ok || !entryMatches(text, s.modelKey, s.dimension, entry) {
		return nil, false
	}
	return entry.Vector, true
}

// putCachedVector persists a freshly encoded vector with the verifiable
// identity of the current model configuration and content.
func (s *service) putCachedVector(ctx context.Context, text string, embedding []float64) error {
	return s.cache.Put(ctx, text, cacheEntry{
		Vector:      cloneVector(embedding),
		ModelKey:    s.modelKey,
		Dimension:  s.dimension,
		ContentHash: hashContent(text),
	})
}

// Encode converts text to a normalized embedding vector.
func (s *service) Encode(ctx context.Context, text string) ([]float64, error) {
	ctx, span := telemetry.StartSpan(ctx, "embeddings-encode")
	defer span.End()

	// Check cache first - every hit is verified against the current model
	// configuration, content identity and dimension.
	if emb, ok := s.getCachedVector(ctx, text); ok {
		span.SetAttributes(attribute.Bool("cache_hit", true))
		return emb, nil
	}

	span.SetAttributes(attribute.Bool("cache_hit", false))

	// Encode with provider
	embeddings, err := s.provider.Embed(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	if len(embeddings) == 0 {
		return nil, ErrInvalidResponse
	}

	// Validate dimension
	if len(embeddings[0]) != s.dimension {
		return nil, fmt.Errorf("%w: got %d, expected %d",
			ErrDimensionMismatch, len(embeddings[0]), s.dimension)
	}

	// Cache the result. A failed query-cache write does not break the
	// request, but it is logged and recorded rather than swallowed; the
	// vector was simply not durably confirmed.
	if err := s.putCachedVector(ctx, text, embeddings[0]); err != nil {
		span.RecordError(err)
		s.logger.Error().Logf("embeddings: failed to cache encoded query: %v", err)
	}

	return embeddings[0], nil
}

// EncodeBatch encodes multiple texts efficiently.
func (s *service) EncodeBatch(ctx context.Context, texts []string) ([][]float64, error) {
	_, span := telemetry.StartSpan(ctx, "embeddings-encode-batch", trace.WithAttributes(
		attribute.Int("batch_size", len(texts)),
	))
	defer span.End()

	// Check cache for all texts
	result := make([][]float64, len(texts))
	uncachedIndices := make([]int, 0, len(texts))
	uncachedTexts := make([]string, 0, len(texts))

	for i, text := range texts {
		if emb, ok := s.getCachedVector(ctx, text); ok {
			result[i] = emb
		} else {
			uncachedIndices = append(uncachedIndices, i)
			uncachedTexts = append(uncachedTexts, text)
		}
	}

	span.SetAttributes(
		attribute.Int("cache_hits", len(texts)-len(uncachedTexts)),
		attribute.Int("cache_misses", len(uncachedTexts)),
	)

	// Encode uncached texts
	if len(uncachedTexts) > 0 {
		embeddings, err := s.provider.Embed(ctx, uncachedTexts)
		if err != nil {
			return nil, err
		}
		if len(embeddings) != len(uncachedTexts) {
			return nil, fmt.Errorf("%w: got %d embeddings for %d texts",
				ErrInvalidResponse, len(embeddings), len(uncachedTexts))
		}

		// Validate and fill in results and cache
		for i, idx := range uncachedIndices {
			if len(embeddings[i]) != s.dimension {
				return nil, fmt.Errorf("%w: text %q embedding got %d, expected %d",
					ErrDimensionMismatch, uncachedTexts[i], len(embeddings[i]), s.dimension)
			}
			result[idx] = embeddings[i]
			if err := s.putCachedVector(ctx, uncachedTexts[i], embeddings[i]); err != nil {
				// Non-fatal for ad-hoc query encoding, but never silent.
				span.RecordError(err)
				s.logger.Error().Logf("embeddings: failed to cache encoded batch text: %v", err)
			}
		}
	}

	return result, nil
}

// rosterFingerprint fingerprints the exact ordered pairing of entity IDs and
// names. Order matters and lengths are framed, so reordering, renaming or a
// roster change produces a different generation.
func rosterFingerprint(ids, names []string) string {
	h := sha256.New()
	var frame [8]byte
	write := func(value string) {
		binary.BigEndian.PutUint64(frame[:], uint64(len(value)))
		h.Write(frame[:])
		h.Write([]byte(value))
	}
	for i := range ids {
		write(ids[i])
		write(names[i])
	}
	return hex.EncodeToString(h.Sum(nil))
}

// BuildIndex creates a searchable index from entity names.
//
// The rebuild fixes the model, dimension and the exact ordered roster up
// front and assembles a fully verified candidate index offline. Provider or
// cache-write failures leave the active index untouched while confirmed
// batches remain in the cache, so retrying BuildIndex (including after a
// restart with a persistent cache) resumes the same rebuild. The candidate
// replaces the active index in a single swap only after vector count,
// dimensions and entity mapping all verify, and only if no newer rebuild has
// started in the meantime.
func (s *service) BuildIndex(ctx context.Context, names []string, ids []string) error {
	// Apply timeout from config
	var cancel context.CancelFunc
	if s.config.IndexBuildTimeout > 0 {
		ctx, cancel = context.WithTimeout(ctx, s.config.IndexBuildTimeout)
		defer cancel()
	}

	ctx, span := telemetry.StartSpan(ctx, "embeddings-build-index", trace.WithAttributes(
		attribute.Int("entity_count", len(names)),
	))
	defer span.End()

	if len(names) != len(ids) {
		return fmt.Errorf("embeddings: names and ids must have same length")
	}

	// 1. Snapshot the roster and fix the model configuration for the whole
	// rebuild. Caller mutations after this point cannot mix into the
	// candidate, and every vector below must match this generation.
	snapNames := append([]string(nil), names...)
	snapIDs := append([]string(nil), ids...)

	dimension := s.dimension
	gen := generation{
		rosterHash: rosterFingerprint(snapIDs, snapNames),
		modelKey:   s.modelKey,
		dimension:  dimension,
	}
	span.SetAttributes(attribute.String("generation", gen.rosterHash))

	// Claim a rebuild token. A build that is no longer the newest by the
	// time its candidate is ready must discard it.
	s.mu.Lock()
	s.buildSeq++
	myBuildSeq := s.buildSeq
	s.mu.Unlock()

	s.logger.Info().Logf("embeddings: building index for %d entities (generation %s)", len(snapNames), gen.rosterHash)

	// 2. Assemble candidate vectors positionally, resuming from confirmed
	// cache entries. Hits are validated for model, content and dimension;
	// anything else is (re)encoded.
	vectors := make([][]float64, len(snapNames))
	missingIndices := make([]int, 0, len(snapNames))
	missingNames := make([]string, 0, len(snapNames))

	for i, name := range snapNames {
		if emb, ok := s.getCachedVector(ctx, name); ok {
			vectors[i] = emb
			continue
		}
		missingIndices = append(missingIndices, i)
		missingNames = append(missingNames, name)
	}

	cacheHits := len(snapNames) - len(missingNames)
	span.SetAttributes(
		attribute.Int("cache_hits", cacheHits),
		attribute.Int("cache_misses", len(missingNames)),
	)

	if cacheHits > 0 {
		s.logger.Info().Logf("embeddings: cache hits: %d/%d (%.1f%%), resuming confirmed progress",
			cacheHits, len(snapNames), float64(cacheHits)/float64(len(snapNames))*100)
	}

	// 3. Encode missing names in fixed roster order. A batch is confirmed
	// only after every vector passes validation and its cache write
	// succeeds; on failure the active index is untouched and the vectors
	// are not counted as completed progress.
	batchSize := s.config.BatchSize
	for i := 0; i < len(missingNames); i += batchSize {
		select {
		case <-ctx.Done():
			return fmt.Errorf("embeddings: build index cancelled: %w", ctx.Err())
		default:
		}

		end := i + batchSize
		if end > len(missingNames) {
			end = len(missingNames)
		}

		batch := missingNames[i:end]
		batchEmbeddings, err := s.provider.Embed(ctx, batch)
		if err != nil {
			return fmt.Errorf("embeddings: failed to encode batch %d: %w", i/batchSize, err)
		}
		if len(batchEmbeddings) != len(batch) {
			return fmt.Errorf("%w: got %d embeddings for %d texts in batch %d",
				ErrInvalidResponse, len(batchEmbeddings), len(batch), i/batchSize)
		}
		for j, emb := range batchEmbeddings {
			if len(emb) != dimension {
				return fmt.Errorf("%w: %q got %d, expected %d in batch %d",
					ErrDimensionMismatch, batch[j], len(emb), dimension, i/batchSize)
			}
		}

		// Persist before counting the batch as confirmed progress.
		for j, emb := range batchEmbeddings {
			name := batch[j]
			if err := s.putCachedVector(ctx, name, emb); err != nil {
				return fmt.Errorf("%w: batch %d entity at position %d: %w",
					ErrCacheWrite, i/batchSize, missingIndices[i+j], err)
			}
			vectors[missingIndices[i+j]] = emb
		}

		if len(missingNames) > 1000 && (i+batchSize)%(batchSize*10) == 0 {
			s.logger.Info().Logf("embeddings: encoded %d/%d uncached entities", i+batchSize, len(missingNames))
		}
	}

	// 4. Build and verify the complete candidate while the active index
	// keeps serving the previous generation. The candidate is never
	// reachable through Search or IndexSize until the swap below.
	candidate, err := newVectorIndexFromVectors(dimension, vectors, snapIDs, snapNames)
	if err != nil {
		return err
	}

	// 5. Swap only if this is still the newest rebuild. A rebuild for an
	// older roster or configuration that finishes later is discarded
	// instead of overwriting the newer active generation.
	s.mu.Lock()
	if myBuildSeq != s.buildSeq {
		s.mu.Unlock()
		span.SetAttributes(attribute.Bool("stale_generation", true))
		return fmt.Errorf("%w: completed generation %s after generation %d started",
			ErrStaleGeneration, gen.rosterHash, s.buildSeq)
	}
	s.index = candidate
	s.activeGen = gen
	s.mu.Unlock()

	s.logger.Info().Logf("embeddings: index built with %d entities generation %s (%d from cache, %d newly encoded)",
		len(snapNames), gen.rosterHash, cacheHits, len(missingNames))

	return nil
}

// Search finds similar names using vector similarity.
func (s *service) Search(ctx context.Context, query string, k int) ([]SearchResult, error) {
	_, span := telemetry.StartSpan(ctx, "embeddings-search", trace.WithAttributes(
		attribute.String("query", query),
		attribute.Int("k", k),
	))
	defer span.End()

	// Encode query BEFORE acquiring lock to avoid blocking index during inference
	queryEmb, err := s.Encode(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("embeddings: failed to encode query: %w", err)
	}

	// Capture the single active generation under the lock. Only this index
	// is ever searched; candidates in flight or discarded generations are
	// not reachable.
	s.mu.RLock()
	idx := s.index
	activeGen := s.activeGen
	s.mu.RUnlock()

	if idx.Size() == 0 {
		return nil, ErrIndexNotBuilt
	}

	// A query encoded under a different model generation can never be
	// scored against this index.
	if len(queryEmb) != activeGen.dimension {
		return nil, fmt.Errorf("%w: query dimension %d, active index dimension %d",
			ErrDimensionMismatch, len(queryEmb), activeGen.dimension)
	}

	// Search index
	results, err := idx.Search(queryEmb, k)
	if err != nil {
		return nil, err
	}

	span.SetAttributes(attribute.Int("results_count", len(results)))

	return results, nil
}

// Similarity computes cosine similarity between two texts.
func (s *service) Similarity(ctx context.Context, text1, text2 string) (float64, error) {
	_, span := telemetry.StartSpan(ctx, "embeddings-similarity")
	defer span.End()

	// Encode both texts
	emb1, err := s.Encode(ctx, text1)
	if err != nil {
		return 0, err
	}

	emb2, err := s.Encode(ctx, text2)
	if err != nil {
		return 0, err
	}

	// Compute cosine similarity (vectors are L2-normalized, so dot product = cosine)
	return dotProduct(emb1, emb2), nil
}

// ShouldUseEmbeddings determines if a query should use embeddings.
func (s *service) ShouldUseEmbeddings(query string) bool {
	if !s.config.CrossScriptOnly {
		return true // Always use embeddings
	}
	return IsNonLatin(query)
}

// IndexSize returns the number of items in the search index.
func (s *service) IndexSize() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.index.Size()
}

// Shutdown releases resources held by the service.
func (s *service) Shutdown() {
	s.logger.Info().Log("embeddings: shutting down")
	if s.provider != nil {
		s.provider.Close()
	}
}
