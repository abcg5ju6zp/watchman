package embeddings

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"

	"github.com/moov-io/watchman/internal/db"
)

// cacheEntry is a single persisted embedding together with the verifiable
// identity of how it was produced. A hit is only usable when the entry's
// model configuration, content hash and vector dimension all match the
// current request; entries left behind by a different model or an older
// roster are treated as misses instead of being silently reused.
type cacheEntry struct {
	// Vector is the L2-normalized embedding.
	Vector []float64

	// ModelKey is the fingerprint of the provider configuration that
	// produced the vector (provider, endpoint, model, normalization).
	ModelKey string

	// Dimension is the configured dimension when the vector was stored.
	Dimension int

	// ContentHash is the sha256 hash of the full input text, so that an
	// entry can never be served for a different input (this also covers
	// truncation or same-prefix collisions in the storage key).
	ContentHash string
}

// Cache stores embedding vectors between encodes and index rebuilds.
//
// Put must return an error when the entry could not be durably stored.
// Callers running an index rebuild treat a Put failure as a failed batch:
// the vectors are not counted as confirmed progress and the active index
// is left untouched.
type Cache interface {
	Get(ctx context.Context, text string) (cacheEntry, bool)
	Put(ctx context.Context, text string, entry cacheEntry) error
}

func NewCache(ctx context.Context, config Config, database db.DB) (Cache, error) {
	switch strings.ToLower(config.Cache.Type) {
	case "", "none":
		return &noopCache{}, nil

	case "memory":
		return newMemoryCache(config.Cache.Size)

	case "sql", "mysql", "postgres":
		return newSqlRepository(ctx, config, database)
	}

	return nil, fmt.Errorf("embeddings: unknown cache type %q", config.Cache.Type)
}

type noopCache struct{}

var _ Cache = (&noopCache{})

func (c *noopCache) Get(_ context.Context, _ string) (cacheEntry, bool) {
	return cacheEntry{}, false
}

func (c *noopCache) Put(_ context.Context, _ string, _ cacheEntry) error {
	return nil
}

// hashContent returns the hex sha256 of the full input text.
func hashContent(text string) string {
	sum := sha256.Sum256([]byte(text))
	return hex.EncodeToString(sum[:])
}

// modelConfigKey fingerprints the provider configuration that determines the
// geometry of the vectors. A different provider, endpoint, model, dimension
// or normalization setting yields a different key, so cached vectors from a
// previous configuration can never validate against a new one.
func modelConfigKey(cfg ProviderConfig) string {
	return fmt.Sprintf("provider=%s|base_url=%s|model=%s|dim=%d|normalize=%t",
		strings.ToLower(strings.TrimSpace(cfg.Name)),
		strings.TrimSpace(cfg.BaseURL),
		cfg.Model,
		cfg.Dimension,
		cfg.NormalizeVectors)
}

// entryMatches verifies that a cached entry is valid for the given text and
// fixed model configuration. Every field is checked independently:
// model configuration, content identity and vector dimension.
func entryMatches(text, modelKey string, dimension int, entry cacheEntry) bool {
	return entry.ModelKey == modelKey &&
		entry.Dimension == dimension &&
		len(entry.Vector) == dimension &&
		entry.ContentHash == hashContent(text)
}

// cloneVector returns an independent copy of a vector.
func cloneVector(vec []float64) []float64 {
	out := make([]float64, len(vec))
	copy(out, vec)
	return out
}
