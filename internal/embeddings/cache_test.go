package embeddings

import (
	"context"
	"testing"

	"github.com/moov-io/watchman/internal/db"

	"github.com/stretchr/testify/require"
)

// putCacheVector stores a vector as a fully identified cache entry.
func putCacheVector(t *testing.T, c Cache, text string, embedding []float64) {
	t.Helper()
	err := c.Put(context.Background(), text, cacheEntry{
		Vector:      cloneVector(embedding),
		ModelKey:    "test-model",
		Dimension:   len(embedding),
		ContentHash: hashContent(text),
	})
	require.NoError(t, err)
}

func TestEmbeddingCache_PutGet(t *testing.T) {
	forEachCache(t, 100, func(c Cache) {
		ctx := context.Background()

		embedding := []float64{0.1, 0.2, 0.3}
		putCacheVector(t, c, "test", embedding)

		entry, ok := c.Get(ctx, "test")
		require.True(t, ok)
		require.Equal(t, embedding, entry.Vector)
		require.Equal(t, "test-model", entry.ModelKey)
		require.Equal(t, 3, entry.Dimension)
		require.Equal(t, hashContent("test"), entry.ContentHash)
	})
}

func TestEmbeddingCache_Miss(t *testing.T) {
	forEachCache(t, 100, func(c Cache) {
		ctx := context.Background()

		_, ok := c.Get(ctx, "nonexistent")
		require.False(t, ok)
	})
}

func TestEmbeddingCache_Memory_Eviction(t *testing.T) {
	ctx := context.Background()

	c, err := newMemoryCache(2)
	require.NoError(t, err)

	putCacheVector(t, c, "a", []float64{1})
	putCacheVector(t, c, "b", []float64{2})
	putCacheVector(t, c, "c", []float64{3}) // Should evict "a"

	_, ok := c.Get(ctx, "a")
	require.False(t, ok, "Expected 'a' to be evicted")

	_, ok = c.Get(ctx, "b")
	require.True(t, ok)

	_, ok = c.Get(ctx, "c")
	require.True(t, ok)
}

func TestEmbeddingCache_Memory_Size(t *testing.T) {
	cache, err := newMemoryCache(100)
	require.NoError(t, err)

	require.Equal(t, 0, cache.Size())

	putCacheVector(t, cache, "a", []float64{1})
	require.Equal(t, 1, cache.Size())

	putCacheVector(t, cache, "b", []float64{2})
	require.Equal(t, 2, cache.Size())
}

func TestEmbeddingCache_Memory_Clear(t *testing.T) {
	cache, err := newMemoryCache(100)
	require.NoError(t, err)

	putCacheVector(t, cache, "a", []float64{1})
	putCacheVector(t, cache, "b", []float64{2})
	require.Equal(t, 2, cache.Size())

	cache.Clear()
	require.Equal(t, 0, cache.Size())
}

func TestEmbeddingCache_CopyOnPut(t *testing.T) {
	forEachCache(t, 100, func(c Cache) {
		ctx := context.Background()

		original := []float64{1, 2, 3}
		putCacheVector(t, c, "test", original)

		// Modify original
		original[0] = 999

		// Retrieved should be unchanged
		entry, ok := c.Get(ctx, "test")
		require.True(t, ok)
		require.Equal(t, float64(1), entry.Vector[0], "Cache should store a copy, not a reference")
	})
}

func TestEmbeddingCache_Memory_DefaultSize(t *testing.T) {
	// Size 0 should use default
	cache, err := newMemoryCache(100)
	require.NoError(t, err)
	require.NotNil(t, cache)
}

func TestEmbeddingCache_Stats(t *testing.T) {
	cache, err := newMemoryCache(100)
	require.NoError(t, err)

	putCacheVector(t, cache, "a", []float64{1})
	putCacheVector(t, cache, "b", []float64{2})

	stats := cache.Stats()
	require.Equal(t, 2, stats.Size)
}

func forEachCache(t *testing.T, size int, fn func(c Cache)) {
	t.Helper()

	t.Run("memory", func(t *testing.T) {
		// memory
		mem, err := newMemoryCache(size)
		require.NoError(t, err)
		require.NotNil(t, mem)

		fn(mem)
	})

	if !testing.Short() {
		t.Run("database", func(t *testing.T) {
			db.ForEachDatabase(t, func(database db.DB) {
				config := Config{
					Provider: ProviderConfig{
						Model:     "testing",
						Dimension: 123,
					},
				}

				cc, err := newSqlRepository(t.Context(), config, database)
				require.NoError(t, err)

				fn(cc)
			})
		})
	}
}
