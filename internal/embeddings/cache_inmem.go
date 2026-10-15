package embeddings

import (
	"context"
	"sync"

	lru "github.com/hashicorp/golang-lru/v2"
)

// memoryCache provides an LRU cache for embedding vectors.
// This reduces inference latency for repeated queries.
//
// Entries carry their full cacheEntry identity (model configuration,
// dimension and content hash); the service re-validates every hit before
// use so vectors from an older configuration can never be reused.
type memoryCache struct {
	mu       sync.RWMutex
	cache    *lru.Cache[string, cacheEntry]
	capacity int
}

var _ Cache = (*memoryCache)(nil)

// newMemoryCache creates a new embedding cache with the specified size.
func newMemoryCache(size int) (*memoryCache, error) {
	if size <= 0 {
		size = 1000 // Default size
	}

	c, err := lru.New[string, cacheEntry](size)
	if err != nil {
		return nil, err
	}

	return &memoryCache{
		cache:    c,
		capacity: size,
	}, nil
}

// Get retrieves an entry from the cache.
// Returns the entry and true if found, the zero value and false otherwise.
func (c *memoryCache) Get(_ context.Context, text string) (cacheEntry, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	entry, ok := c.cache.Get(text)
	if !ok {
		return cacheEntry{}, false
	}
	// Defensive copy: concurrent rebuilds and queries share the cache.
	entry.Vector = cloneVector(entry.Vector)
	return entry, true
}

// Put adds an entry to the cache. It never fails for the in-memory cache.
func (c *memoryCache) Put(_ context.Context, text string, entry cacheEntry) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Make a copy to avoid external modifications
	entry.Vector = cloneVector(entry.Vector)
	c.cache.Add(text, entry)
	return nil
}

// Size returns the current number of items in the cache.
func (c *memoryCache) Size() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.cache.Len()
}

// Clear removes all items from the cache.
func (c *memoryCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.cache.Purge()
}

// CacheStats returns cache statistics.
type CacheStats struct {
	Size     int
	Capacity int
}

// Stats returns current cache statistics.
func (c *memoryCache) Stats() CacheStats {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return CacheStats{
		Size:     c.cache.Len(),
		Capacity: c.capacity,
	}
}
