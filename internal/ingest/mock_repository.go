package ingest

import (
	"context"
	"fmt"
	"slices"
	"sort"
	"sync"

	"github.com/moov-io/watchman/pkg/search"
)

// mockSourceState is one source's committed generation.
type mockSourceState struct {
	revision int64
	version  string
	entities []search.Entity[search.Value]
	pending  bool
}

type MockRepository struct {
	Err error

	mu       sync.RWMutex
	entities map[string]*mockSourceState // keyed by fileType/source
}

var _ Repository = (&MockRepository{})

func (r *MockRepository) Upsert(ctx context.Context, fileType string, entities []search.Entity[search.Value]) error {
	if r.Err != nil {
		return r.Err
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	if r.entities == nil {
		r.entities = make(map[string]*mockSourceState)
	}

	if len(entities) == 0 {
		delete(r.entities, fileType)
		return nil
	}

	cloned := cloneSourceEntities(fileType, entities)
	state, ok := r.entities[fileType]
	if !ok {
		state = &mockSourceState{}
		r.entities[fileType] = state
	}
	version, err := EntityVersion(cloned)
	if err != nil {
		return err
	}
	// Legacy Upsert semantics: every non-empty call is a fresh replacement.
	state.revision++
	state.version = version
	state.entities = cloned
	state.pending = false
	return nil
}

func (r *MockRepository) StageVersion(ctx context.Context, fileType string, entities []search.Entity[search.Value]) (SourceChecksum, error) {
	if r.Err != nil {
		return SourceChecksum{}, r.Err
	}
	if len(entities) == 0 {
		return SourceChecksum{}, fmt.Errorf("refusing to stage %s: no entities", fileType)
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	if r.entities == nil {
		r.entities = make(map[string]*mockSourceState)
	}

	cloned := cloneSourceEntities(fileType, entities)
	version, err := EntityVersion(cloned)
	if err != nil {
		return SourceChecksum{}, err
	}

	state, ok := r.entities[fileType]
	if !ok {
		state = &mockSourceState{}
		r.entities[fileType] = state
	}

	// New content identity only: identical content keeps the revision and
	// rows so a resubmit never creates a new visible version.
	if state.version != version {
		state.revision++
		state.version = version
		state.entities = cloned
	}
	state.pending = true

	return SourceChecksum{
		Source:      fileType,
		Revision:    state.revision,
		Version:     version,
		EntityCount: len(state.entities),
		Checksum:    version,
		Published:   false,
	}, nil
}

func (r *MockRepository) CurrentVersion(ctx context.Context, source string) (SourceChecksum, error) {
	if r.Err != nil {
		return SourceChecksum{}, r.Err
	}

	r.mu.RLock()
	defer r.mu.RUnlock()

	state, ok := r.entities[source]
	if !ok {
		return SourceChecksum{}, ErrSourceVersionNotFound
	}
	return mockChecksum(source, state), nil
}

func (r *MockRepository) MarkPublished(ctx context.Context, source string, revision int64) (bool, error) {
	if r.Err != nil {
		return false, r.Err
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	state, ok := r.entities[source]
	if !ok || state.revision != revision {
		return false, nil
	}
	state.pending = false
	return true, nil
}

func (r *MockRepository) Get(ctx context.Context, sourceID string, source search.SourceList) (*search.Entity[search.Value], error) {
	if r.Err != nil {
		return nil, r.Err
	}

	r.mu.RLock()
	defer r.mu.RUnlock()

	state := r.entities[string(source)]
	if state == nil {
		return nil, nil
	}

	entities := state.entities
	for i := range entities {
		if entities[i].SourceID == sourceID {
			return &entities[i], nil
		}
	}

	return nil, nil
}

func (r *MockRepository) ListBySource(ctx context.Context, lastSourceID string, source search.SourceList, limit int) ([]search.Entity[search.Value], error) {
	if r.Err != nil {
		return nil, r.Err
	}

	r.mu.RLock()
	defer r.mu.RUnlock()

	state := r.entities[string(source)]
	if state == nil {
		return nil, nil
	}
	entities := state.entities

	// Handle pagination
	startIdx := 0
	if lastSourceID != "" {
		for i, e := range entities {
			if e.SourceID == lastSourceID {
				startIdx = i + 1
				break
			}
		}
	}

	if startIdx >= len(entities) {
		return nil, nil
	}

	endIdx := startIdx + limit
	if endIdx > len(entities) {
		endIdx = len(entities)
	}

	return slices.Clone(entities[startIdx:endIdx]), nil
}

func (r *MockRepository) ListAll(ctx context.Context) ([]search.Entity[search.Value], error) {
	if r.Err != nil {
		return nil, r.Err
	}

	r.mu.RLock()
	defer r.mu.RUnlock()

	var all []search.Entity[search.Value]
	for _, state := range r.entities {
		all = append(all, state.entities...)
	}
	sort.Slice(all, func(i, j int) bool {
		if all[i].Source != all[j].Source {
			return all[i].Source < all[j].Source
		}
		return all[i].SourceID < all[j].SourceID
	})
	return slices.Clone(all), nil
}

func (r *MockRepository) Checksums(ctx context.Context) ([]SourceChecksum, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	r.mu.RLock()
	defer r.mu.RUnlock()

	sources := make([]string, 0, len(r.entities))
	for src := range r.entities {
		sources = append(sources, src)
	}
	sort.Strings(sources)

	out := make([]SourceChecksum, 0, len(sources))
	for _, src := range sources {
		out = append(out, mockChecksum(src, r.entities[src]))
	}
	return out, nil
}

func cloneSourceEntities(fileType string, entities []search.Entity[search.Value]) []search.Entity[search.Value] {
	cloned := slices.Clone(entities)
	for i := range cloned {
		cloned[i].Source = search.SourceList(fileType)
	}
	return cloned
}

func mockChecksum(source string, state *mockSourceState) SourceChecksum {
	sum, err := checksumEntities(state.entities)
	if err != nil {
		panic(err)
	}
	return SourceChecksum{
		Source:      source,
		Revision:    state.revision,
		Version:     state.version,
		EntityCount: len(state.entities),
		Checksum:    sum,
		Published:   !state.pending,
	}
}
