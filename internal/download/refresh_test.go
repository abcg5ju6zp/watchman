package download

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/moov-io/base/log"
	"github.com/moov-io/watchman/internal/embeddings"
	"github.com/moov-io/watchman/pkg/search"

	"github.com/stretchr/testify/require"
)

// fakeDownloader implements Downloader for tests.
type fakeDownloader struct {
	stats Stats
	err   error

	started chan struct{}
	release chan struct{}

	mu    sync.Mutex
	calls int
}

func (f *fakeDownloader) RefreshAll(ctx context.Context) (Stats, error) {
	f.mu.Lock()
	f.calls++
	f.mu.Unlock()

	if f.started != nil {
		f.started <- struct{}{}
	}
	if f.release != nil {
		<-f.release
	}
	if f.err != nil {
		return Stats{}, f.err
	}
	return f.stats, nil
}

func (f *fakeDownloader) callCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.calls
}

func okStats() Stats {
	return Stats{
		Lists:         map[string]int{"us_ofac": 3},
		ListHashes:    map[string]string{"us_ofac": "abc"},
		SourceResults: map[string]SourceResult{
			"us_ofac": {Name: "us_ofac", Entities: 3, Hash: "abc", Fresh: true},
		},
	}
}

// fakeCandidate is the text candidate handle used in refresher tests.
type fakeCandidate struct {
	stats Stats
}

func (*fakeCandidate) IsRefreshCandidate() {}

// testPublishedLists is a minimal online text view for refresher tests.
type testPublishedLists struct {
	mu sync.Mutex

	last      Stats
	published []Stats
	prepared  int

	prior map[string]PriorSource
}

func (t *testPublishedLists) Update(s Stats) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.last = s
	t.published = append(t.published, s)
}

func (t *testPublishedLists) PrepareCandidate(s Stats) TextCandidate {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.prepared++
	t.last = s
	return &fakeCandidate{stats: s}
}

func (t *testPublishedLists) PublishCandidate(c TextCandidate) {
	t.mu.Lock()
	defer t.mu.Unlock()
	fc, ok := c.(*fakeCandidate)
	if !ok {
		return
	}
	t.published = append(t.published, fc.stats)
}

func (t *testPublishedLists) PriorSource(source search.SourceList) (PriorSource, bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	p, ok := t.prior[string(source)]
	return p, ok
}

func (t *testPublishedLists) publishCount() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	return len(t.published)
}

// fakeEmbeddings is a controllable embeddingCandidates for tests.
type fakeEmbeddings struct {
	enabled      bool
	buildErr     error
	mu           sync.Mutex
	prepareCalls int
	publishCalls int
	lastSize     int
}

func (f *fakeEmbeddings) EmbeddingsEnabled() bool { return f.enabled }

func (f *fakeEmbeddings) PrepareEmbeddingIndex(ctx context.Context, names, ids []string) (embeddings.IndexCandidate, error) {
	f.mu.Lock()
	f.prepareCalls++
	f.lastSize = len(names)
	f.mu.Unlock()
	if f.buildErr != nil {
		return embeddings.IndexCandidate{}, f.buildErr
	}
	return embeddings.IndexCandidate{}, nil
}

func (f *fakeEmbeddings) PublishEmbeddingIndex(_ embeddings.IndexCandidate) {
	f.mu.Lock()
	f.publishCalls++
	f.mu.Unlock()
}

func newTestRefresher(t *testing.T, dl Downloader, lists publishedLists, es embeddingCandidates) *Refresher {
	t.Helper()
	return NewRefresher(context.Background(), log.NewTestLogger(), dl, lists, es)
}

func TestRefresher_RefreshNowSuccess(t *testing.T) {
	dl := &fakeDownloader{stats: okStats()}
	lists := &testPublishedLists{}
	r := newTestRefresher(t, dl, lists, nil)

	require.Equal(t, StateIdle, r.Status().State)

	err := r.RefreshNow(context.Background(), TriggerStartup)
	require.NoError(t, err)

	st := r.Status()
	require.Equal(t, StateSucceeded, st.State)
	require.Equal(t, TriggerStartup, st.Trigger)
	require.NotNil(t, st.StartedAt)
	require.NotNil(t, st.FinishedAt)
	require.NotEmpty(t, st.Duration)
	require.Empty(t, st.LastError)
	require.NotEmpty(t, st.GenerationID)

	require.Equal(t, 1, lists.prepared)
	require.Equal(t, 1, lists.publishCount())
	require.Equal(t, map[string]int{"us_ofac": 3}, lists.published[0].Lists)

	require.Len(t, st.Sources, 1)
	require.Equal(t, "us_ofac", st.Sources[0].Source)
	require.True(t, st.Sources[0].Fresh)
	require.False(t, st.Sources[0].Stale)
}

func TestRefresher_RefreshNowFailure(t *testing.T) {
	dl := &fakeDownloader{err: errors.New("boom")}
	lists := &testPublishedLists{}
	r := newTestRefresher(t, dl, lists, nil)

	err := r.RefreshNow(context.Background(), TriggerManual)
	require.Error(t, err)

	st := r.Status()
	require.Equal(t, StateFailed, st.State)
	require.Equal(t, "boom", st.LastError)
	require.NotNil(t, st.FinishedAt)

	// Required download failure: the candidate is never prepared or published.
	require.Equal(t, 0, lists.prepared)
	require.Equal(t, 0, lists.publishCount())
	require.Empty(t, st.GenerationID)

	// Slot must be released
	dl.err = nil
	require.NoError(t, r.RefreshNow(context.Background(), TriggerScheduled))
	require.Equal(t, StateSucceeded, r.Status().State)
}

func TestRefresher_ConcurrencyGuard(t *testing.T) {
	dl := &fakeDownloader{
		stats:   okStats(),
		started: make(chan struct{}),
		release: make(chan struct{}),
	}
	r := newTestRefresher(t, dl, &testPublishedLists{}, nil)

	done := make(chan error, 1)
	go func() {
		done <- r.RefreshNow(context.Background(), TriggerScheduled)
	}()

	<-dl.started

	require.Equal(t, StateRunning, r.Status().State)

	require.ErrorIs(t, r.RefreshNow(context.Background(), TriggerManual), ErrAlreadyRunning)
	require.False(t, r.TriggerAsync(TriggerManual))

	close(dl.release)
	require.NoError(t, <-done)

	require.Equal(t, StateSucceeded, r.Status().State)
	require.Equal(t, 1, dl.callCount())
}

func TestRefresher_TriggerAsync(t *testing.T) {
	dl := &fakeDownloader{stats: okStats()}
	r := newTestRefresher(t, dl, &testPublishedLists{}, nil)

	require.True(t, r.TriggerAsync(TriggerManual))

	require.Eventually(t, func() bool {
		return r.Status().State == StateSucceeded
	}, 2*time.Second, 5*time.Millisecond)

	require.Equal(t, TriggerManual, r.Status().Trigger)
}

// TestRefresher_EmbeddingsFailureKeepsPreviousGeneration verifies the core
// atomicity guarantee: when the text candidate is ready but the vector build
// fails, neither text nor vectors move, so the online view keeps serving the
// complete previous generation.
func TestRefresher_EmbeddingsFailureKeepsPreviousGeneration(t *testing.T) {
	dl := &fakeDownloader{stats: okStats()}
	lists := &testPublishedLists{}

	// Seed a "previous" published generation.
	lists.Update(Stats{
		GenerationID: "gen-previous",
		Lists:        map[string]int{"us_ofac": 2},
		ListHashes:   map[string]string{"us_ofac": "old"},
	})

	es := &fakeEmbeddings{enabled: true, buildErr: errors.New("provider unavailable")}
	r := newTestRefresher(t, dl, lists, es)

	err := r.RefreshNow(context.Background(), TriggerScheduled)
	require.Error(t, err)
	require.Contains(t, err.Error(), "building embedding index")

	st := r.Status()
	require.Equal(t, StateFailed, st.State)
	// Online generation id is still the previous one.
	require.Equal(t, "gen-previous", st.GenerationID)

	// Text candidate was prepared but never published.
	require.Equal(t, 1, lists.prepared)
	require.Equal(t, 1, lists.publishCount(), "only the seed generation is published")
	require.Equal(t, "gen-previous", lists.published[0].GenerationID)

	// The vector candidate was built but never published.
	require.Equal(t, 1, es.prepareCalls)
	require.Equal(t, 0, es.publishCalls)

	// A later successful refresh (provider recovered) publishes one generation.
	es.buildErr = nil
	require.NoError(t, r.RefreshNow(context.Background(), TriggerScheduled))
	require.Equal(t, 1, es.publishCalls)
	require.Equal(t, 2, lists.publishCount())
	require.Equal(t, StateSucceeded, r.Status().State)
}

// TestRefresher_EmbeddingsDisabledPublishesText verifies backward
// compatibility: without embeddings the text candidate publishes on its own.
func TestRefresher_EmbeddingsDisabledPublishesText(t *testing.T) {
	dl := &fakeDownloader{stats: okStats()}
	lists := &testPublishedLists{}
	es := &fakeEmbeddings{enabled: false}
	r := newTestRefresher(t, dl, lists, es)

	require.NoError(t, r.RefreshNow(context.Background(), TriggerStartup))
	require.Equal(t, 1, lists.publishCount())
	require.Equal(t, 0, es.prepareCalls)
	require.Equal(t, 0, es.publishCalls)
}
