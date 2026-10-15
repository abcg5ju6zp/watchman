package download

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/moov-io/base/log"
	"github.com/moov-io/watchman/pkg/search"

	"github.com/stretchr/testify/require"
)

// fakePrior serves a previous generation for fallback tests.
type fakePrior struct {
	sources map[string]PriorSource
}

func (f *fakePrior) PriorSource(source search.SourceList) (PriorSource, bool) {
	p, ok := f.sources[string(source)]
	return p, ok
}

func TestApplyLoadFailures_RequiredFailureRecorded(t *testing.T) {
	stats := Stats{
		Lists:         map[string]int{},
		ListHashes:    map[string]string{},
		SourceResults: map[string]SourceResult{},
	}
	applyLoadFailures(log.NewTestLogger(), nil, stats, []loadFailure{
		{source: search.SourceUSOFAC, err: errSentinel("origin down"), allowed: false},
	})

	res := stats.SourceResults[string(search.SourceUSOFAC)]
	require.False(t, res.Fresh)
	require.False(t, res.Stale)
	require.Equal(t, "origin down", res.Error)
	require.Empty(t, stats.Lists)
}

func TestApplyLoadFailures_AllowedFailureWithoutPriorLegacy(t *testing.T) {
	// No runtime/prior: legacy behavior — the source is absent from Lists but
	// the stale marker is recorded for Status.
	stats := Stats{
		Lists:         map[string]int{},
		ListHashes:    map[string]string{},
		SourceResults: map[string]SourceResult{},
	}
	applyLoadFailures(log.NewTestLogger(), nil, stats, []loadFailure{
		{source: search.SourceEUCSL, err: errSentinel("eu down"), allowed: true},
	})

	res := stats.SourceResults[string(search.SourceEUCSL)]
	require.True(t, res.Stale)
	require.False(t, res.Fresh)
	require.Equal(t, "eu down", res.Error)
	require.Empty(t, stats.Lists)
}

func TestApplyLoadFailures_AllowedFailureCarriesPrior(t *testing.T) {
	priorEntities := []search.Entity[search.Value]{
		{Name: "Prior One", Source: search.SourceEUCSL, SourceID: "p1"},
		{Name: "Prior Two", Source: search.SourceEUCSL, SourceID: "p2"},
	}
	prior := &fakePrior{sources: map[string]PriorSource{
		string(search.SourceEUCSL): {
			Entities:     priorEntities,
			Hash:         "prior-hash",
			GenerationID: "gen-old",
		},
	}}

	root := t.TempDir()
	store, err := NewGenerationStore(log.NewTestLogger(), Config{RefreshStateDirectory: root})
	require.NoError(t, err)
	workspace, err := store.Create(false)
	require.NoError(t, err)

	stats := Stats{
		Lists:         map[string]int{},
		ListHashes:    map[string]string{},
		SourceResults: map[string]SourceResult{},
	}
	rt := &refreshRuntime{
		generationID: "gen-new",
		prior:        prior,
		workspace:    workspace,
	}
	applyLoadFailures(log.NewTestLogger(), rt, stats, []loadFailure{
		{source: search.SourceEUCSL, err: errSentinel("eu down"), allowed: true},
	})

	res := stats.SourceResults[string(search.SourceEUCSL)]
	require.True(t, res.Stale)
	require.False(t, res.Fresh)
	require.Equal(t, 2, res.Entities)
	require.Equal(t, "prior-hash", res.Hash)
	require.Equal(t, "gen-old", res.PriorGenerationID)
	require.Equal(t, "eu down", res.Error)

	require.Equal(t, 2, stats.Lists[string(search.SourceEUCSL)])
	require.Equal(t, "prior-hash", stats.ListHashes[string(search.SourceEUCSL)])
	require.Len(t, stats.Entities, 2)
}

type errSentinel string

func (e errSentinel) Error() string { return string(e) }

// TestRefreshAll_FailedIgnoredSourceCarriesPriorGeneration runs a full refresh
// against a pinned candidate directory: OFAC loads fresh from pinned files
// while an ignored custom source fails and is served from the prior generation.
func TestRefreshAll_FailedIgnoredSourceCarriesPriorGeneration(t *testing.T) {
	const staleList search.SourceList = "senzing-stale-prior"

	initial := filepath.Join("..", "..", "test", "testdata")
	root := t.TempDir()
	store, err := NewGenerationStore(log.NewTestLogger(), Config{
		RefreshStateDirectory: root,
		InitialDataDirectory:  initial,
	})
	require.NoError(t, err)
	ws, err := store.Create(false)
	require.NoError(t, err)
	t.Cleanup(ws.Discard)

	priorEntities := []search.Entity[search.Value]{
		{Name: "Carry One", Source: staleList, SourceID: "c1"},
		{Name: "Carry Two", Source: staleList, SourceID: "c2"},
	}
	prior := &fakePrior{sources: map[string]PriorSource{
		string(staleList): {Entities: priorEntities, Hash: "carried-hash", GenerationID: "gen-1"},
	}}

	conf := Config{
		InitialDataDirectory: initial,
		IncludedLists:        []search.SourceList{search.SourceUSOFAC},
		Senzing: []SenzingList{
			{SourceList: staleList, Location: "file:///nonexistent/stale.jsonl"},
		},
		IgnoredDownloadErrors: []search.SourceList{staleList},
	}
	dl, err := NewDownloader(log.NewTestLogger(), conf, nil)
	require.NoError(t, err)
	dl.(refreshRuntimeSetter).SetRefreshRuntime(&refreshRuntime{
		workingDir:   ws.DataDir(),
		generationID: "gen-2",
		prior:        prior,
		workspace:    ws,
	})

	stats, err := dl.RefreshAll(context.Background())
	require.NoError(t, err)

	// Fresh OFAC from pinned files.
	ofac := stats.SourceResults[string(search.SourceUSOFAC)]
	require.True(t, ofac.Fresh)
	require.Greater(t, ofac.Entities, 100)

	// Stale custom source carried explicitly from the prior generation.
	stale := stats.SourceResults[string(staleList)]
	require.True(t, stale.Stale)
	require.False(t, stale.Fresh)
	require.Equal(t, 2, stale.Entities)
	require.Equal(t, "carried-hash", stale.Hash)
	require.Equal(t, "gen-1", stale.PriorGenerationID)
	require.NotEmpty(t, stale.Error)
	require.Equal(t, 2, stats.Lists[string(staleList)])

	require.Equal(t, "gen-2", stats.GenerationID)

	// Combined corpus (fresh + carried) feeds TF-IDF.
	require.NotNil(t, stats.TFIDFIndex)
}

// TestRefreshAll_RequiredFailureDoesNotPublishData verifies a required source
// failure returns an error and per-source error provenance.
func TestRefreshAll_RequiredFailureDoesNotPublishData(t *testing.T) {
	const requiredList search.SourceList = "senzing-required-missing"

	conf := Config{
		Senzing: []SenzingList{
			{SourceList: requiredList, Location: "file:///nonexistent/missing.jsonl"},
		},
	}
	dl, err := NewDownloader(log.NewTestLogger(), conf, nil)
	require.NoError(t, err)

	stats, err := dl.RefreshAll(context.Background())
	require.Error(t, err)

	res := stats.SourceResults[string(requiredList)]
	require.False(t, res.Fresh)
	require.False(t, res.Stale)
	require.NotEmpty(t, res.Error)
}
