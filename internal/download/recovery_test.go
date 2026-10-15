package download

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/moov-io/base/log"
	"github.com/moov-io/watchman/pkg/search"

	"github.com/stretchr/testify/require"
)

func readCandidateDirs(root string) ([]string, error) {
	entries, err := os.ReadDir(filepath.Join(root, candidatesDirName))
	if os.IsNotExist(err) {
		return nil, nil
	} else if err != nil {
		return nil, err
	}
	var names []string
	for _, e := range entries {
		names = append(names, e.Name())
	}
	return names, nil
}

func readLatestCurrentFiles(root string) ([]string, error) {
	entries, err := os.ReadDir(filepath.Join(root, currentDirName))
	if err != nil {
		return nil, err
	}
	var names []string
	for _, e := range entries {
		names = append(names, e.Name())
	}
	return names, nil
}

func removeOnePinnedFile(dataDir string) error {
	entries, err := os.ReadDir(dataDir)
	if err != nil {
		return err
	}
	for _, e := range entries {
		if !e.IsDir() && e.Name() != ".watchman-candidate" {
			return os.Remove(filepath.Join(dataDir, e.Name()))
		}
	}
	return nil
}

// TestRefresher_StartupRecoveryCompletesSameCandidate simulates a process exit
// after a candidate was prepared but before it was published. A brand-new
// Refresher (same state root) must resume that exact generation from its pinned
// files — even with an unusable initial data directory — without downloading or
// mixing files.
func TestRefresher_StartupRecoveryCompletesSameCandidate(t *testing.T) {
	ctx := context.Background()
	logger := log.NewTestLogger()
	initial, err := filepath.Abs(filepath.Join("..", "..", "test", "testdata"))
	require.NoError(t, err)

	root := t.TempDir()
	conf := Config{
		RefreshStateDirectory: root,
		InitialDataDirectory:  initial,
		ErrorOnEmptyList:      true,
		IncludedLists:         []search.SourceList{search.SourceUSOFAC},
	}

	// --- First process: publish generation 1. ---
	store1, err := NewGenerationStore(logger, conf)
	require.NoError(t, err)
	dl1, err := NewDownloader(logger, conf, nil)
	require.NoError(t, err)
	lists1 := &testPublishedLists{}
	r1 := NewRefresher(ctx, logger, dl1, lists1, nil, WithGenerationStore(store1))

	require.NoError(t, r1.Recover(ctx))
	require.NoError(t, r1.RefreshNow(ctx, TriggerStartup))
	gen1 := r1.Status().GenerationID
	require.NotEmpty(t, gen1)
	require.Equal(t, 1, lists1.publishCount())

	// current/ holds the published pinned files.
	_, err = readLatestCurrentFiles(root)
	require.NoError(t, err)

	// --- Crash window: prepare a generation-2 candidate but never publish. ---
	ws2, err := store1.Create(false)
	require.NoError(t, err)
	gen2 := ws2.ID()
	require.NotEqual(t, gen1, gen2)

	dl2, err := NewDownloader(logger, conf, nil)
	require.NoError(t, err)
	dl2.(refreshRuntimeSetter).SetRefreshRuntime(&refreshRuntime{
		workingDir:   ws2.DataDir(),
		generationID: gen2,
	})
	stats2, err := dl2.RefreshAll(ctx)
	require.NoError(t, err)
	require.Greater(t, stats2.Lists[string(search.SourceUSOFAC)], 100)
	require.NoError(t, ws2.MarkPrepared(stats2, false))
	dl2.(refreshRuntimeSetter).SetRefreshRuntime(nil)

	// The candidate exists in prepared state; current is still generation 1.
	require.Equal(t, gen1, store1.currentGenerationID())

	// --- Second "process": fresh Refresher, initial dir is now unusable. ---
	recoveryConf := conf
	recoveryConf.InitialDataDirectory = filepath.Join(t.TempDir(), "does-not-exist")
	store2, err := NewGenerationStore(logger, recoveryConf)
	require.NoError(t, err)
	dl3, err := NewDownloader(logger, recoveryConf, nil)
	require.NoError(t, err)
	lists2 := &testPublishedLists{}
	r2 := NewRefresher(ctx, logger, dl3, lists2, nil, WithGenerationStore(store2))

	require.NoError(t, r2.Recover(ctx))

	// Startup completes the same candidate.
	require.NoError(t, r2.RefreshNow(ctx, TriggerStartup))
	st := r2.Status()
	require.Equal(t, StateSucceeded, st.State)
	require.Equal(t, gen2, st.GenerationID)
	require.Len(t, lists2.published, 1)
	require.Equal(t, gen2, lists2.published[0].GenerationID)
	require.Equal(t, stats2.ListHashes[string(search.SourceUSOFAC)],
		lists2.published[0].ListHashes[string(search.SourceUSOFAC)])

	// Candidate is gone and current.json advanced to generation 2.
	require.Equal(t, gen2, store2.currentGenerationID())
	entries, err := readCandidateDirs(root)
	require.NoError(t, err)
	require.Empty(t, entries)

	// No candidates remain for a later recovery.
	again, err := store2.Recover()
	require.NoError(t, err)
	require.Nil(t, again)
}

// TestRefresher_StartupRecoveryDiscardsAndRefreshes verifies an interrupted
// candidate that can no longer be completed (pinned file missing) is discarded
// and replaced by one coherent fresh generation.
func TestRefresher_StartupRecoveryDiscardsAndRefreshes(t *testing.T) {
	ctx := context.Background()
	logger := log.NewTestLogger()
	initial, err := filepath.Abs(filepath.Join("..", "..", "test", "testdata"))
	require.NoError(t, err)

	root := t.TempDir()
	conf := Config{
		RefreshStateDirectory: root,
		InitialDataDirectory:  initial,
		ErrorOnEmptyList:      true,
		IncludedLists:         []search.SourceList{search.SourceUSOFAC},
	}

	store, err := NewGenerationStore(logger, conf)
	require.NoError(t, err)

	// Prepared candidate, then one pinned file disappears.
	ws, err := store.Create(false)
	require.NoError(t, err)
	dl, err := NewDownloader(logger, conf, nil)
	require.NoError(t, err)
	dl.(refreshRuntimeSetter).SetRefreshRuntime(&refreshRuntime{
		workingDir:   ws.DataDir(),
		generationID: ws.ID(),
	})
	stats, err := dl.RefreshAll(ctx)
	require.NoError(t, err)
	require.NoError(t, ws.MarkPrepared(stats, false))
	dl.(refreshRuntimeSetter).SetRefreshRuntime(nil)
	require.NoError(t, removeOnePinnedFile(ws.DataDir()))

	// Recovery discards the broken candidate.
	got, err := store.Recover()
	require.NoError(t, err)
	require.Nil(t, got)

	// A normal startup then builds a fresh, coherent generation.
	dl2, err := NewDownloader(logger, conf, nil)
	require.NoError(t, err)
	lists := &testPublishedLists{}
	r := NewRefresher(ctx, logger, dl2, lists, nil, WithGenerationStore(store))
	require.NoError(t, r.Recover(ctx))
	require.NoError(t, r.RefreshNow(ctx, TriggerStartup))
	require.Equal(t, StateSucceeded, r.Status().State)
	require.NotEmpty(t, r.Status().GenerationID)
	require.Equal(t, 1, lists.publishCount())
}
