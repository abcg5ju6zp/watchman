package download

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/moov-io/base/log"
	"github.com/moov-io/watchman/pkg/search"

	"github.com/stretchr/testify/require"
)

func newTestStore(t *testing.T) (*generationStore, string) {
	t.Helper()
	root := t.TempDir()
	s, err := NewGenerationStore(log.NewTestLogger(), Config{RefreshStateDirectory: root})
	require.NoError(t, err)
	require.Equal(t, root, s.root)
	return s, root
}

func genStatsFor(ids ...string) Stats {
	entities := make([]search.Entity[search.Value], 0, len(ids))
	results := make(map[string]SourceResult, len(ids))
	lists := make(map[string]int, len(ids))
	hashes := make(map[string]string, len(ids))
	for i, id := range ids {
		entities = append(entities, search.Entity[search.Value]{
			Name:     "Entity " + id,
			Source:   search.SourceUSOFAC,
			SourceID: id,
		})
		name := string(search.SourceUSOFAC)
		_ = i
		lists[name]++
	}
	name := string(search.SourceUSOFAC)
	hashes[name] = "hash-" + ids[0]
	results[name] = SourceResult{Name: name, Entities: len(ids), Hash: "hash-" + ids[0], Fresh: true}
	return Stats{
		Entities:      entities,
		Lists:         lists,
		ListHashes:    hashes,
		SourceResults: results,
	}
}

func TestGenerationStore_CreateSeedsAndMarksCandidate(t *testing.T) {
	root := t.TempDir()
	initial := filepath.Join(root, "initial")
	require.NoError(t, os.MkdirAll(initial, 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(initial, "sdn.csv"), []byte("seed"), 0o600))

	s, err := NewGenerationStore(log.NewTestLogger(), Config{
		RefreshStateDirectory: filepath.Join(root, "state"),
		InitialDataDirectory:  initial,
	})
	require.NoError(t, err)

	ws, err := s.Create(false)
	require.NoError(t, err)
	require.NotEmpty(t, ws.ID())
	require.True(t, ws.ID() > "gen-0")

	// Marker and seed are present; manifest starts in "building".
	marker, err := os.ReadFile(filepath.Join(ws.DataDir(), ".watchman-candidate"))
	require.NoError(t, err)
	require.Contains(t, string(marker), ws.ID())

	seeded, err := os.ReadFile(filepath.Join(ws.DataDir(), "sdn.csv"))
	require.NoError(t, err)
	require.Equal(t, "seed", string(seeded))

	var m generationManifest
	require.NoError(t, readJSON(filepath.Join(ws.dir, manifestFileName), &m))
	require.Equal(t, phaseBuilding, m.Phase)
	require.Equal(t, ws.ID(), m.GenerationID)
}

func TestGenerationStore_PreparePublishAndRecoverEmpty(t *testing.T) {
	s, _ := newTestStore(t)

	ws, err := s.Create(true)
	require.NoError(t, err)

	// Simulate GetFiles persisting fetched bytes into the candidate data dir.
	require.NoError(t, os.WriteFile(filepath.Join(ws.DataDir(), "add.csv"), []byte("a"), 0o600))
	require.NoError(t, os.WriteFile(filepath.Join(ws.DataDir(), "sdn.csv"), []byte("b"), 0o600))

	stats := genStatsFor("1", "2")
	require.NoError(t, ws.MarkPrepared(stats, true))

	resumed, err := s.Recover()
	require.NoError(t, err)
	require.NotNil(t, resumed)
	require.True(t, resumed.Resumed())
	require.Equal(t, ws.ID(), resumed.ID())
	require.ElementsMatch(t, []string{"add.csv", "sdn.csv"}, resumed.manifest.Files)
	require.True(t, resumed.manifest.EmbeddingsEnabled)

	// Hash mismatch must refuse completion.
	bad := genStatsFor("1", "2")
	bad.ListHashes[string(search.SourceUSOFAC)] = "different"
	require.Error(t, resumed.Matches(bad))
	require.NoError(t, resumed.Matches(stats))

	// Publish promotes files and removes the candidate.
	require.NoError(t, resumed.Publish(stats))

	_, err = os.Stat(filepath.Join(s.root, currentDirName, "sdn.csv"))
	require.NoError(t, err)

	meta, err := os.Stat(filepath.Join(s.root, currentMetaName))
	require.NoError(t, err)
	require.WithinDuration(t, time.Now(), meta.ModTime(), 5*time.Second)

	entries, err := os.ReadDir(filepath.Join(s.root, candidatesDirName))
	require.NoError(t, err)
	require.Empty(t, entries)

	// Nothing to recover afterwards; current generation id is readable.
	require.Equal(t, ws.ID(), s.currentGenerationID())
	again, err := s.Recover()
	require.NoError(t, err)
	require.Nil(t, again)
}

func TestGenerationStore_RecoverDiscardsBuildingCandidate(t *testing.T) {
	s, _ := newTestStore(t)

	ws, err := s.Create(false)
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(filepath.Join(ws.DataDir(), "sdn.csv"), []byte("partial"), 0o600))
	// No MarkPrepared: crash happened during downloads.

	got, err := s.Recover()
	require.NoError(t, err)
	require.Nil(t, got)

	_, err = os.Stat(ws.dir)
	require.True(t, os.IsNotExist(err), "incomplete candidate must be discarded")
}

func TestGenerationStore_RecoverDiscardsCandidateWithMissingFile(t *testing.T) {
	s, _ := newTestStore(t)

	ws, err := s.Create(false)
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(filepath.Join(ws.DataDir(), "sdn.csv"), []byte("b"), 0o600))
	require.NoError(t, ws.MarkPrepared(genStatsFor("1"), false))

	require.NoError(t, os.Remove(filepath.Join(ws.DataDir(), "sdn.csv")))

	got, err := s.Recover()
	require.NoError(t, err)
	require.Nil(t, got, "candidate with missing pinned file must be discarded, not mixed with a fetch")

	_, err = os.Stat(ws.dir)
	require.True(t, os.IsNotExist(err))
}

func TestGenerationStore_CarryCurrentFilesForStaleSource(t *testing.T) {
	s, _ := newTestStore(t)

	// Publish generation 1.
	ws1, err := s.Create(false)
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(filepath.Join(ws1.DataDir(), "sdn.csv"), []byte("gen1"), 0o600))
	stats1 := genStatsFor("1")
	require.NoError(t, ws1.MarkPrepared(stats1, false))
	require.NoError(t, ws1.Publish(stats1))

	// Generation 2: the source fails; carry the prior files over.
	ws2, err := s.Create(false)
	require.NoError(t, err)
	require.Equal(t, ws1.ID(), ws2.PriorGenerationID())
	require.NoError(t, ws2.CarryCurrentFiles())

	got, err := os.ReadFile(filepath.Join(ws2.DataDir(), "sdn.csv"))
	require.NoError(t, err)
	require.Equal(t, "gen1", string(got))

	// The carried files make generation 2 resumable with identical hashes.
	stats2 := genStatsFor("1")
	stale := stats2.SourceResults[string(search.SourceUSOFAC)]
	stale.Fresh = false
	stale.Stale = true
	stale.PriorGenerationID = ws1.ID()
	stats2.SourceResults[string(search.SourceUSOFAC)] = stale
	require.NoError(t, ws2.MarkPrepared(stats2, false))

	resumed, err := s.Recover()
	require.NoError(t, err)
	require.NotNil(t, resumed)
	require.NoError(t, resumed.Matches(stats2))
	resumed.ApplyProvenance(&stats2)
	require.True(t, stats2.SourceResults[string(search.SourceUSOFAC)].Stale)
	require.Equal(t, ws1.ID(), stats2.SourceResults[string(search.SourceUSOFAC)].PriorGenerationID)
}

func TestGenerationStore_RecoverPicksNewestPreparedCandidate(t *testing.T) {
	s, _ := newTestStore(t)

	older, err := s.Create(false)
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(filepath.Join(older.DataDir(), "sdn.csv"), []byte("old"), 0o600))
	require.NoError(t, older.MarkPrepared(genStatsFor("old"), false))
	// Force a distinct, sortable timestamp on the newer candidate.
	time.Sleep(10 * time.Millisecond)

	newer, err := s.Create(false)
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(filepath.Join(newer.DataDir(), "sdn.csv"), []byte("new"), 0o600))
	statsNew := genStatsFor("new")
	require.NoError(t, newer.MarkPrepared(statsNew, false))

	chosen, err := s.Recover()
	require.NoError(t, err)
	require.NotNil(t, chosen)
	require.Equal(t, newer.ID(), chosen.ID())

	_, err = os.Stat(older.dir)
	require.True(t, os.IsNotExist(err), "older leftover candidate must be discarded")
}
