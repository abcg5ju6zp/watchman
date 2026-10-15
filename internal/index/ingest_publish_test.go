package index

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/moov-io/base/log"
	"github.com/moov-io/watchman/internal/db"
	"github.com/moov-io/watchman/internal/ingest"
	"github.com/moov-io/watchman/pkg/search"

	"github.com/stretchr/testify/require"
)

// After a publish, the live generation reports the exact revision and
// content version committed, and pending rows are marked published.
func TestLists_IngestVersion(t *testing.T) {
	repo := &ingest.MockRepository{}
	lists := NewLists(repo)
	ctx := context.Background()

	staged, err := repo.StageVersion(ctx, "fincen-person", []search.Entity[search.Value]{
		testIngestEntity(1),
		testIngestEntity(2),
	})
	require.NoError(t, err)
	require.False(t, staged.Published)

	if _, ok := lists.IngestVersion("fincen-person"); ok {
		t.Fatal("version reported before any publish")
	}

	require.NoError(t, lists.RefreshIngest(ctx))

	live, ok := lists.IngestVersion("fincen-person")
	require.True(t, ok)
	require.Equal(t, staged.Revision, live.Revision)
	require.Equal(t, staged.Version, live.Version)
	require.Equal(t, staged.Checksum, live.Checksum)
	require.Equal(t, 2, live.EntityCount)

	// Rebuilding marks the pending revision published in the repository.
	sums, err := repo.Checksums(ctx)
	require.NoError(t, err)
	require.Len(t, sums, 1)
	require.True(t, sums[0].Published)
}

// A refresh that loaded an older snapshot must not publish over a newer
// revision already live in memory.
func TestLists_RefreshIngest_NoDowngrade(t *testing.T) {
	repo := &ingest.MockRepository{}
	lists := NewLists(repo).(*lists)
	ctx := context.Background()

	_, err := repo.StageVersion(ctx, "fincen-person", []search.Entity[search.Value]{
		testIngestEntity(1),
	})
	require.NoError(t, err)
	require.NoError(t, lists.RefreshIngest(ctx))

	live, ok := lists.IngestVersion("fincen-person")
	require.True(t, ok)
	require.Equal(t, int64(1), live.Revision)

	// Simulate a winning concurrent publish that memory already serves while a
	// stale refresh still holds revision 1 from the database.
	lists.mu.Lock()
	lists.ingestRevisions["fincen-person"] = 99
	lists.mu.Unlock()

	require.NoError(t, lists.RefreshIngest(ctx))

	live, ok = lists.IngestVersion("fincen-person")
	require.True(t, ok)
	require.Equal(t, int64(99), live.Revision, "older refresh must not downgrade memory")
}

// errorCache fails every in-memory publish, simulating a corpus build failure
// after the database commit.
type errorCache struct {
	err error
}

func (c *errorCache) RefreshIngest(context.Context) error { return c.err }

// A failed in-memory publish leaves a recoverable pending revision; retrying
// the same file resumes the publish stage without writing rows again, and a
// fresh process rebuilds online search from the committed rows.
func TestLists_IngestPublishRecovery(t *testing.T) {
	db.ForEachDatabase(t, func(database db.DB) {
		repo := ingest.NewRepository(database)
		ctx := context.Background()

		logger := log.NewTestLogger()
		conf := ingest.Config{
			Files: map[string]ingest.File{
				"banks": {
					Format: ingest.FormatCSV,
					Mapping: ingest.Mapping{
						Name:     ingest.ColumnDef{Column: "name"},
						SourceID: ingest.ColumnDef{Column: "id"},
						Type:     ingest.Type{Default: "business"},
					},
				},
			},
		}
		svc := ingest.NewService(logger, conf, repo)
		lists := NewLists(repo)

		body := "id,name\n1,First Bank\n2,Second Bank\n"

		// First attempt: database commit succeeds, in-memory publish fails.
		failing := &errorCache{err: errors.New("corpus build failed")}
		_, err := svc.IngestUpload(ctx, "banks", strings.NewReader(body), failing)
		require.Error(t, err)

		pending, err := repo.CurrentVersion(ctx, "banks")
		require.NoError(t, err)
		require.False(t, pending.Published)
		require.Equal(t, 2, pending.EntityCount)

		// Search must not serve the failed upload before it is published.
		if _, ok := lists.IngestVersion("banks"); ok {
			t.Fatal("failed publish must not be live")
		}

		// Retry the exact same file after the cache recovers: no new revision.
		parsed, err := svc.IngestUpload(ctx, "banks", strings.NewReader(body), lists)
		require.NoError(t, err)
		require.Len(t, parsed.Entities, 2)

		after, err := repo.CurrentVersion(ctx, "banks")
		require.NoError(t, err)
		require.True(t, after.Published)
		require.Equal(t, pending.Revision, after.Revision, "retry must not rewrite rows")
		require.Equal(t, pending.Version, after.Version)

		live, ok := lists.IngestVersion("banks")
		require.True(t, ok)
		require.Equal(t, pending.Revision, live.Revision)

		// Online search serves the recovered version.
		query := search.Entity[search.Value]{
			Name:   "First Bank",
			Type:   search.EntityBusiness,
			Source: "banks",
		}.Normalize()
		cands, err := lists.SelectCandidates(ctx, query)
		require.NoError(t, err)
		require.Greater(t, cands.Len(), 0)

		// Restart: a brand new lists rebuilds the committed version on its own.
		restarted := NewLists(repo)
		require.NoError(t, restarted.RefreshIngest(ctx))
		recovered, ok := restarted.IngestVersion("banks")
		require.True(t, ok)
		require.Equal(t, pending.Revision, recovered.Revision)
		require.Equal(t, pending.Version, recovered.Version)

		cands, err = restarted.SelectCandidates(ctx, query)
		require.NoError(t, err)
		require.Greater(t, cands.Len(), 0)
	})
}

// Pending rows are picked up on startup without re-uploading the file.
func TestLists_IngestRestartPublishesPending(t *testing.T) {
	db.ForEachDatabase(t, func(database db.DB) {
		repo := ingest.NewRepository(database)
		ctx := context.Background()

		// Process A dies between commit and publish.
		staged, err := repo.StageVersion(ctx, "fincen-person", []search.Entity[search.Value]{
			testIngestEntity(1),
			testIngestEntity(2),
		})
		require.NoError(t, err)
		require.False(t, staged.Published)

		// Process B starts and only has the database to recover from.
		restarted := NewLists(repo)
		require.NoError(t, restarted.RefreshIngest(ctx))

		live, ok := restarted.IngestVersion("fincen-person")
		require.True(t, ok)
		require.Equal(t, staged.Revision, live.Revision)
		require.Equal(t, staged.Version, live.Version)
		require.Equal(t, 2, live.EntityCount)

		// The startup rebuild also closes the pending stage.
		current, err := repo.CurrentVersion(ctx, "fincen-person")
		require.NoError(t, err)
		require.True(t, current.Published)
	})
}

// End-to-end version behavior against a real database through the service.
func TestLists_IngestUploadVersions(t *testing.T) {
	db.ForEachDatabase(t, func(database db.DB) {
		repo := ingest.NewRepository(database)
		ctx := context.Background()

		conf := ingest.Config{
			Files: map[string]ingest.File{
				"banks": {
					Format: ingest.FormatCSV,
					Mapping: ingest.Mapping{
						Name:     ingest.ColumnDef{Column: "name"},
						SourceID: ingest.ColumnDef{Column: "id"},
						Type:     ingest.Type{Default: "business"},
					},
				},
			},
		}
		svc := ingest.NewService(log.NewTestLogger(), conf, repo)
		lists := NewLists(repo)

		v1Body := "id,name\n1,First Bank\n2,Second Bank\n"
		_, err := svc.IngestUpload(ctx, "banks", strings.NewReader(v1Body), lists)
		require.NoError(t, err)
		v1, err := repo.CurrentVersion(ctx, "banks")
		require.NoError(t, err)
		require.True(t, v1.Published)

		// Identical upload: same revision.
		_, err = svc.IngestUpload(ctx, "banks", strings.NewReader(v1Body), lists)
		require.NoError(t, err)
		again, err := repo.CurrentVersion(ctx, "banks")
		require.NoError(t, err)
		require.Equal(t, v1.Revision, again.Revision)

		// New content: revision advances and search serves the new names while
		// the old ones disappear.
		v2Body := "id,name\n1,First Bank Renamed\n3,Third Bank\n"
		_, err = svc.IngestUpload(ctx, "banks", strings.NewReader(v2Body), lists)
		require.NoError(t, err)
		v2, err := repo.CurrentVersion(ctx, "banks")
		require.NoError(t, err)
		require.Equal(t, v1.Revision+1, v2.Revision)
		require.True(t, v2.Published)

		entities, err := lists.GetEntities(ctx, "banks")
		require.NoError(t, err)
		require.Len(t, entities, 2)
		names := map[string]bool{}
		for _, e := range entities {
			names[e.Name] = true
		}
		require.True(t, names["First Bank Renamed"])
		require.True(t, names["Third Bank"])
		require.False(t, names["Second Bank"])

		live, ok := lists.IngestVersion("banks")
		require.True(t, ok)
		require.Equal(t, v2.Revision, live.Revision)
	})
}
