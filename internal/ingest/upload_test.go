package ingest

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"

	"github.com/moov-io/base/log"
	"github.com/moov-io/watchman/pkg/search"

	"github.com/stretchr/testify/require"
)

func uploadTestConfig() Config {
	return Config{
		Files: map[string]File{
			"banks": {
				Format: FormatCSV,
				Mapping: Mapping{
					Name:     ColumnDef{Column: "name"},
					SourceID: ColumnDef{Column: "id"},
					Type:     Type{Default: "business"},
				},
			},
			"unions": {
				Format: FormatCSV,
				Mapping: Mapping{
					Name:     ColumnDef{Column: "name"},
					SourceID: ColumnDef{Column: "id"},
					Type:     Type{Default: "business"},
				},
			},
			"persons": {
				Format: FormatCSV,
				Mapping: Mapping{
					Name:     ColumnDef{Column: "name"},
					SourceID: ColumnDef{Column: "id"},
					Type:     Type{Default: "person"},
					Person: &Person{
						Name:      ColumnDef{Column: "name"},
						BirthDate: ColumnDef{Column: "dob"},
					},
				},
			},
		},
	}
}

const banksV1 = `id,name
1,First Bank
2,Second Bank
`

const banksV2 = `id,name
1,First Bank Renamed
3,Third Bank
`

// fakeIngestCache simulates the in-memory lists: a successful refresh serves
// exactly the revisions currently committed in the repository.
type fakeIngestCache struct {
	mu sync.Mutex

	repo Repository
	fail error

	refreshes int
	live      map[string]SourceChecksum

	// overrides force IngestVersion answers to simulate another upload having
	// published a newer revision in memory.
	overrides map[string]SourceChecksum
}

func (c *fakeIngestCache) RefreshIngest(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.refreshes++
	if c.fail != nil {
		return c.fail
	}
	sums, err := c.repo.Checksums(ctx)
	if err != nil {
		return err
	}
	live := make(map[string]SourceChecksum, len(sums))
	for _, s := range sums {
		s.Published = true
		live[s.Source] = s
	}
	c.live = live
	return nil
}

func (c *fakeIngestCache) IngestVersion(source string) (SourceChecksum, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if v, ok := c.overrides[source]; ok {
		return v, true
	}
	v, ok := c.live[source]
	return v, ok
}

func newUploadService(t *testing.T, repo Repository) (Service, *fakeIngestCache) {
	t.Helper()

	cache := &fakeIngestCache{repo: repo}
	svc := NewService(log.NewTestLogger(), uploadTestConfig(), repo)
	return svc, cache
}

func sourceEntities(t *testing.T, ctx context.Context, repo Repository, source string) []search.Entity[search.Value] {
	t.Helper()

	entities, err := repo.ListBySource(ctx, "", search.SourceList(source), 1000)
	require.NoError(t, err)
	return entities
}

// Identical content resubmitted is idempotent: no new revision, no duplicate
// rows and no extra entity writes.
func TestService_IngestUpload_IdempotentSameContent(t *testing.T) {
	ctx := context.Background()
	repo := &MockRepository{}
	svc, cache := newUploadService(t, repo)

	first, err := svc.IngestUpload(ctx, "banks", strings.NewReader(banksV1), cache)
	require.NoError(t, err)
	require.Len(t, first.Entities, 2)

	v1, err := repo.CurrentVersion(ctx, "banks")
	require.NoError(t, err)
	require.Equal(t, int64(1), v1.Revision)
	require.True(t, v1.Published)
	require.Len(t, v1.Version, 64)
	require.Len(t, sourceEntities(t, ctx, repo, "banks"), 2)

	// Same file again (gzip or plain text parse to the same entity set).
	second, err := svc.IngestUpload(ctx, "banks", strings.NewReader(banksV1), cache)
	require.NoError(t, err)
	require.Len(t, second.Entities, 2)

	v2, err := repo.CurrentVersion(ctx, "banks")
	require.NoError(t, err)
	require.Equal(t, v1.Revision, v2.Revision, "identical content must not allocate a revision")
	require.Equal(t, v1.Version, v2.Version)
	require.Len(t, sourceEntities(t, ctx, repo, "banks"), 2, "identical content must not duplicate rows")

	require.Equal(t, 2, cache.refreshes)
}

// Different content creates a new revision and replaces the rows; online
// search then reports the newer version.
func TestService_IngestUpload_NewContentReplaces(t *testing.T) {
	ctx := context.Background()
	repo := &MockRepository{}
	svc, cache := newUploadService(t, repo)

	_, err := svc.IngestUpload(ctx, "banks", strings.NewReader(banksV1), cache)
	require.NoError(t, err)

	_, err = svc.IngestUpload(ctx, "banks", strings.NewReader(banksV2), cache)
	require.NoError(t, err)

	v, err := repo.CurrentVersion(ctx, "banks")
	require.NoError(t, err)
	require.Equal(t, int64(2), v.Revision)
	require.True(t, v.Published)

	entities := sourceEntities(t, ctx, repo, "banks")
	require.Len(t, entities, 2)
	ids := map[string]string{}
	for _, e := range entities {
		ids[e.SourceID] = e.Name
	}
	require.Equal(t, "First Bank Renamed", ids["1"])
	require.Equal(t, "Third Bank", ids["3"])
	require.NotContains(t, ids, "2", "old version rows must be gone")

	live, ok := cache.IngestVersion("banks")
	require.True(t, ok)
	require.Equal(t, int64(2), live.Revision)
	require.Equal(t, v.Version, live.Version)
}

// Parse, schema and merge failures happen before any database write, so the
// previously visible version and other sources stay intact.
func TestService_IngestUpload_ParseFailureKeepsOldVersion(t *testing.T) {
	ctx := context.Background()
	repo := &MockRepository{}
	svc, cache := newUploadService(t, repo)

	_, err := svc.IngestUpload(ctx, "banks", strings.NewReader(banksV1), cache)
	require.NoError(t, err)
	before, err := repo.CurrentVersion(ctx, "banks")
	require.NoError(t, err)

	_, err = svc.IngestUpload(ctx, "unions", strings.NewReader(`id,name
9,Other Credit Union
`), cache)
	require.NoError(t, err)

	// Unknown fileType/schema is rejected.
	_, err = svc.IngestUpload(ctx, "unknown-schema", strings.NewReader(banksV1), cache)
	require.Error(t, err)

	// Known schema but zero entities after parsing is rejected.
	_, err = svc.IngestUpload(ctx, "banks", strings.NewReader("id,name\n"), cache)
	require.Error(t, err)

	// Malformed row (bad date) is rejected mid-parse.
	_, err = svc.IngestUpload(ctx, "persons", strings.NewReader("id,name,dob\n1,Pat,not-a-date\n"), cache)
	require.Error(t, err)

	after, err := repo.CurrentVersion(ctx, "banks")
	require.NoError(t, err)
	require.Equal(t, before, after, "banks must remain on the old version")
	require.Len(t, sourceEntities(t, ctx, repo, "banks"), 2)

	unions, err := repo.CurrentVersion(ctx, "unions")
	require.NoError(t, err)
	require.Equal(t, int64(1), unions.Revision)
	require.Len(t, sourceEntities(t, ctx, repo, "unions"), 1)

	// persons never created a version.
	_, err = repo.CurrentVersion(ctx, "persons")
	require.ErrorIs(t, err, ErrSourceVersionNotFound)
}

// When the database commit succeeds but the in-memory publish fails, the
// revision stays pending with its rows intact; retrying the same content only
// reruns the publish stage and must not rewrite or duplicate rows.
func TestService_IngestUpload_PublishFailureIsRecoverable(t *testing.T) {
	ctx := context.Background()
	repo := &MockRepository{}
	svc, cache := newUploadService(t, repo)

	cache.fail = errors.New("corpus publish failed")
	_, err := svc.IngestUpload(ctx, "banks", strings.NewReader(banksV1), cache)
	require.Error(t, err)

	pending, err := repo.CurrentVersion(ctx, "banks")
	require.NoError(t, err)
	require.False(t, pending.Published, "failed publish leaves the revision pending")
	require.Equal(t, int64(1), pending.Revision)
	require.Len(t, sourceEntities(t, ctx, repo, "banks"), 2, "committed rows are retained for recovery")

	// Client retries the same upload after the cache recovers.
	cache.fail = nil
	_, err = svc.IngestUpload(ctx, "banks", strings.NewReader(banksV1), cache)
	require.NoError(t, err)

	again, err := repo.CurrentVersion(ctx, "banks")
	require.NoError(t, err)
	require.Equal(t, pending.Revision, again.Revision, "retry must not replace data")
	require.Equal(t, pending.Version, again.Version)
	require.True(t, again.Published)
	require.Len(t, sourceEntities(t, ctx, repo, "banks"), 2)
}

// An in-memory publish that already serves a newer revision cannot be claimed
// by an older upload.
func TestService_IngestUpload_OlderPublishDoesNotWin(t *testing.T) {
	ctx := context.Background()
	repo := &MockRepository{}
	svc, cache := newUploadService(t, repo)

	// Pretend a newer upload is already live in memory while we stage rev 1.
	cache.overrides = map[string]SourceChecksum{
		"banks": {Source: "banks", Revision: 99, Version: strings.Repeat("f", 64), EntityCount: 9, Published: true},
	}

	_, err := svc.IngestUpload(ctx, "banks", strings.NewReader(banksV1), cache)
	require.ErrorIs(t, err, ErrVersionSuperseded)

	staged, err := repo.CurrentVersion(ctx, "banks")
	require.NoError(t, err)
	require.False(t, staged.Published, "losing upload must not mark its revision published")
	require.Equal(t, int64(1), staged.Revision)
}

// Restart-style recovery marks pending revisions published using only the
// committed database rows.
func TestService_RecoverPending(t *testing.T) {
	ctx := context.Background()
	repo := &MockRepository{}
	svc, cache := newUploadService(t, repo)

	require.NoError(t, svc.RecoverPending(ctx, cache), "nothing pending is a no-op")
	require.Equal(t, 0, cache.refreshes)

	staged, err := repo.StageVersion(ctx, "banks", []search.Entity[search.Value]{
		{Name: "First Bank", Type: search.EntityBusiness, SourceID: "1"},
	})
	require.NoError(t, err)
	require.False(t, staged.Published)

	cache.fail = errors.New("startup publish failed")
	require.Error(t, svc.RecoverPending(ctx, cache))
	still, err := repo.CurrentVersion(ctx, "banks")
	require.NoError(t, err)
	require.False(t, still.Published)
	require.Equal(t, staged.Revision, still.Revision)

	cache.fail = nil
	require.NoError(t, svc.RecoverPending(ctx, cache))
	recovered, err := repo.CurrentVersion(ctx, "banks")
	require.NoError(t, err)
	require.True(t, recovered.Published)
	require.Equal(t, staged.Revision, recovered.Revision)
	require.Len(t, sourceEntities(t, ctx, repo, "banks"), 1, "recovery rewrites no rows")
}

// With no in-memory cache configured the upload still succeeds; the revision
// is simply marked published in the database.
func TestService_IngestUpload_NoCache(t *testing.T) {
	ctx := context.Background()
	repo := &MockRepository{}
	svc, _ := newUploadService(t, repo)

	_, err := svc.IngestUpload(ctx, "banks", strings.NewReader(banksV1), nil)
	require.NoError(t, err)

	v, err := repo.CurrentVersion(ctx, "banks")
	require.NoError(t, err)
	require.True(t, v.Published)
}

// ReadEntitiesFromFile still reports a helpful failure for an unknown format.
func TestService_IngestUpload_UnknownFormat(t *testing.T) {
	conf := uploadTestConfig()
	conf.Files["weird"] = File{Format: Format("yaml")}
	svc := NewService(log.NewTestLogger(), conf, &MockRepository{})

	_, err := svc.IngestUpload(context.Background(), "weird", nil, nil)
	require.Error(t, err)
}
