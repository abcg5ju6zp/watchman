package ingest

import (
	"context"
	"fmt"
	"testing"

	"github.com/moov-io/watchman/internal/db"
	"github.com/moov-io/watchman/internal/ofactest"
	"github.com/moov-io/watchman/pkg/search"

	"github.com/stretchr/testify/require"
)

func TestRepository(t *testing.T) {
	db.ForEachDatabase(t, func(db db.DB) {
		repo := NewRepository(db)

		ctx := context.Background()
		entity := ofactest.FindEntity(t, "44525")
		entities := []search.Entity[search.Value]{entity}

		err := repo.Upsert(ctx, "fincen-person", entities)
		require.NoError(t, err)

		found, err := repo.Get(ctx, entity.SourceID, "fincen-person")
		require.NoError(t, err)
		require.NotNil(t, found)

		// Empty SourceData // TODO(adam): can we unmarshal?
		entity.SourceData = make(map[string]interface{})
		found.SourceData = make(map[string]interface{})
		entity.Source = "fincen-person"

		// Compare objects
		require.Equal(t, entity.Normalize(), found.Normalize())

		// List
		entities, err = repo.ListBySource(ctx, "", "fincen-person", 10)
		require.NoError(t, err)
		require.Len(t, entities, 1)
		require.Equal(t, entity.SourceID, entities[0].SourceID)

		// List again
		entities, err = repo.ListBySource(ctx, "12345", "fincen-person", 10)
		require.NoError(t, err)
		require.Len(t, entities, 1)
		require.Equal(t, entity.SourceID, entities[0].SourceID)

		// Find nothing
		entities, err = repo.ListBySource(ctx, entity.SourceID, "fincen-person", 10)
		require.NoError(t, err)
		require.Empty(t, entities)
	})
}

func TestRepository_Normalize(t *testing.T) {
	db.ForEachDatabase(t, func(db db.DB) {
		repo := NewRepository(db)

		ctx := context.Background()
		entity := ofactest.FindEntity(t, "44525")
		entities := []search.Entity[search.Value]{entity}

		err := repo.Upsert(ctx, "fincen-person", entities)
		require.NoError(t, err)

		found, err := repo.Get(ctx, entity.SourceID, "fincen-person")
		require.NoError(t, err)
		require.NotNil(t, found)

		require.NotEmpty(t, found.PreparedFields.NameFields)
	})
}

func TestRepository_Upsert(t *testing.T) {
	db.ForEachDatabase(t, func(db db.DB) {
		repo := NewRepository(db)

		ctx := context.Background()
		entity := ofactest.FindEntity(t, "44525")
		entity2 := ofactest.FindEntity(t, "48727")
		entities := []search.Entity[search.Value]{entity, entity2}

		err := repo.Upsert(ctx, "fincen-person", entities)
		require.NoError(t, err)

		// Remove entity2 from the upsert list
		entities = []search.Entity[search.Value]{entity}
		entities[0].Name = "john doe"

		err = repo.Upsert(ctx, "fincen-person", entities)
		require.NoError(t, err)

		found, err := repo.Get(ctx, entity.SourceID, "fincen-person")
		require.NoError(t, err)
		require.NotNil(t, found)

		require.Equal(t, "john doe", found.Name)
	})
}

func TestRepository_ChecksumsAndListAll(t *testing.T) {
	db.ForEachDatabase(t, func(db db.DB) {
		repo := NewRepository(db)
		ctx := context.Background()

		var entities []search.Entity[search.Value]
		for i := 0; i < 1205; i++ {
			entities = append(entities, search.Entity[search.Value]{
				Name:     fmt.Sprintf("person %05d", i),
				Type:     search.EntityPerson,
				SourceID: fmt.Sprintf("%05d", i),
			})
		}

		err := repo.Upsert(ctx, "fincen-person", entities)
		require.NoError(t, err)

		sums, err := repo.Checksums(ctx)
		require.NoError(t, err)
		require.Len(t, sums, 1)
		require.Equal(t, "fincen-person", sums[0].Source)
		require.Equal(t, 1205, sums[0].EntityCount)
		require.Len(t, sums[0].Checksum, 64)

		all, err := repo.ListAll(ctx)
		require.NoError(t, err)
		require.Len(t, all, 1205)
		require.Equal(t, "00000", all[0].SourceID)
		require.Equal(t, "01204", all[len(all)-1].SourceID)

		page, err := repo.ListBySource(ctx, "", "fincen-person", 1000)
		require.NoError(t, err)
		require.Len(t, page, 1000)

		page2, err := repo.ListBySource(ctx, page[len(page)-1].SourceID, "fincen-person", 1000)
		require.NoError(t, err)
		require.Len(t, page2, 205)

		err = repo.Upsert(ctx, "fincen-person", nil)
		require.NoError(t, err)
		sums, err = repo.Checksums(ctx)
		require.NoError(t, err)
		require.Empty(t, sums)
	})
}

func stageEntities(names map[string]string) []search.Entity[search.Value] {
	var out []search.Entity[search.Value]
	for id, name := range names {
		out = append(out, search.Entity[search.Value]{
			Name:     name,
			Type:     search.EntityBusiness,
			SourceID: id,
		})
	}
	return out
}

// StageVersion commits an isolated per-source replacement: stable content
// identity, monotonic revisions, no leftover rows from older versions and a
// pending publish stage that is confirmed conditionally.
func TestRepository_StageVersion(t *testing.T) {
	db.ForEachDatabase(t, func(db db.DB) {
		repo := NewRepository(db)
		ctx := context.Background()

		// Empty staging is rejected rather than deleting the source.
		_, err := repo.StageVersion(ctx, "banks", nil)
		require.Error(t, err)
		_, err = repo.CurrentVersion(ctx, "banks")
		require.ErrorIs(t, err, ErrSourceVersionNotFound)

		v1, err := repo.StageVersion(ctx, "banks", stageEntities(map[string]string{
			"1": "First Bank",
			"9": "Old Ninth Bank",
		}))
		require.NoError(t, err)
		require.Equal(t, "banks", v1.Source)
		require.Equal(t, int64(1), v1.Revision)
		require.Len(t, v1.Version, 64)
		require.Equal(t, v1.Version, v1.Checksum)
		require.False(t, v1.Published)

		cur, err := repo.CurrentVersion(ctx, "banks")
		require.NoError(t, err)
		require.Equal(t, v1, cur)

		// Replace with a partially overlapping version.
		v2, err := repo.StageVersion(ctx, "banks", stageEntities(map[string]string{
			"1": "First Bank Renamed",
			"2": "Second Bank",
		}))
		require.NoError(t, err)
		require.Equal(t, int64(2), v2.Revision)
		require.NotEqual(t, v1.Version, v2.Version)
		require.False(t, v2.Published)

		// Only v2 rows are visible: the old id is gone and nothing duplicates.
		page, err := repo.ListBySource(ctx, "", "banks", 100)
		require.NoError(t, err)
		require.Len(t, page, 2)
		byID := map[string]string{}
		for _, e := range page {
			byID[e.SourceID] = e.Name
		}
		require.Equal(t, "First Bank Renamed", byID["1"])
		require.Equal(t, "Second Bank", byID["2"])
		require.NotContains(t, byID, "9")

		// Exactly one row version exists for the source: no mixed generations.
		rows, err := db.QueryContext(ctx,
			"SELECT DISTINCT version FROM ingested_entities WHERE source = ?;", "banks")
		require.NoError(t, err)
		var versions []string
		for rows.Next() {
			var v string
			require.NoError(t, rows.Scan(&v))
			versions = append(versions, v)
		}
		require.NoError(t, rows.Close())
		require.Equal(t, []string{v2.Version}, versions)

		// Summary describes exactly the committed version.
		sums, err := repo.Checksums(ctx)
		require.NoError(t, err)
		require.Len(t, sums, 1)
		require.Equal(t, 2, sums[0].EntityCount)
		require.Equal(t, v2.Version, sums[0].Version)
		require.False(t, sums[0].Published)

		// An older revision cannot be confirmed once replaced.
		ok, err := repo.MarkPublished(ctx, "banks", v1.Revision)
		require.NoError(t, err)
		require.False(t, ok)

		// The current revision flips to published, idempotently.
		ok, err = repo.MarkPublished(ctx, "banks", v2.Revision)
		require.NoError(t, err)
		require.True(t, ok)
		ok, err = repo.MarkPublished(ctx, "banks", v2.Revision)
		require.NoError(t, err)
		require.True(t, ok)

		cur, err = repo.CurrentVersion(ctx, "banks")
		require.NoError(t, err)
		require.True(t, cur.Published)
	})
}

// Replacing one source never touches another source's rows or summary.
func TestRepository_StageVersion_IsolatesSources(t *testing.T) {
	db.ForEachDatabase(t, func(db db.DB) {
		repo := NewRepository(db)
		ctx := context.Background()

		banks, err := repo.StageVersion(ctx, "banks", stageEntities(map[string]string{"1": "First Bank"}))
		require.NoError(t, err)
		unions, err := repo.StageVersion(ctx, "unions", stageEntities(map[string]string{"1": "Credit Union"}))
		require.NoError(t, err)

		_, err = repo.StageVersion(ctx, "banks", stageEntities(map[string]string{"2": "Second Bank"}))
		require.NoError(t, err)

		banksEntities, err := repo.ListBySource(ctx, "", "banks", 100)
		require.NoError(t, err)
		require.Len(t, banksEntities, 1)
		require.Equal(t, "2", banksEntities[0].SourceID)

		unionsEntities, err := repo.ListBySource(ctx, "", "unions", 100)
		require.NoError(t, err)
		require.Len(t, unionsEntities, 1)
		require.Equal(t, "Credit Union", unionsEntities[0].Name)

		cur, err := repo.CurrentVersion(ctx, "unions")
		require.NoError(t, err)
		require.Equal(t, unions.Version, cur.Version)
		require.Equal(t, int64(1), cur.Revision, "other source revision must not advance")

		cur, err = repo.CurrentVersion(ctx, "banks")
		require.NoError(t, err)
		require.Equal(t, int64(2), cur.Revision)
		_ = banks
	})
}
