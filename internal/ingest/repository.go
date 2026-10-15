package ingest

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/moov-io/watchman/internal/db"
	"github.com/moov-io/watchman/pkg/search"
)

const listPageSize = 1000

// ErrSourceVersionNotFound is returned when a source has no staged version.
var ErrSourceVersionNotFound = errors.New("ingest source version not found")

// ErrConcurrentReplacement is returned when two replacements for the same
// source race their commit and one loses. Nothing was committed by the loser;
// retrying the upload is safe.
var ErrConcurrentReplacement = errors.New("concurrent ingest replacement for the same source")

type Repository interface {
	// Upsert replaces every row of one source. Prefer StageVersion for upload
	// handling: Upsert marks the replacement immediately published and is kept
	// for callers that replace rows without an in-memory publish stage.
	Upsert(ctx context.Context, fileType string, entities []search.Entity[search.Value]) error

	// StageVersion replaces one source inside a single isolated transaction
	// and leaves the new revision in the pending publish stage. The previous
	// version stays fully visible until commit; a parse or transaction failure
	// cannot leave old and new rows mixed. Empty entity sets are rejected.
	StageVersion(ctx context.Context, fileType string, entities []search.Entity[search.Value]) (SourceChecksum, error)

	// CurrentVersion returns the committed (possibly still pending) revision
	// of one source or ErrSourceVersionNotFound.
	CurrentVersion(ctx context.Context, source string) (SourceChecksum, error)

	// MarkPublished confirms that the given revision is live in memory. It is
	// conditional on the source still pointing at that revision: a newer
	// replacement makes the call a no-op returning false. Re-marking an
	// already published revision is idempotent and returns true.
	MarkPublished(ctx context.Context, source string, revision int64) (bool, error)

	Get(ctx context.Context, sourceID string, source search.SourceList) (*search.Entity[search.Value], error)
	ListBySource(ctx context.Context, lastSourceID string, source search.SourceList, limit int) ([]search.Entity[search.Value], error)
	ListAll(ctx context.Context) ([]search.Entity[search.Value], error)
	Checksums(ctx context.Context) ([]SourceChecksum, error)
}

func NewRepository(database db.DB) Repository {
	if database == nil {
		return &MockRepository{}
	}
	return &sqlRepository{db: database}
}

type sqlRepository struct {
	db db.DB
}

// canonicalEntity is one marshaled entity in the stable order used for both
// the content version hash and the row writes.
type canonicalEntity struct {
	entity search.Entity[search.Value]
	raw    []byte
}

// canonicalize copies the entities, forces the source, sorts by SourceID and
// returns the rows to write plus the content version (sha256 over canonical
// JSON). Identical input always produces the same ordering and version.
func canonicalize(fileType string, entities []search.Entity[search.Value]) ([]canonicalEntity, string, error) {
	indexed := append([]search.Entity[search.Value](nil), entities...)
	for i := range indexed {
		indexed[i].Source = search.SourceList(fileType)
	}
	sort.Slice(indexed, func(i, j int) bool {
		return indexed[i].SourceID < indexed[j].SourceID
	})

	rows := make([]canonicalEntity, 0, len(indexed))
	h := sha256.New()
	for i := range indexed {
		raw, err := json.Marshal(indexed[i])
		if err != nil {
			return nil, "", fmt.Errorf("json marshal: %w", err)
		}
		writeChecksum(h, indexed[i].SourceID, raw)
		rows = append(rows, canonicalEntity{entity: indexed[i], raw: raw})
	}
	return rows, hex.EncodeToString(h.Sum(nil)), nil
}

func (r *sqlRepository) Upsert(ctx context.Context, fileType string, entities []search.Entity[search.Value]) error {
	_, err := r.replace(ctx, fileType, entities, publishStatePublished)
	return err
}

func (r *sqlRepository) StageVersion(ctx context.Context, fileType string, entities []search.Entity[search.Value]) (SourceChecksum, error) {
	if len(entities) == 0 {
		return SourceChecksum{}, fmt.Errorf("refusing to stage %s: no entities", fileType)
	}
	return r.replace(ctx, fileType, entities, publishStatePending)
}

// replace swaps one source's rows for a new version in a single transaction:
//
//  1. lock the source summary row so replacements for the same source assign
//     revisions in commit order,
//  2. insert the new version rows while the previous rows remain in place
//     (uniqueness is scoped by version),
//  3. delete every older version of the same source,
//  4. point the summary row at the new revision.
//
// Any failure rolls the whole transaction back, leaving the prior version
// intact with no mixed old/new rows or wrong checksum.
func (r *sqlRepository) replace(ctx context.Context, fileType string, entities []search.Entity[search.Value], state string) (SourceChecksum, error) {
	var out SourceChecksum

	tx, err := r.db.BeginTx(ctx, nil)
	if err != nil {
		return out, fmt.Errorf("begin ingest replace: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	if len(entities) == 0 {
		if err := r.deleteEntities(ctx, tx, fileType); err != nil {
			return out, err
		}
		if _, err := tx.ExecContext(ctx, "DELETE FROM ingest_sources WHERE source = ?;", fileType); err != nil {
			return out, fmt.Errorf("deleting %s ingest checksum: %w", fileType, err)
		}
		if err := tx.Commit(); err != nil {
			return out, fmt.Errorf("commit empty ingest replace: %w", err)
		}
		return out, nil
	}

	revision, err := lockSourceRevision(ctx, tx, fileType)
	if err != nil {
		return out, err
	}
	revision++

	rows, version, err := canonicalize(fileType, entities)
	if err != nil {
		return out, err
	}

	for i := range rows {
		if err := insertEntity(ctx, tx, rows[i], version); err != nil {
			if isDuplicateKey(err) {
				return out, fmt.Errorf("upserting (%s) %s/%s entity: %w: %w", fileType, rows[i].entity.Source, rows[i].entity.SourceID, ErrConcurrentReplacement, err)
			}
			return out, fmt.Errorf("upserting (%s) %s/%s entity: %w", fileType, rows[i].entity.Source, rows[i].entity.SourceID, err)
		}
	}

	// Remove rows belonging to any previous version of this source only. Other
	// sources are untouched by this transaction.
	if _, err := tx.ExecContext(ctx,
		"DELETE FROM ingested_entities WHERE source = ? AND version <> ?;",
		fileType, version); err != nil {
		return out, fmt.Errorf("deleting %s previous entities: %w", fileType, err)
	}

	if _, err := tx.ExecContext(ctx, "DELETE FROM ingest_sources WHERE source = ?;", fileType); err != nil {
		return out, fmt.Errorf("deleting %s ingest checksum: %w", fileType, err)
	}

	_, err = tx.ExecContext(ctx,
		`INSERT INTO ingest_sources (source, revision, version, entity_count, checksum, publish_state, updated_at)
		 VALUES (?, ?, ?, ?, ?, ?, ?);`,
		fileType,
		revision,
		version,
		len(rows),
		version,
		state,
		time.Now().UTC(),
	)
	if err != nil {
		if isDuplicateKey(err) {
			return out, fmt.Errorf("inserting %s ingest checksum: %w: %w", fileType, ErrConcurrentReplacement, err)
		}
		return out, fmt.Errorf("inserting %s ingest checksum: %w", fileType, err)
	}

	if err := tx.Commit(); err != nil {
		if isDuplicateKey(err) {
			return out, fmt.Errorf("commit ingest replace: %w: %w", ErrConcurrentReplacement, err)
		}
		return out, fmt.Errorf("commit ingest replace: %w", err)
	}

	out = SourceChecksum{
		Source:      fileType,
		Revision:    revision,
		Version:     version,
		EntityCount: len(rows),
		Checksum:    version,
		Published:   state == publishStatePublished,
	}
	return out, nil
}

// lockSourceRevision locks the summary row for one source and returns its
// current revision. Missing rows report zero; the primary key on source still
// rejects two concurrent inserts.
func lockSourceRevision(ctx context.Context, tx db.Tx, fileType string) (int64, error) {
	var revision int64
	err := tx.QueryRowContext(ctx,
		"SELECT revision FROM ingest_sources WHERE source = ? FOR UPDATE;",
		fileType,
	).Scan(&revision)
	if errors.Is(err, sql.ErrNoRows) {
		return 0, nil
	}
	if err != nil {
		return 0, fmt.Errorf("locking %s ingest source: %w", fileType, err)
	}
	return revision, nil
}

// isDuplicateKey reports whether a driver error is a unique-constraint
// violation. Both supported drivers phrase the error with "duplicate".
func isDuplicateKey(err error) bool {
	return err != nil && strings.Contains(strings.ToLower(err.Error()), "duplicate")
}

func (r *sqlRepository) CurrentVersion(ctx context.Context, source string) (SourceChecksum, error) {
	row := r.db.QueryRowContext(ctx,
		`SELECT source, revision, version, entity_count, checksum, publish_state
		 FROM ingest_sources WHERE source = ? LIMIT 1;`,
		source,
	)
	sum, err := scanSourceChecksum(row)
	if errors.Is(err, sql.ErrNoRows) {
		return SourceChecksum{}, ErrSourceVersionNotFound
	}
	if err != nil {
		return SourceChecksum{}, fmt.Errorf("getting %s ingest version: %w", source, err)
	}
	return sum, nil
}

func (r *sqlRepository) MarkPublished(ctx context.Context, source string, revision int64) (bool, error) {
	res, err := r.db.ExecContext(ctx,
		`UPDATE ingest_sources SET publish_state = ?, updated_at = ? WHERE source = ? AND revision = ?;`,
		publishStatePublished,
		time.Now().UTC(),
		source,
		revision,
	)
	if err != nil {
		return false, fmt.Errorf("marking %s revision %d published: %w", source, revision, err)
	}
	if n, _ := res.RowsAffected(); n > 0 {
		return true, nil
	}

	// Zero rows is either a superseded revision or an already-published row
	// whose updated_at has second granularity (MySQL DATETIME). Re-read to
	// distinguish the two.
	current, err := r.CurrentVersion(ctx, source)
	if errors.Is(err, ErrSourceVersionNotFound) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	return current.Revision == revision, nil
}

func (r *sqlRepository) deleteEntities(ctx context.Context, tx db.Tx, fileType string) error {
	_, err := tx.ExecContext(ctx, "DELETE FROM ingested_entities WHERE source = ?;", fileType)
	if err != nil {
		return fmt.Errorf("deleting %s entities: %w", fileType, err)
	}
	return nil
}

func insertEntity(ctx context.Context, tx db.Tx, row canonicalEntity, version string) error {
	_, err := tx.ExecContext(ctx,
		`INSERT INTO ingested_entities (type, source, source_id, entity, version) VALUES (?, ?, ?, ?, ?);`,
		string(row.entity.Type),
		string(row.entity.Source),
		string(row.entity.SourceID),
		row.raw,
		version,
	)
	if err != nil {
		return fmt.Errorf("inserting ingested entity: %w", err)
	}
	return nil
}

func (r *sqlRepository) Get(ctx context.Context, sourceID string, source search.SourceList) (*search.Entity[search.Value], error) {
	qry := `SELECT entity FROM ingested_entities WHERE source_id = ? AND source = ? LIMIT 1;`

	rows, err := r.queryScanEntities(ctx, qry, sourceID, string(source))
	if err != nil {
		return nil, fmt.Errorf("getting ingested entity: %w", err)
	}
	if len(rows) > 0 {
		return &rows[0], nil
	}
	return nil, errors.New("no entity found")
}

func (r *sqlRepository) ListBySource(ctx context.Context, lastSourceID string, source search.SourceList, limit int) ([]search.Entity[search.Value], error) {
	if limit <= 0 {
		limit = listPageSize
	}
	qry := `SELECT entity FROM ingested_entities WHERE source = ? AND source_id > ? ORDER BY source_id ASC LIMIT ?;`

	rows, err := r.queryScanEntities(ctx, qry, string(source), lastSourceID, limit)
	if err != nil {
		return nil, fmt.Errorf("listing ingested entities by source: %w", err)
	}
	return rows, nil
}

func (r *sqlRepository) ListAll(ctx context.Context) ([]search.Entity[search.Value], error) {
	var all []search.Entity[search.Value]
	lastSource, lastID := "", ""
	for {
		batch, err := r.listAfter(ctx, lastSource, lastID, listPageSize)
		if err != nil {
			return nil, err
		}
		if len(batch) == 0 {
			break
		}
		all = append(all, batch...)
		last := batch[len(batch)-1]
		lastSource, lastID = string(last.Source), last.SourceID
		if len(batch) < listPageSize {
			break
		}
	}
	return all, nil
}

func (r *sqlRepository) listAfter(ctx context.Context, lastSource, lastSourceID string, limit int) ([]search.Entity[search.Value], error) {
	qry := `SELECT entity FROM ingested_entities
WHERE source > ? OR (source = ? AND source_id > ?)
ORDER BY source ASC, source_id ASC
LIMIT ?;`

	rows, err := r.queryScanEntities(ctx, qry, lastSource, lastSource, lastSourceID, limit)
	if err != nil {
		return nil, fmt.Errorf("listing all ingested entities: %w", err)
	}
	return rows, nil
}

// rowScanner is satisfied by both *sql.Row and *sql.Rows.
type rowScanner interface {
	Scan(dest ...any) error
}

func scanSourceChecksum(scanner rowScanner) (SourceChecksum, error) {
	var row SourceChecksum
	var state string
	if err := scanner.Scan(
		&row.Source,
		&row.Revision,
		&row.Version,
		&row.EntityCount,
		&row.Checksum,
		&state,
	); err != nil {
		return SourceChecksum{}, err
	}
	row.Published = state == publishStatePublished
	return row, nil
}

func (r *sqlRepository) Checksums(ctx context.Context) ([]SourceChecksum, error) {
	rows, err := r.db.QueryContext(ctx, `
SELECT source, revision, version, entity_count, checksum, publish_state
FROM ingest_sources ORDER BY source ASC;`)
	if err != nil {
		return nil, fmt.Errorf("listing ingest checksums: %w", err)
	}
	defer func() { _ = rows.Close() }()

	var out []SourceChecksum
	for rows.Next() {
		row, err := scanSourceChecksum(rows)
		if err != nil {
			return nil, fmt.Errorf("scanning ingest checksum: %w", err)
		}
		out = append(out, row)
	}
	return out, rows.Err()
}

func (r *sqlRepository) queryScanEntities(ctx context.Context, qry string, args ...interface{}) ([]search.Entity[search.Value], error) {
	rows, err := r.db.QueryContext(ctx, qry, args...)
	if err != nil {
		return nil, fmt.Errorf("query for ingested entities: %w", err)
	}
	defer func() { _ = rows.Close() }()

	var out []search.Entity[search.Value]
	for rows.Next() {
		var data string
		err = rows.Scan(&data)
		if err != nil {
			return nil, fmt.Errorf("scanning entity json: %w", err)
		}

		var row search.Entity[search.Value]
		err = json.NewDecoder(strings.NewReader(data)).Decode(&row)
		if err != nil {
			return nil, fmt.Errorf("json decode: %w", err)
		}

		out = append(out, row.Normalize())
	}
	return out, rows.Err()
}
