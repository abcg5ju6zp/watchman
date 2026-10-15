package embeddings

import (
	"context"
	"database/sql"
	"fmt"
	"strconv"
	"strings"

	"github.com/moov-io/watchman/internal/db"

	"github.com/moov-io/base/telemetry"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

type sqlRepository struct {
	db    db.DB
	table string
}

func newSqlRepository(ctx context.Context, config Config, database db.DB) (*sqlRepository, error) {
	table := createTableName(config)

	ctx, span := telemetry.StartSpan(ctx, "embeddings-cache-sql", trace.WithAttributes(
		attribute.String("table", table),
	))
	defer span.End()

	if err := setupTable(ctx, table, database); err != nil {
		err = fmt.Errorf("creating %s failed: %w", table, err)
		span.RecordError(err)
		return nil, err
	}

	return &sqlRepository{
		db:    database,
		table: table,
	}, nil
}

const (
	maxDataSize = 512
	// hashSize holds a hex sha256 digest.
	hashSize = 64
)

func createTableName(config Config) string {
	table := fmt.Sprintf("cache_%s_dim%d", config.Provider.Model, config.Provider.Dimension)

	return strings.ReplaceAll(table, "-", "_")
}

func setupTable(ctx context.Context, table string, database db.DB) error {
	if database == nil {
		return fmt.Errorf("missing database: %v", database)
	}

	var create string

	switch database.Dialect() {
	case "mysql":
		// MySQL 9.5+ supports native JSON for storing vectors
		create = fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %s (
			data VARCHAR(%d) NOT NULL,
			content_hash CHAR(%d) NOT NULL,
			model_key VARCHAR(512) NOT NULL,
			dim INTEGER NOT NULL,
			embedding JSON NOT NULL,
			PRIMARY KEY (data)
		)`, table, maxDataSize, hashSize)

	case "postgres":
		// PostgreSQL using ARRAY type for native vector storage
		create = fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %s (
			data VARCHAR(%d) PRIMARY KEY,
			content_hash CHAR(%d) NOT NULL,
			model_key VARCHAR(512) NOT NULL,
			dim INTEGER NOT NULL,
			embedding DOUBLE PRECISION[] NOT NULL
		)`, table, maxDataSize, hashSize)

	default:
		return fmt.Errorf("unknown dialect: %v", database.Dialect())
	}

	if _, err := database.ExecContext(ctx, create); err != nil {
		return err
	}

	// Upgrade tables created before cache entries carried a verifiable
	// identity. The defaults can never validate (empty hash/key, zero
	// dimension), so existing rows self-heal as cache misses on the next
	// rebuild and get overwritten by the upsert.
	if err := ensureColumn(ctx, database, table, "content_hash",
		fmt.Sprintf("CHAR(%d) NOT NULL DEFAULT ''", hashSize)); err != nil {
		return err
	}
	if err := ensureColumn(ctx, database, table, "model_key", "VARCHAR(512) NOT NULL DEFAULT ''"); err != nil {
		return err
	}
	if err := ensureColumn(ctx, database, table, "dim", "INTEGER NOT NULL DEFAULT 0"); err != nil {
		return err
	}

	return nil
}

// ensureColumn adds a missing column to a table created by an older version.
func ensureColumn(ctx context.Context, database db.DB, table, column, definition string) error {
	var query string
	switch database.Dialect() {
	case "mysql":
		query = `SELECT 1 FROM information_schema.columns
			WHERE table_schema = DATABASE() AND LOWER(table_name) = LOWER(?) AND LOWER(column_name) = LOWER(?)`
	case "postgres":
		query = `SELECT 1 FROM information_schema.columns
			WHERE table_schema = current_schema() AND LOWER(table_name) = LOWER($1) AND LOWER(column_name) = LOWER($2)`
	default:
		return fmt.Errorf("unknown dialect: %v", database.Dialect())
	}

	var marker int
	switch err := database.QueryRowContext(ctx, query, table, column).Scan(&marker); {
	case err == nil:
		return nil // column already exists
	case err != sql.ErrNoRows:
		return fmt.Errorf("checking column %s.%s: %w", table, column, err)
	}

	alter := fmt.Sprintf("ALTER TABLE %s ADD COLUMN %s %s", table, column, definition)
	if _, err := database.ExecContext(ctx, alter); err != nil {
		return fmt.Errorf("adding column %s.%s: %w", table, column, err)
	}
	return nil
}

var _ Cache = (*sqlRepository)(nil)

func (r *sqlRepository) Get(ctx context.Context, data string) (cacheEntry, bool) {
	ctx, span := telemetry.StartSpan(ctx, "embeddings-cache-sql-get", trace.WithAttributes(
		attribute.String("table", r.table),
	))
	defer span.End()

	var query string
	switch r.db.Dialect() {
	case "mysql":
		query = fmt.Sprintf(`SELECT embedding, content_hash, model_key, dim FROM %s WHERE data = ?`, r.table)
	case "postgres":
		query = fmt.Sprintf(`SELECT embedding, content_hash, model_key, dim FROM %s WHERE data = $1`, r.table)
	default:
		span.RecordError(fmt.Errorf("unknown dialect: %v", r.db.Dialect()))
		return cacheEntry{}, false
	}

	var (
		embStr      string
		contentHash string
		modelKey    string
		dimension   int
	)
	err := r.db.QueryRowContext(ctx, query, data).Scan(&embStr, &contentHash, &modelKey, &dimension)
	if err != nil {
		if err != sql.ErrNoRows {
			span.RecordError(err)
		}
		return cacheEntry{}, false
	}

	var emb []float64
	switch r.db.Dialect() {
	case "mysql":
		// MySQL returns JSON array like "[0.1,0.2,0.3]"
		emb, err = parseVector(embStr)
	case "postgres":
		// PostgreSQL returns array like "{0.1,0.2,0.3}"
		emb, err = parsePostgresArray(embStr)
	default:
		return cacheEntry{}, false
	}

	if err != nil {
		span.RecordError(err)
		return cacheEntry{}, false
	}

	return cacheEntry{
		Vector:      emb,
		ModelKey:    modelKey,
		Dimension:   dimension,
		ContentHash: contentHash,
	}, true
}

// parseVector converts a JSON array string like "[0.1,0.2,0.3]" to a slice of float64 (for MySQL).
func parseVector(embStr string) ([]float64, error) {
	if len(embStr) < 2 || embStr[0] != '[' || embStr[len(embStr)-1] != ']' {
		return nil, fmt.Errorf("invalid vector format: %s", embStr)
	}
	trimmed := embStr[1 : len(embStr)-1]
	if trimmed == "" {
		return []float64{}, nil
	}
	parts := strings.Split(trimmed, ",")
	emb := make([]float64, len(parts))
	for i, part := range parts {
		val, err := strconv.ParseFloat(strings.TrimSpace(part), 64)
		if err != nil {
			return nil, fmt.Errorf("failed to parse float at index %d: %w", i, err)
		}
		emb[i] = val
	}
	return emb, nil
}

// parsePostgresArray converts a PostgreSQL array string like "{0.1,0.2,0.3}" to a slice of float64.
func parsePostgresArray(embStr string) ([]float64, error) {
	if len(embStr) < 2 || embStr[0] != '{' || embStr[len(embStr)-1] != '}' {
		return nil, fmt.Errorf("invalid postgres array format: %s", embStr)
	}
	trimmed := embStr[1 : len(embStr)-1]
	if trimmed == "" {
		return []float64{}, nil
	}
	parts := strings.Split(trimmed, ",")
	emb := make([]float64, len(parts))
	for i, part := range parts {
		val, err := strconv.ParseFloat(strings.TrimSpace(part), 64)
		if err != nil {
			return nil, fmt.Errorf("failed to parse float at index %d: %w", i, err)
		}
		emb[i] = val
	}
	return emb, nil
}

func (r *sqlRepository) Put(ctx context.Context, data string, entry cacheEntry) error {
	ctx, span := telemetry.StartSpan(ctx, "embeddings-cache-sql-put", trace.WithAttributes(
		attribute.String("table", r.table),
	))
	defer span.End()

	key := data
	if len(data) > maxDataSize {
		key = data[:maxDataSize]
	}

	var query string
	var embStr string

	switch r.db.Dialect() {
	case "mysql":
		query = fmt.Sprintf(`INSERT INTO %s (data, content_hash, model_key, dim, embedding)
			VALUES (?, ?, ?, ?, ?)
			ON DUPLICATE KEY UPDATE
				content_hash = VALUES(content_hash),
				model_key = VALUES(model_key),
				dim = VALUES(dim),
				embedding = VALUES(embedding)`, r.table)
		embStr = formatVector(entry.Vector)

	case "postgres":
		query = fmt.Sprintf(`INSERT INTO %s (data, content_hash, model_key, dim, embedding)
			VALUES ($1, $2, $3, $4, $5)
			ON CONFLICT (data) DO UPDATE SET
				content_hash = EXCLUDED.content_hash,
				model_key = EXCLUDED.model_key,
				dim = EXCLUDED.dim,
				embedding = EXCLUDED.embedding`, r.table)
		embStr = formatPostgresArray(entry.Vector)

	default:
		err := fmt.Errorf("unknown dialect: %v", r.db.Dialect())
		span.RecordError(err)
		return err
	}

	// The content hash always covers the full (untruncated) input text, so
	// same-prefix keys can never resolve to the wrong vector.
	contentHash := entry.ContentHash
	if contentHash == "" {
		contentHash = hashContent(data)
	}

	if _, err := r.db.ExecContext(ctx, query, key, contentHash, entry.ModelKey, entry.Dimension, embStr); err != nil {
		span.RecordError(err)
		return fmt.Errorf("writing embedding cache entry: %w", err)
	}
	return nil
}

// formatVector converts a slice of float64 to a JSON array string, e.g., "[0.1,0.2,0.3]" (for MySQL).
func formatVector(embedding []float64) string {
	if len(embedding) == 0 {
		return "[]"
	}
	var sb strings.Builder
	sb.WriteString("[")
	for i, val := range embedding {
		if i > 0 {
			sb.WriteString(",")
		}
		sb.WriteString(fmt.Sprintf("%f", val))
	}
	sb.WriteString("]")
	return sb.String()
}

// formatPostgresArray converts a slice of float64 to a PostgreSQL array string, e.g., "{0.1,0.2,0.3}".
func formatPostgresArray(embedding []float64) string {
	if len(embedding) == 0 {
		return "{}"
	}
	var sb strings.Builder
	sb.WriteString("{")
	for i, val := range embedding {
		if i > 0 {
			sb.WriteString(",")
		}
		sb.WriteString(fmt.Sprintf("%f", val))
	}
	sb.WriteString("}")
	return sb.String()
}
