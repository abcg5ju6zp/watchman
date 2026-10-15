-- Stable, content-derived identity for each uploaded file version plus a
-- recoverable publish stage for the in-memory search index.
--
--   version: sha256 of the canonical entity set for this upload. Identical
--            file contents always map to the same version, so a repeated
--            upload neither creates a new visible version nor duplicate rows.
--   revision: monotonic per-source sequence assigned when rows are replaced;
--             concurrent and out-of-order publishes are arbitrated with it.
--   publish_state: 'pending' rows are committed in the database but not yet
--             confirmed live in the corpus/TFIDF/search generation. Startup
--             recovery and upload retries finish publishing without rewriting
--             any entity rows.
ALTER TABLE ingested_entities ADD COLUMN version CHAR(64) NOT NULL DEFAULT '';

-- Two versions of one source may coexist only inside the replacement
-- transaction, so the uniqueness key includes the version. Readers outside
-- that transaction only ever see the committed generation.
ALTER TABLE ingested_entities DROP INDEX ingested_entities_id_uq;
CREATE UNIQUE INDEX ingested_entities_id_uq ON ingested_entities (source, version, source_id);

ALTER TABLE ingest_sources ADD COLUMN revision BIGINT NOT NULL DEFAULT 0;
ALTER TABLE ingest_sources ADD COLUMN version CHAR(64) NOT NULL DEFAULT '';
ALTER TABLE ingest_sources ADD COLUMN publish_state VARCHAR(20) NOT NULL DEFAULT 'published';
