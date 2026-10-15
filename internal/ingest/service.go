package ingest

import (
	"bytes"
	"compress/gzip"
	"context"
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"slices"
	"strings"
	"time"

	"github.com/moov-io/base/log"
	"github.com/moov-io/watchman/internal/norm"
	"github.com/moov-io/watchman/pkg/search"
	"github.com/moov-io/watchman/pkg/sources/senzing"
)

// ErrVersionSuperseded is returned when an upload's revision was replaced by a
// newer upload before its in-memory publish was confirmed. The older upload
// must not become visible over the newer one.
var ErrVersionSuperseded = errors.New("ingest version superseded by a newer upload")

type Service interface {
	ReadEntitiesFromFile(ctx context.Context, name string, contents io.Reader) (FileEntities, error)

	// IngestUpload parses and validates a whole upload, replaces the source in
	// one isolated transaction, then only reports success once the in-memory
	// corpus, TFIDF and online search confirm the same revision. A failure
	// after the commit leaves a recoverable pending stage instead of exposing
	// partial state.
	IngestUpload(ctx context.Context, name string, contents io.Reader, cache IndexCache) (FileEntities, error)

	// RecoverPending finishes the publish stage for revisions committed to the
	// database but never confirmed in memory (e.g. after a crash between commit
	// and publish). It rewrites no entity rows.
	RecoverPending(ctx context.Context, cache IndexCache) error

	ReplaceEntities(ctx context.Context, fileType string, entities []search.Entity[search.Value]) error
	GetEntitiesBySource(ctx context.Context, source string) ([]search.Entity[search.Value], error)
}

// IndexCache is the in-memory ingest search index. Optional; callers may pass nil.
type IndexCache interface {
	RefreshIngest(ctx context.Context) error
}

// LiveIngestVersions is implemented by caches that report which source
// revision/version their live corpus, TFIDF and online search currently serve.
// Caches without it are trusted once RefreshIngest returns without error.
type LiveIngestVersions interface {
	IndexCache
	IngestVersion(source string) (SourceChecksum, bool)
}

func NewService(logger log.Logger, conf Config, repo Repository) Service {
	logger.Info().Logf("found %d ingest fileTypes, using %T", len(conf.Files), repo)

	return &service{
		logger: logger,
		conf:   conf,
		repo:   repo,
	}
}

type service struct {
	logger log.Logger
	conf   Config
	repo   Repository
}

type FileEntities struct {
	FileType string
	Entities []search.Entity[search.Value]
}

func (s *service) ReadEntitiesFromFile(ctx context.Context, name string, contents io.Reader) (FileEntities, error) {
	var out FileEntities
	var err error

	for fileType, schema := range s.conf.Files {
		if strings.EqualFold(name, fileType) {
			// Process the file according to the schema type
			switch Format(strings.ToLower(string(schema.Format))) {
			case FormatCSV:
				out, err = s.readEntitiesFromCSVFile(ctx, fileType, schema, contents)

			case FormatSenzing, FormatSenzingJSON, FormatSenzingJSONL:
				out, err = s.readEntitiesFromSenzingFile(ctx, fileType, contents)

			default:
				return out, fmt.Errorf("unknown format %v", schema.Format)
			}

			if err == nil {
				s.logger.With(log.Fields{
					"name":      log.String(name),
					"file_type": log.String(fileType),
				}).Logf("read %d entities", len(out.Entities))
			}
		}
	}
	if err != nil {
		return out, fmt.Errorf("reading entities from %s failed: %w", name, err)
	}

	// Merge the entities
	out.Entities = search.Merge(out.Entities)

	s.logger.With(log.Fields{
		"name": log.String(name),
	}).Logf("after merge have %d entities", len(out.Entities))

	if len(out.Entities) == 0 {
		return out, fmt.Errorf("schema %s not found", name)
	}
	return out, nil
}

func maybeDecompressBody(r io.Reader) io.Reader {
	header := make([]byte, 10)
	n, err := io.ReadFull(r, header)
	if err != nil || n < 2 {
		if n > 0 {
			return io.MultiReader(bytes.NewReader(header[:n]), r)
		}
		return r
	}

	gz, err := gzip.NewReader(io.MultiReader(bytes.NewReader(header), r))
	if err == nil {
		return gz
	}

	return io.MultiReader(bytes.NewReader(header), r)
}

func (s *service) readEntitiesFromCSVFile(ctx context.Context, name string, schema File, contents io.Reader) (FileEntities, error) {
	out := FileEntities{
		FileType: name,
	}

	r := csv.NewReader(maybeDecompressBody(contents))

	headers, err := r.Read()
	if err != nil {
		return out, fmt.Errorf("problem reading headers: %w", err)
	}
	for idx := range headers {
		// Sometimes we get UTF-8 replacement characters (U+FFFD) in files which can be ignored
		headers[idx] = strings.ReplaceAll(headers[idx], "\uFFFD", "")
	}

	for {
		// Read each row until we run out
		row, err := r.Read()
		if err != nil {
			if err == io.EOF {
				break
			}
			return out, fmt.Errorf("problem reading row: %w", err)
		}
		for idx := range row {
			// ignore UTF-8 replacement characters
			row[idx] = strings.ReplaceAll(row[idx], "\uFFFD", "")
		}

		var entity search.Entity[search.Value]

		// Read First set of common fields
		entity.Name = readColumnDef(headers, schema.Mapping.Name, row)
		entity.Type = search.EntityType(readType(headers, schema.Mapping.Type, row))
		entity.Source = search.SourceList(name)
		entity.SourceID = readColumnDef(headers, schema.Mapping.SourceID, row)

		// Read Business, Person, etc fields
		if schema.Mapping.Person != nil {
			entity.Person = &search.Person{
				Name:          entity.Name,
				AltNames:      readColumnArrayDef(headers, schema.Mapping.Person.AltNames, row),
				GovernmentIDs: readGovernmentIDs(headers, schema.Mapping.Person.GovernmentIDs, row),
			}

			birthDate, err := readTime(readColumnDef(headers, schema.Mapping.Person.BirthDate, row))
			if err != nil {
				return out, fmt.Errorf("reading person birth date: %w", err)
			}
			if !birthDate.IsZero() {
				entity.Person.BirthDate = &birthDate
			}
		}
		if schema.Mapping.Business != nil {
			entity.Business = &search.Business{
				Name:          entity.Name,
				AltNames:      readColumnArrayDef(headers, schema.Mapping.Business.AltNames, row),
				GovernmentIDs: readGovernmentIDs(headers, schema.Mapping.Business.GovernmentIDs, row),
			}

			created, err := readTime(readColumnDef(headers, schema.Mapping.Business.Created, row))
			if err != nil {
				return out, fmt.Errorf("reading business creation time: %w", err)
			}
			if !created.IsZero() {
				entity.Business.Created = &created
			}
		}

		// Read More common fields
		entity.Contact.PhoneNumbers = readColumnArrayDef(headers, schema.Mapping.Contact.PhoneNumbers, row)

		entity.Addresses = readAddresses(headers, schema.Mapping.Addresses, row)

		swiftCols := append(
			readColumnArrayDef(headers, schema.Mapping.Swift, row),
			readColumnArrayDef(headers, schema.Mapping.BIC, row)...,
		)
		entity.SwiftBICs = norm.ParseSwiftBICs(swiftCols...)

		out.Entities = append(out.Entities, entity.Normalize())
	}

	return out, nil
}

func readType(headers []string, def Type, row []string) string {
	return def.Default
}

func readColumnDef(headers []string, def ColumnDef, row []string) string {
	var fields []string

	if def.Column != "" {
		idx := slices.Index(headers, def.Column)
		if idx >= 0 && idx < len(row) {
			value := strings.TrimSpace(row[idx])
			if value != "" {
				fields = append(fields, value)
			}
		}
	}
	for _, col := range def.Merge {
		idx := slices.Index(headers, col)
		if idx >= 0 && idx < len(row) {
			value := strings.TrimSpace(row[idx])
			if value != "" {
				fields = append(fields, value)
			}
		}
	}

	return strings.Join(fields, " ")
}

func readColumnArrayDef(headers []string, def ColumnArrayDef, row []string) []string {
	var fields []string

	if def.Columns != "" {
		idx := slices.Index(headers, def.Columns)
		if idx >= 0 && idx < len(row) {
			value := strings.TrimSpace(row[idx])
			if value != "" {
				fields = append(fields, value)
			}
		}
	}

	var name string
	for _, col := range def.Merge {
		idx := slices.Index(headers, col)
		if idx >= 0 && idx < len(row) {
			value := strings.TrimSpace(row[idx])
			if value != "" {
				name += fmt.Sprintf(" %s", value)
			}
		}
	}

	name = strings.TrimSpace(name)

	if name != "" {
		fields = append(fields, name)
	}

	return fields
}

var (
	acceptedTimeFormats = []string{
		"2006-01-02",
		"1/2/2006",
		"01/02/2006",
	}
)

func readTime(value string) (tt time.Time, err error) {
	if value == "" {
		return
	}
	for _, fmt := range acceptedTimeFormats {
		tt, err = time.Parse(fmt, value)
		if err == nil {
			return tt, nil
		}
	}
	return
}

func readGovernmentIDs(headers []string, def GovernmentIDs, row []string) (out []search.GovernmentID) {
	var id search.GovernmentID
	// id.Type = search.GovernmentIDType()
	// id.Country = ""
	id.Identifier = readColumnDef(headers, def.Identifier, row)

	if id.Identifier != "" {
		return append(out, id)
	}
	return
}

func readAddresses(headers []string, def Addresses, row []string) (out []search.Address) {
	line1 := readColumnArrayDef(headers, def.Line1, row)
	line2 := readColumnArrayDef(headers, def.Line2, row)
	city := readColumnArrayDef(headers, def.City, row)
	state := readColumnArrayDef(headers, def.State, row)
	postalCode := readColumnArrayDef(headers, def.PostalCode, row)
	country := readColumnArrayDef(headers, def.Country, row)

	for idx := range line1 {
		var addr search.Address

		if idx < len(line1) {
			addr.Line1 = line1[idx]
		}
		if idx < len(line2) {
			addr.Line2 = line2[idx]
		}
		if idx < len(city) {
			addr.City = city[idx]
		}
		if idx < len(state) {
			addr.State = state[idx]
		}
		if idx < len(postalCode) {
			addr.PostalCode = postalCode[idx]
		}
		if idx < len(country) {
			addr.Country = country[idx]
		}

		if addr.Line1 != "" {
			out = append(out, addr)
		}
	}

	return
}

// IngestUpload runs the full upload pipeline:
//
//  1. Parse, merge and validate the whole upload. Any failure happens before
//     the database is touched, so the previously visible version is unchanged.
//  2. Replace this source's rows and summary in one isolated transaction and
//     leave the new revision pending. Identical content reuses the existing
//     revision (idempotent resubmit, no duplicate rows).
//  3. Publish to the in-memory corpus / TFIDF / online search generation and
//     require it to confirm the same revision.
//  4. Mark the revision published. A failure after the commit is recoverable:
//     the revision stays pending in the database and retrying the same upload
//     (or restarting) only repeats the publish stage.
func (s *service) IngestUpload(ctx context.Context, name string, contents io.Reader, cache IndexCache) (FileEntities, error) {
	parsed, err := s.ReadEntitiesFromFile(ctx, name, contents)
	if err != nil {
		return parsed, err
	}

	if s.repo == nil {
		return parsed, nil
	}

	staged, err := s.stageUpload(ctx, parsed.FileType, parsed.Entities)
	if err != nil {
		return parsed, fmt.Errorf("problem updating %s entities: %w", parsed.FileType, err)
	}

	if cache != nil {
		if err := cache.RefreshIngest(ctx); err != nil {
			// The revision is committed and stays pending: a retry of the same
			// content resumes here without replacing any database rows.
			return parsed, fmt.Errorf("problem refreshing %s ingest search index: %w", parsed.FileType, err)
		}

		// Require the live generation (summary, corpus and TFIDF share it) to
		// serve exactly the revision we staged.
		if live, ok := cache.(LiveIngestVersions); ok {
			current, found := live.IngestVersion(parsed.FileType)
			if !found || current.Revision != staged.Revision || current.Version != staged.Version {
				return parsed, ErrVersionSuperseded
			}
		}
	}

	if err := s.confirmVersion(ctx, staged); err != nil {
		return parsed, err
	}
	return parsed, nil
}

// stageUpload returns the current revision when the uploaded content identity
// already matches the committed (pending or published) version, and otherwise
// stages a new revision in an isolated transaction.
func (s *service) stageUpload(ctx context.Context, fileType string, entities []search.Entity[search.Value]) (SourceChecksum, error) {
	version, err := EntityVersion(entities)
	if err != nil {
		return SourceChecksum{}, fmt.Errorf("computing %s version: %w", fileType, err)
	}

	current, err := s.repo.CurrentVersion(ctx, fileType)
	switch {
	case err == nil && current.Version == version:
		// Same file contents: do not write rows or allocate a new revision,
		// even if the existing revision is still pending publication.
		return current, nil
	case err == nil:
		// Different contents: replace below.
	case errors.Is(err, ErrSourceVersionNotFound):
		// First upload for this source: replace below.
	default:
		return SourceChecksum{}, err
	}

	staged, err := s.repo.StageVersion(ctx, fileType, entities)
	if err != nil {
		return SourceChecksum{}, err
	}
	return staged, nil
}

// confirmVersion flips a staged revision to published. A revision that was
// replaced by a newer upload reports ErrVersionSuperseded instead of silently
// succeeding.
func (s *service) confirmVersion(ctx context.Context, version SourceChecksum) error {
	if version.Published {
		return nil
	}
	ok, err := s.repo.MarkPublished(ctx, version.Source, version.Revision)
	if err != nil {
		return fmt.Errorf("confirming %s revision %d: %w", version.Source, version.Revision, err)
	}
	if !ok {
		return ErrVersionSuperseded
	}
	return nil
}

// RecoverPending completes the publish stage for every committed revision that
// is still pending. The in-memory cache rebuilds from the committed rows, so
// recovery works after a restart even though the original file is gone.
func (s *service) RecoverPending(ctx context.Context, cache IndexCache) error {
	if s.repo == nil {
		return nil
	}

	sums, err := s.repo.Checksums(ctx)
	if err != nil {
		return fmt.Errorf("listing pending ingest versions: %w", err)
	}

	var pending []SourceChecksum
	for _, sum := range sums {
		if !sum.Published {
			pending = append(pending, sum)
		}
	}
	if len(pending) == 0 {
		return nil
	}

	s.logger.Info().Logf("recovering %d pending ingest version(s)", len(pending))

	if cache != nil {
		if err := cache.RefreshIngest(ctx); err != nil {
			return fmt.Errorf("rebuilding ingest corpus for pending versions: %w", err)
		}
	}

	var recoverErr error
	for _, p := range pending {
		if err := s.confirmVersion(ctx, p); err != nil {
			// A superseded pending revision was replaced by a newer upload,
			// which the refresh above also loaded. Anything else is retried on
			// the next recovery.
			if !errors.Is(err, ErrVersionSuperseded) {
				recoverErr = errors.Join(recoverErr, err)
			}
		}
	}
	return recoverErr
}

// ReplaceEntities replaces one source's rows without driving an in-memory
// publish. The next search observes the new checksum and lazily rebuilds, as
// before; uploads go through IngestUpload instead so publish is confirmed.
func (s *service) ReplaceEntities(ctx context.Context, fileType string, entities []search.Entity[search.Value]) error {
	if s.repo == nil {
		return nil
	}

	if len(entities) == 0 {
		if err := s.repo.Upsert(ctx, fileType, nil); err != nil {
			return fmt.Errorf("problem replacing %s entities: %w", fileType, err)
		}
		return nil
	}

	staged, err := s.repo.StageVersion(ctx, fileType, entities)
	if err != nil {
		return fmt.Errorf("problem replacing %s entities: %w", fileType, err)
	}
	if _, err := s.repo.MarkPublished(ctx, fileType, staged.Revision); err != nil {
		return fmt.Errorf("problem replacing %s entities: %w", fileType, err)
	}
	return nil
}

func (s *service) readEntitiesFromSenzingFile(ctx context.Context, name string, contents io.Reader) (FileEntities, error) {
	out := FileEntities{
		FileType: name,
	}

	sourceList := search.SourceList(name)

	entities, err := senzing.ReadEntities(maybeDecompressBody(contents), sourceList)
	if err != nil {
		return out, fmt.Errorf("reading senzing entities: %w", err)
	}

	out.Entities = entities

	return out, nil
}

const defaultPaginationLimit = 1000

func (s *service) GetEntitiesBySource(ctx context.Context, source string) ([]search.Entity[search.Value], error) {
	if s.repo == nil {
		return nil, fmt.Errorf("no repository configured")
	}

	var allEntities []search.Entity[search.Value]
	lastSourceID := ""
	limit := s.conf.PaginationLimit
	if limit <= 0 {
		limit = defaultPaginationLimit
	}

	for {
		entities, err := s.repo.ListBySource(ctx, lastSourceID, search.SourceList(source), limit)
		if err != nil {
			return nil, fmt.Errorf("listing entities by source %s: %w", source, err)
		}

		if len(entities) == 0 {
			break
		}

		allEntities = append(allEntities, entities...)
		lastSourceID = entities[len(entities)-1].SourceID

		if len(entities) < limit {
			break
		}
	}

	return allEntities, nil
}
