package download

import (
	"bytes"
	"cmp"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/moov-io/base/log"
	"github.com/moov-io/base/strx"
	"github.com/moov-io/base/telemetry"
	"github.com/moov-io/watchman/internal/tfidf"
	"github.com/moov-io/watchman/pkg/search"

	"golang.org/x/sync/errgroup"
)

// GeocodingService defines the interface for geocoding addresses.
// This is used to decouple the download package from the geocoding implementation.
type GeocodingService interface {
	GeocodeAddresses(ctx context.Context, addresses []search.Address) []search.Address
}

type Downloader interface {
	RefreshAll(ctx context.Context) (Stats, error)
}

func NewDownloader(logger log.Logger, conf Config, geocoder GeocodingService) (Downloader, error) {
	return &downloader{
		logger:   logger,
		conf:     conf,
		geocoder: geocoder,
	}, nil
}

type downloader struct {
	logger   log.Logger
	conf     Config
	geocoder GeocodingService

	// runtime is set by the Refresher for a coordinated, generation-pinned
	// refresh and reset afterward. Direct RefreshAll callers (tests, one-off
	// loads) leave it nil and retain the historical behavior.
	runtime *refreshRuntime
}

// refreshRuntime binds one RefreshAll call to an isolated generation.
type refreshRuntime struct {
	// workingDir is the candidate data directory; source files are read from
	// (and written into) it instead of the configured initial data directory.
	workingDir string

	// generationID is stamped onto the returned Stats.
	generationID string

	// prior provides the currently published generation so sources whose
	// failure is allowed can carry the previous generation's data.
	prior PriorData

	// workspace, when set, receives the previous generation's raw files when
	// at least one source has to be carried over (best effort, for recovery).
	workspace *GenerationWorkspace
}

// refreshRuntimeSetter is implemented by the production downloader so the
// Refresher can bind a generation without exposing the concrete type.
type refreshRuntimeSetter interface {
	SetRefreshRuntime(rt *refreshRuntime)
}

func (dl *downloader) SetRefreshRuntime(rt *refreshRuntime) {
	dl.runtime = rt
}

// effectiveDataDirectory returns the directory source loaders should use for
// one refresh: the isolated candidate directory when a runtime is bound,
// otherwise the configured initial data directory.
func effectiveDataDirectory(conf Config) string {
	if conf.workingDir != "" {
		return conf.workingDir
	}
	return initialDataDirectory(conf)
}

// knownDownloadableLists is the set of standard built-in lists that watchman
// can download.  It is used when validating IgnoredDownloadErrors entries.
var knownDownloadableLists = []search.SourceList{
	search.SourceEUCSL,
	search.SourceUKCSL,
	search.SourceUSCSL,
	search.SourceUSOFAC,
	search.SourceUSNonSDN,
	search.SourceUSFinCEN311,
	search.SourceUNCSL,
	search.SourceUSTEL,
}

func (dl *downloader) RefreshAll(ctx context.Context) (Stats, error) {
	ctx, span := telemetry.StartSpan(ctx, "refresh-all")
	defer span.End()

	start := time.Now()

	// Bind this call to its pinned generation: source files are staged in, and
	// read from, the candidate's isolated data directory.
	conf := dl.conf
	generationID := ""
	if dl.runtime != nil {
		conf.workingDir = dl.runtime.workingDir
		generationID = dl.runtime.generationID
	}

	stats := Stats{
		Lists:         make(map[string]int),
		ListHashes:    make(map[string]string),
		SourceResults: make(map[string]SourceResult),
		StartedAt:     time.Now().In(time.UTC),
		GenerationID:  generationID,
	}

	// Validate IgnoredDownloadErrors before spawning any goroutines so we can
	// return a clean error without having to tear down the errgroup machinery.
	ignoredLists := getIgnoredDownloadErrors(dl.conf)
	if err := validateIgnoredDownloadErrors(dl.conf, ignoredLists); err != nil {
		return stats, err
	}

	logger := dl.logger.Info().With(log.Fields{
		"initial_data_directory": log.String(expandInitialDir(effectiveDataDirectory(conf))),
		"generation_id":          log.String(generationID),
	})

	// Collect every source load failure (required and allowed) so provenance
	// and refresh Status can distinguish the two after producers finish.
	var failuresMu sync.Mutex
	var loadFailures []loadFailure
	recordFailure := func(source search.SourceList, cause error, allowed bool) {
		failuresMu.Lock()
		loadFailures = append(loadFailures, loadFailure{source: source, err: cause, allowed: allowed})
		failuresMu.Unlock()
	}

	g, ctx := errgroup.WithContext(ctx)
	preparedLists := make(chan preparedList, 10)

	// Start a goroutine to accumulate results
	resultsDone := make(chan struct{})
	go func() {
		defer close(resultsDone)

		for list := range preparedLists {
			entities := list.Entities

			// Apply geocoding to entities if service is available
			if dl.geocoder != nil {
				entities = dl.geocodeEntities(ctx, entities)
			}

			// Catch any non-normalized entities before they're added into the index.
			var normalized int
			for idx := range entities {
				if entities[idx].PreparedFields.Name == "" || len(entities[idx].PreparedFields.NameFields) == 0 {
					normalized++
					entities[idx] = entities[idx].Normalize()
				}
			}
			if normalized > 0 {
				logger.Warn().Logf("normalized %d entities before index inclusion - normalize in the mapper", normalized)
			}

			logger.Info().Logf("adding %d entities from %v", len(entities), list.ListName)

			name := string(list.ListName)
			stats.Lists[name] = len(entities)
			stats.ListHashes[name] = list.Hash
			stats.SourceResults[name] = SourceResult{
				Name:     name,
				Entities: len(entities),
				Hash:     list.Hash,
				Fresh:    true,
			}

			stats.Entities = append(stats.Entities, entities...)
		}
	}()

	// Create a WaitGroup to track all producers
	var producerWg sync.WaitGroup

	// Track what lists have been requested and loaded
	requestedLists := getIncludedLists(conf)
	var listsLoaded []search.SourceList

	if len(requestedLists) == 0 {
		logger.Warn().Log("no lists have been configured!")
	}
	logger.Info().Logf("starting list refresh of %v", requestedLists)

	// OFAC Records
	if slices.Contains(requestedLists, search.SourceUSOFAC) {
		listsLoaded = append(listsLoaded, search.SourceUSOFAC)

		producerWg.Add(1)
		g.Go(func() error {
			defer producerWg.Done()

			err := loadOFACRecords(ctx, logger, conf, preparedLists)
			if err != nil {
				if handleLoadError(logger, search.SourceUSOFAC, ignoredLists, err, recordFailure) {
					return nil
				}
				return fmt.Errorf("loading OFAC records: %w", err)
			}
			return nil
		})
	}

	// OFAC Non-SDN Records
	if slices.Contains(requestedLists, search.SourceUSNonSDN) {
		listsLoaded = append(listsLoaded, search.SourceUSNonSDN)

		producerWg.Add(1)
		g.Go(func() error {
			defer producerWg.Done()

			err := loadUSNonSDNRecords(ctx, logger, conf, preparedLists)
			if err != nil {
				if handleLoadError(logger, search.SourceUSNonSDN, ignoredLists, err, recordFailure) {
					return nil
				}
				return fmt.Errorf("loading OFAC Non-SDN records: %w", err)
			}
			return nil
		})
	}

	// CSL Records
	if slices.Contains(requestedLists, search.SourceUSCSL) {
		listsLoaded = append(listsLoaded, search.SourceUSCSL)

		producerWg.Add(1)
		g.Go(func() error {
			defer producerWg.Done()

			err := loadCSLUSRecords(ctx, logger, conf, preparedLists)
			if err != nil {
				if handleLoadError(logger, search.SourceUSCSL, ignoredLists, err, recordFailure) {
					return nil
				}
				return fmt.Errorf("loading US CSL records: %w", err)
			}
			return nil
		})
	}

	// FinCEN 311 Records
	if slices.Contains(requestedLists, search.SourceUSFinCEN311) {
		listsLoaded = append(listsLoaded, search.SourceUSFinCEN311)

		producerWg.Add(1)
		g.Go(func() error {
			defer producerWg.Done()

			err := loadFinCEN311Records(ctx, logger, conf, preparedLists)
			if err != nil {
				if handleLoadError(logger, search.SourceUSFinCEN311, ignoredLists, err, recordFailure) {
					return nil
				}
				return fmt.Errorf("loading FinCEN 311 records: %w", err)
			}
			return nil
		})
	}

	// UK CSL Records
	if slices.Contains(requestedLists, search.SourceUKCSL) {
		listsLoaded = append(listsLoaded, search.SourceUKCSL)

		producerWg.Add(1)
		g.Go(func() error {
			defer producerWg.Done()

			err := loadUKCSLRecords(ctx, logger, conf, preparedLists)
			if err != nil {
				if handleLoadError(logger, search.SourceUKCSL, ignoredLists, err, recordFailure) {
					return nil
				}
				return fmt.Errorf("loading UK CSL records: %w", err)
			}
			return nil
		})
	}

	// EU CSL Records
	if slices.Contains(requestedLists, search.SourceEUCSL) {
		listsLoaded = append(listsLoaded, search.SourceEUCSL)

		producerWg.Add(1)
		g.Go(func() error {
			defer producerWg.Done()

			err := loadEUCSLRecords(ctx, logger, conf, preparedLists)
			if err != nil {
				if handleLoadError(logger, search.SourceEUCSL, ignoredLists, err, recordFailure) {
					return nil
				}
				return fmt.Errorf("loading EU CSL records: %w", err)
			}
			return nil
		})
	}

	// OpenSanctions lists
	for _, list := range conf.OpenSanctions.Lists {
		listsLoaded = append(listsLoaded, normalizeListName(list.SourceList))
	}
	producerWg.Add(1)
	g.Go(func() error {
		defer producerWg.Done()

		err := loadOpensanctionsRecords(ctx, logger, conf, ignoredLists, preparedLists, recordFailure)
		if err != nil {
			return fmt.Errorf("loading opensanctions lists: %w", err)
		}
		return nil
	})

	// Senzing lists
	for _, list := range conf.Senzing {
		listsLoaded = append(listsLoaded, normalizeListName(list.SourceList))
	}
	producerWg.Add(1)
	g.Go(func() error {
		defer producerWg.Done()

		err := loadSenzingRecords(ctx, logger, conf, ignoredLists, preparedLists, recordFailure)
		if err != nil {
			return fmt.Errorf("loading senzing lists: %w", err)
		}
		return nil
	})

	// UN CSL Records
	if slices.Contains(requestedLists, search.SourceUNCSL) {
		listsLoaded = append(listsLoaded, search.SourceUNCSL)

		producerWg.Add(1)
		g.Go(func() error {
			defer producerWg.Done()

			err := loadUNCSLRecords(ctx, logger, conf, preparedLists)
			if err != nil {
				if handleLoadError(logger, search.SourceUNCSL, ignoredLists, err, recordFailure) {
					return nil
				}
				return fmt.Errorf("loading UN CSL records: %w", err)
			}
			return nil
		})
	}

	// US TEL Records
	if slices.Contains(requestedLists, search.SourceUSTEL) {
		listsLoaded = append(listsLoaded, search.SourceUSTEL)

		producerWg.Add(1)
		g.Go(func() error {
			defer producerWg.Done()

			err := loadUSTelRecords(ctx, logger, conf, preparedLists)
			if err != nil {
				if handleLoadError(logger, search.SourceUSTEL, ignoredLists, err, recordFailure) {
					return nil
				}
				return fmt.Errorf("loading US TEL records: %w", err)
			}
			return nil
		})
	}

	// Compare the configured lists against those we actually loaded.
	// Any extra lists are an error as we don't want to silently ignore them.
	if len(listsLoaded) > len(requestedLists) {
		close(preparedLists)

		return stats, fmt.Errorf("loaded extra lists: %#v loaded compared to %#v configured", listsLoaded, requestedLists)
	}
	if extra := findExtraLists(requestedLists, listsLoaded); extra != "" {
		close(preparedLists)

		return stats, fmt.Errorf("unknown lists: %v", extra)
	}

	// Add a goroutine to close the channel when all producers are done
	g.Go(func() error {
		producerWg.Wait()
		close(preparedLists)
		return nil
	})

	// Wait for both producers and consumer to finish
	err := g.Wait()
	<-resultsDone

	// Resolve failed sources. Allowed failures keep serving the explicit
	// previous generation's data for that source; required failures mark the
	// candidate unpublishable.
	applyLoadFailures(logger, dl.runtime, stats, loadFailures)

	if err != nil {
		// A required source failed: do not carry this generation forward.
		// Stats (including per-source errors) are returned only for Status.
		return stats, fmt.Errorf("problem loading lists: %v", err)
	}

	logger.Info().Logf("finished all lists: %v", time.Since(start))

	// Build TF-IDF index from all entity names (fresh downloads + any data
	// explicitly carried over from the previous generation).
	stats.TFIDFIndex = buildTFIDFIndex(logger, stats.Entities)

	stats.EndedAt = time.Now().In(time.UTC)

	return stats, nil
}

// loadFailure records one source load error and whether the source is allowed
// to fail.
type loadFailure struct {
	source  search.SourceList
	err     error
	allowed bool
}

// handleLoadError logs a source load failure, records it for provenance and
// returns true when the failure is allowed (the source is in ignoredLists).
func handleLoadError(
	logger log.Logger,
	source search.SourceList,
	ignoredLists []search.SourceList,
	cause error,
	record func(source search.SourceList, cause error, allowed bool),
) bool {
	allowed := slices.Contains(ignoredLists, source)
	if allowed {
		logger.Warn().Logf("ignoring error loading %s: %v", source, cause)
	}
	if record != nil {
		record(source, cause, allowed)
	}
	return allowed
}

// applyLoadFailures converts recorded source failures into Stats provenance.
// Allowed failures carry the previous generation's data for the failed source
// (when available) and are marked stale; required failures are surfaced with
// their error and publish nothing.
func applyLoadFailures(logger log.Logger, rt *refreshRuntime, stats Stats, failures []loadFailure) {
	for _, failure := range failures {
		name := string(failure.source)
		if failure.err == nil {
			continue
		}

		if !failure.allowed {
			if cur, ok := stats.SourceResults[name]; ok {
				cur.Error = failure.err.Error()
				cur.Fresh = false
				stats.SourceResults[name] = cur
			} else {
				stats.SourceResults[name] = SourceResult{
					Name:  name,
					Fresh: false,
					Error: failure.err.Error(),
				}
			}
			continue
		}

		result := SourceResult{
			Name:  name,
			Fresh: false,
			Stale: true,
			Error: failure.err.Error(),
		}

		if rt != nil && rt.prior != nil {
			if prior, ok := rt.prior.PriorSource(failure.source); ok && len(prior.Entities) > 0 {
				// Pin the previous generation's raw files too so the carry-over
				// survives a process exit and can be rebuilt after recovery.
				if rt.workspace != nil {
					if err := rt.workspace.CarryCurrentFiles(); err != nil {
						logger.Warn().Logf("carrying previous files for %s: %v", name, err)
					}
				}

				result.Entities = len(prior.Entities)
				result.Hash = prior.Hash
				result.PriorGenerationID = prior.GenerationID

				stats.Lists[name] = len(prior.Entities)
				stats.ListHashes[name] = prior.Hash
				stats.Entities = append(stats.Entities, prior.Entities...)

				logger.Warn().Logf("serving %d entities for %s from previous generation %s",
					len(prior.Entities), name, prior.GenerationID)
			} else {
				logger.Warn().Logf("no previous generation data available for %s; source will be empty", name)
			}
		}

		stats.SourceResults[name] = result
	}
}

// buildTFIDFIndex creates a TF-IDF index from all entity names.
// It extracts NameFields and AltNameFields from each entity's PreparedFields.
func buildTFIDFIndex(logger log.Logger, entities []search.Entity[search.Value]) *tfidf.Index {
	cfg := tfidf.ConfigFromEnvironment()
	idx := tfidf.NewIndex(cfg)

	if !cfg.Enabled {
		logger.Info().Log("TF-IDF indexing disabled")
		return idx
	}

	start := time.Now()

	// Collect all name terms as documents
	// Each entity's name (and alt names) is treated as a separate document
	var documents [][]string

	for i := range entities {
		// Add primary name fields
		if len(entities[i].PreparedFields.NameFields) > 0 {
			documents = append(documents, entities[i].PreparedFields.NameFields)
		}

		// Add alternate name fields
		for _, altFields := range entities[i].PreparedFields.AltNameFields {
			if len(altFields) > 0 {
				documents = append(documents, altFields)
			}
		}
	}

	idx.Build(documents)

	stats := idx.Stats()
	logger.Info().Logf("built TF-IDF index: %d documents, %d unique terms in %v",
		stats.TotalDocuments, stats.UniqueTerms, time.Since(start))

	return idx
}

func getIncludedLists(conf Config) []search.SourceList {
	out := make([]search.SourceList, 0, len(conf.IncludedLists))
	for _, v := range conf.IncludedLists {
		if list := normalizeListName(v); list != "" {
			out = append(out, list)
		}
	}

	fromEnvStr := strings.TrimSpace(os.Getenv("INCLUDED_LISTS"))
	if fromEnvStr != "" {
		for _, v := range strings.Split(fromEnvStr, ",") {
			if list := normalizeListName(search.SourceList(v)); list != "" {
				out = append(out, list)
			}
		}
	}

	// Now add senzing lists (normalized so that custom names and IncludedLists
	// entries are treated case-insensitively and whitespace is ignored, matching
	// the behavior of IgnoredDownloadErrors and INCLUDED_LISTS env var).
	for _, list := range conf.OpenSanctions.Lists {
		if list := normalizeListName(list.SourceList); list != "" {
			out = append(out, list)
		}
	}
	for _, list := range conf.Senzing {
		if list := normalizeListName(list.SourceList); list != "" {
			out = append(out, list)
		}
	}

	// Sort and remove duplicates
	slices.Sort(out)
	return slices.Compact(out)
}

// normalizeListName lowercases and trims a SourceList value. User-supplied list
// names (from YAML or env) may have accidental case or whitespace; we normalize
// so that matching against the lowercase constants and deduping work reliably.
func normalizeListName(s search.SourceList) search.SourceList {
	trimmed := strings.TrimSpace(string(s))
	if trimmed == "" {
		return ""
	}
	return search.SourceList(strings.ToLower(trimmed))
}

// getIgnoredDownloadErrors returns the deduplicated, sorted set of source-list
// names for which download/parse errors should be suppressed.  It merges the
// YAML-configured IgnoredDownloadErrors field with the IGNORED_DOWNLOAD_ERRORS
// environment variable (comma-separated, same format as IncludedLists).
func getIgnoredDownloadErrors(conf Config) []search.SourceList {
	out := make([]search.SourceList, 0, len(conf.IgnoredDownloadErrors))
	for _, v := range conf.IgnoredDownloadErrors {
		if list := normalizeListName(v); list != "" {
			out = append(out, list)
		}
	}

	if fromEnvStr := strings.TrimSpace(os.Getenv("IGNORED_DOWNLOAD_ERRORS")); fromEnvStr != "" {
		for _, v := range strings.Split(fromEnvStr, ",") {
			if list := normalizeListName(search.SourceList(v)); list != "" {
				out = append(out, list)
			}
		}
	}

	slices.Sort(out)
	return slices.Compact(out)
}

// validateIgnoredDownloadErrors returns an error if any entry in ignoredLists is
// not a known standard downloadable list and is not a configured
// OpenSanctions or Senzing source list.
func validateIgnoredDownloadErrors(conf Config, ignoredLists []search.SourceList) error {
	if len(ignoredLists) == 0 {
		return nil
	}

	// Build the full set of valid names from standard + custom lists.
	allValid := make([]search.SourceList, len(knownDownloadableLists))
	copy(allValid, knownDownloadableLists)
	for _, l := range conf.OpenSanctions.Lists {
		allValid = append(allValid, normalizeListName(l.SourceList))
	}
	for _, l := range conf.Senzing {
		allValid = append(allValid, normalizeListName(l.SourceList))
	}

	for _, ignored := range ignoredLists {
		// Normalize the ignored entry for the lookup so that validation itself is
		// robust to casing/whitespace in the (already mostly-normalized) input.
		if !slices.Contains(allValid, normalizeListName(ignored)) {
			return fmt.Errorf("unknown list %q in IgnoredDownloadErrors: not a known downloadable SourceList or configured OpenSanctions/Senzing list", ignored)
		}
	}
	return nil
}

func findExtraLists(config, loaded []search.SourceList) string {
	var extra []search.SourceList

	for _, c := range config {
		var found bool
		for _, l := range loaded {
			if c == l {
				found = true
				break
			}
		}
		if !found {
			extra = append(extra, c)
		}
	}

	var buf bytes.Buffer
	for idx, e := range extra {
		if idx > 0 {
			buf.WriteString(", ")
		}
		buf.WriteString(string(e))
	}
	return buf.String()
}

type preparedList struct {
	ListName search.SourceList
	Entities []search.Entity[search.Value]

	Hash string
}

func expandInitialDir(initialDir string) string {
	dir, err := filepath.Abs(initialDir)
	if err != nil {
		dir = initialDir
	}
	return dir
}

func initialDataDirectory(conf Config) string {
	return cmp.Or(os.Getenv("INITIAL_DATA_DIRECTORY"), conf.InitialDataDirectory)
}

// errorOnEmptyList reports whether an empty list should fail the refresh.
// The ERROR_ON_EMPTY_LIST environment variable takes precedence over the
// YAML-configured ErrorOnEmptyList field when it is set.
func errorOnEmptyList(conf Config) bool {
	if fromEnvStr := strings.TrimSpace(os.Getenv("ERROR_ON_EMPTY_LIST")); fromEnvStr != "" {
		return strx.Yes(fromEnvStr)
	}
	return conf.ErrorOnEmptyList
}

// geocodeEntities applies geocoding to all addresses in the given entities.
func (dl *downloader) geocodeEntities(ctx context.Context, entities []search.Entity[search.Value]) []search.Entity[search.Value] {
	if dl.geocoder == nil {
		return entities
	}

	var wg sync.WaitGroup
	wg.Add(len(entities))

	for i := range entities {
		go func(i int) {
			defer wg.Done()

			if len(entities[i].Addresses) > 0 {
				entities[i].Addresses = dl.geocoder.GeocodeAddresses(ctx, entities[i].Addresses)
			}
		}(i)
	}

	wg.Wait()

	return entities
}
