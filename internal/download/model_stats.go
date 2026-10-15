package download

import (
	"time"

	"github.com/moov-io/watchman/internal/tfidf"
	"github.com/moov-io/watchman/pkg/search"
)

// SourceResult records how a single source list participated in a refresh
// generation. It is the per-source provenance used by refresh Status.
type SourceResult struct {
	Name     string
	Entities int
	Hash     string

	// Fresh is true when the source was downloaded and parsed in this
	// generation. False means the source did not produce new data (a failure
	// or an empty result).
	Fresh bool

	// Stale is true when the source failed to refresh but the failure is
	// allowed (IgnoredDownloadErrors), so the previous generation's data for
	// the source was explicitly carried into this generation instead.
	Stale bool

	// PriorGenerationID identifies the generation the carried-over data came
	// from. Empty when no previous generation was available.
	PriorGenerationID string

	// Error holds the download/parse error for failed sources (required or
	// allowed), surfaced in refresh Status.
	Error string
}

type Stats struct {
	Entities []search.Entity[search.Value] `json:"-"`

	// TFIDFIndex contains precomputed IDF values for name term weighting.
	// This is built from all entity names after loading and used during search.
	TFIDFIndex *tfidf.Index `json:"-"`

	Lists      map[string]int    `json:"lists"`
	ListHashes map[string]string `json:"listHashes"`

	// SourceResults records per-source provenance (fresh download vs carried
	// over previous generation) for this generation.
	SourceResults map[string]SourceResult `json:"-"`

	StartedAt time.Time `json:"startedAt"`
	EndedAt   time.Time `json:"endedAt"`

	Version string `json:"version"`

	// GenerationID uniquely identifies the pinned set of source data that
	// produced these Stats. Empty for callers running RefreshAll outside a
	// coordinated refresh.
	GenerationID string `json:"generationId,omitempty"`
}

// PriorSource is the previous generation's data for one source list, used to
// carry an allowed-to-fail source into a new generation.
type PriorSource struct {
	Entities []search.Entity[search.Value]
	Hash     string

	// GenerationID is the published generation the data came from.
	GenerationID string
}

// PriorData provides access to the currently published generation. The
// downloader uses it to keep serving an allowed-to-fail source from the
// previous generation instead of dropping the source from the new generation.
type PriorData interface {
	PriorSource(source search.SourceList) (PriorSource, bool)
}
