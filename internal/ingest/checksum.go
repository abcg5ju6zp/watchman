package ingest

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"hash"
	"sort"

	"github.com/moov-io/watchman/pkg/search"
)

// Publish states recorded per source in ingest_sources.
const (
	// publishStatePending means the version's entity rows are committed, but
	// the in-memory corpus / TFIDF / search generation has not confirmed the
	// version yet. A retry of the same upload or a service restart completes
	// the publish stage.
	publishStatePending = "pending"

	// publishStatePublished means the version is confirmed live in the
	// in-memory search generation.
	publishStatePublished = "published"
)

// SourceChecksum is a cheap snapshot of one ingested fileType.
// Search compares these rows to the in-memory corpus instead of scanning entity JSON.
type SourceChecksum struct {
	Source      string
	Revision    int64
	Version     string
	EntityCount int
	Checksum    string

	// Published reports the persisted publish stage: false while the version is
	// committed but still waiting for in-memory confirmation.
	Published bool
}

// Fingerprint hashes a set of per-source checksums into one cache key.
func Fingerprint(sums []SourceChecksum) string {
	if len(sums) == 0 {
		return ""
	}
	sorted := append([]SourceChecksum(nil), sums...)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Source < sorted[j].Source
	})
	h := sha256.New()
	for _, s := range sorted {
		writeChecksum(h, s.Source, []byte(fmt.Sprintf("%d\x00%s", s.EntityCount, s.Checksum)))
	}
	return hex.EncodeToString(h.Sum(nil))
}

// EntityVersion returns the stable identity for a parsed file version: the
// sha256 of the canonical per-entity JSON, independent of input order.
//
// The same file contents (including a plain and gzip'd copy of the same CSV)
// always produce the same version, which makes re-uploading identical content
// idempotent: no new visible revision and no duplicate rows.
func EntityVersion(entities []search.Entity[search.Value]) (string, error) {
	return checksumEntities(entities)
}

func checksumEntities(entities []search.Entity[search.Value]) (string, error) {
	indexed := append([]search.Entity[search.Value](nil), entities...)
	sort.Slice(indexed, func(i, j int) bool {
		if indexed[i].Source != indexed[j].Source {
			return indexed[i].Source < indexed[j].Source
		}
		return indexed[i].SourceID < indexed[j].SourceID
	})

	h := sha256.New()
	for i := range indexed {
		raw, err := json.Marshal(indexed[i])
		if err != nil {
			return "", fmt.Errorf("json marshal: %w", err)
		}
		writeChecksum(h, indexed[i].SourceID, raw)
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func writeChecksum(h hash.Hash, sourceID string, raw []byte) {
	h.Write([]byte(sourceID))
	h.Write([]byte{0})
	h.Write(raw)
	h.Write([]byte{0})
}
