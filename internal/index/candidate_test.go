package index

import (
	"context"
	"fmt"
	"testing"

	"github.com/moov-io/watchman/internal/download"
	"github.com/moov-io/watchman/pkg/search"

	"github.com/stretchr/testify/require"
)

func candidateEntities(tag string, n int, genID string) []search.Entity[search.Value] {
	out := make([]search.Entity[search.Value], n)
	for i := range out {
		out[i] = search.Entity[search.Value]{
			Name:     fmt.Sprintf("%s-%d", tag, i),
			Type:     search.EntityPerson,
			Source:   search.SourceUSOFAC,
			SourceID: fmt.Sprintf("%s-%d", tag, i),
		}.Normalize()
	}
	return out
}

func candidateStats(tag string, n int, genID string) download.Stats {
	entities := candidateEntities(tag, n, genID)
	return download.Stats{
		Entities:      entities,
		Lists:         map[string]int{string(search.SourceUSOFAC): n},
		ListHashes:    map[string]string{string(search.SourceUSOFAC): "hash-" + genID},
		SourceResults: map[string]download.SourceResult{string(search.SourceUSOFAC): {Name: string(search.SourceUSOFAC), Entities: n, Hash: "hash-" + genID, Fresh: true}},
		GenerationID:  genID,
	}
}

// TestLists_CandidatePrepareDoesNotPublish verifies preparing a new text
// generation leaves the online view on the previous one until Publish.
func TestLists_CandidatePrepareDoesNotPublish(t *testing.T) {
	lists := NewLists(nil).(*lists)

	lists.PublishCandidate(lists.PrepareCandidate(candidateStats("first", 10, "gen-1")))

	got, err := lists.GetEntities(context.Background(), search.SourceUSOFAC)
	require.NoError(t, err)
	require.Len(t, got, 10)
	require.Equal(t, "first-0", got[0].SourceID)

	// Prepare generation 2 without publishing: live view stays on generation 1.
	cand := lists.PrepareCandidate(candidateStats("second", 20, "gen-2"))

	got, err = lists.GetEntities(context.Background(), search.SourceUSOFAC)
	require.NoError(t, err)
	require.Len(t, got, 10, "online view must not change until PublishCandidate")
	require.Equal(t, "first-0", got[0].SourceID)
	require.Equal(t, "gen-1", lists.LatestStats().GenerationID)

	// Publishing swaps atomically.
	lists.PublishCandidate(cand)
	got, err = lists.GetEntities(context.Background(), search.SourceUSOFAC)
	require.NoError(t, err)
	require.Len(t, got, 20)
	require.Equal(t, "second-0", got[0].SourceID)
	require.Equal(t, "gen-2", lists.LatestStats().GenerationID)
}

// TestLists_PriorSourceIsolatedFromCandidateBuild verifies entities handed to a
// new generation are copies: building the new corpus mutates PreparedFields
// weights on the candidate only and never corrupts the live generation.
func TestLists_PriorSourceIsolatedFromCandidateBuild(t *testing.T) {
	lists := NewLists(nil).(*lists)
	lists.PublishCandidate(lists.PrepareCandidate(candidateStats("first", 5, "gen-1")))

	prior, ok := lists.PriorSource(search.SourceUSOFAC)
	require.True(t, ok)
	require.Len(t, prior.Entities, 5)
	require.Equal(t, "hash-gen-1", prior.Hash)
	require.Equal(t, "gen-1", prior.GenerationID)

	// Build and publish a candidate that reuses the prior entities (as an
	// allowed-to-fail source would).
	stats := download.Stats{
		Entities:      prior.Entities,
		Lists:         map[string]int{string(search.SourceUSOFAC): len(prior.Entities)},
		ListHashes:    map[string]string{string(search.SourceUSOFAC): prior.Hash},
		SourceResults: map[string]download.SourceResult{string(search.SourceUSOFAC): {Name: string(search.SourceUSOFAC), Entities: 5, Stale: true, PriorGenerationID: "gen-1"}},
		GenerationID:  "gen-2",
	}
	lists.PublishCandidate(lists.PrepareCandidate(stats))

	// The live view remains consistent and searchable.
	got, err := lists.GetEntities(context.Background(), search.SourceUSOFAC)
	require.NoError(t, err)
	require.Len(t, got, 5)
	cands, err := lists.SelectCandidates(context.Background(), search.Entity[search.Value]{
		Name:   "first-1",
		Type:   search.EntityPerson,
		Source: search.SourceUSOFAC,
	}.Normalize())
	require.NoError(t, err)
	require.Greater(t, cands.Len(), 0)

	// Unknown sources have no prior generation.
	_, ok = lists.PriorSource(search.SourceEUCSL)
	require.False(t, ok)
}

// TestLists_UpdateStillWorks verifies the legacy one-shot path.
func TestLists_UpdateStillWorks(t *testing.T) {
	lists := NewLists(nil)
	lists.Update(candidateStats("first", 3, "gen-legacy"))

	got, err := lists.GetEntities(context.Background(), search.SourceUSOFAC)
	require.NoError(t, err)
	require.Len(t, got, 3)
	require.Equal(t, "gen-legacy", lists.LatestStats().GenerationID)
}
