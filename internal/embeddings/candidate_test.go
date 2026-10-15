package embeddings

import (
	"context"
	"testing"

	"github.com/moov-io/base/log"

	"github.com/stretchr/testify/require"
)

func newCandidateTestService(t *testing.T, provider Provider) *service {
	t.Helper()
	return &service{
		logger:   log.NewTestLogger(),
		provider: provider,
		index:    newVectorIndex(provider.Dimension()),
		cache:    &noopCache{},
		config: Config{
			BatchSize: 2,
			Provider: ProviderConfig{
				Dimension: provider.Dimension(),
			},
		},
	}
}

// TestService_BuildIndexCandidateDoesNotPublish verifies a candidate vector
// index does not serve searches until PublishIndexCandidate.
func TestService_BuildIndexCandidateDoesNotPublish(t *testing.T) {
	svc := newCandidateTestService(t, NewMockProvider(4))

	// Seed the live index.
	require.NoError(t, svc.BuildIndex(context.Background(), []string{"alpha"}, []string{"1"}))
	require.Equal(t, 1, svc.IndexSize())

	// Build a candidate for the next generation; the live index is untouched.
	candidate, err := svc.BuildIndexCandidate(context.Background(),
		[]string{"bravo", "charlie", "delta"}, []string{"2", "3", "4"})
	require.NoError(t, err)
	require.Equal(t, 1, svc.IndexSize(), "live index must not change before publish")

	// Search still hits the old generation.
	results, err := svc.Search(context.Background(), "alpha", 5)
	require.NoError(t, err)
	require.NotEmpty(t, results)
	require.Equal(t, "1", results[0].ID)

	// Publish swaps the online vector index atomically.
	svc.PublishIndexCandidate(candidate)
	require.Equal(t, 3, svc.IndexSize())
}

// TestService_FailedCandidateKeepsOldIndex verifies a provider failure while
// building the candidate leaves the previous index fully online.
func TestService_FailedCandidateKeepsOldIndex(t *testing.T) {
	// Fail after the first provider call (live seed succeeds, candidate fails).
	provider := NewMockProvider(4, WithMockFailAfter(1))
	svc := newCandidateTestService(t, provider)

	require.NoError(t, svc.BuildIndex(context.Background(), []string{"alpha"}, []string{"1"}))
	require.Equal(t, 1, svc.IndexSize())

	_, err := svc.BuildIndexCandidate(context.Background(), []string{"bravo", "charlie"}, []string{"2", "3"})
	require.Error(t, err)

	// The previous index is still intact. Query it directly so the deliberately
	// failing provider is not asked to encode the query again.
	require.Equal(t, 1, svc.IndexSize())
	results, err := svc.index.Search(provider.deterministicEmbedding("alpha"), 5)
	require.NoError(t, err)
	require.NotEmpty(t, results)
	require.Equal(t, "1", results[0].ID)
}
