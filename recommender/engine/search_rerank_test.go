package engine

import (
	"context"
	"fmt"
	"testing"

	"github.com/navidrome/navidrome/recommender/milvus"
	"github.com/stretchr/testify/assert"
)

type fakeVectorStore struct {
	searchResults map[string]map[string][]milvus.SearchResult
	embeddings    map[string]map[string][]float64
}

func (f *fakeVectorStore) Search(_ context.Context, collection string, vector []float64, opts milvus.SearchOptions) ([]milvus.SearchResult, error) {
	if f.searchResults == nil {
		return nil, nil
	}
	results := append([]milvus.SearchResult(nil), f.searchResults[collection][vectorKey(vector)]...)
	if len(opts.ExcludeNames) == 0 {
		return results, nil
	}
	excluded := make(map[string]struct{}, len(opts.ExcludeNames))
	for _, name := range opts.ExcludeNames {
		excluded[name] = struct{}{}
	}
	filtered := make([]milvus.SearchResult, 0, len(results))
	for _, result := range results {
		if _, skip := excluded[result.Name]; skip {
			continue
		}
		filtered = append(filtered, result)
	}
	return filtered, nil
}

func (f *fakeVectorStore) GetByNames(_ context.Context, collection string, names []string) (map[string][]float64, error) {
	result := make(map[string][]float64)
	if f.embeddings == nil {
		return result, nil
	}
	for _, name := range names {
		if embedding, ok := f.embeddings[collection][name]; ok {
			result[name] = embedding
		}
	}
	return result, nil
}

func TestSearchSingleModelUsesSeedWeightsInRRF(t *testing.T) {
	store := &fakeVectorStore{
		searchResults: map[string]map[string][]milvus.SearchResult{
			milvus.CollectionFlamingo: {
				vectorKey([]float64{1, 0}): []milvus.SearchResult{
					{Name: "heavy-seed-track", Distance: 0.9},
				},
				vectorKey([]float64{0, 1}): []milvus.SearchResult{
					{Name: "light-seed-track", Distance: 0.95},
				},
			},
		},
	}
	engine := New(DefaultConfig(), store, nil)

	candidates, err := engine.searchSingleModel(context.Background(), ModelFlamingo, []seedEmbedding{
		{Key: "seed-a", Embedding: []float64{1, 0}, Weight: 3},
		{Key: "seed-b", Embedding: []float64{0, 1}, Weight: 1},
	}, nil, 10)

	if err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	if assert.Len(t, candidates, 2) {
		assert.Equal(t, "heavy-seed-track", candidates[0].Name)
		assert.True(t, candidates[0].Score > candidates[1].Score)
	}
}

func TestRerankCandidatesUsesDiversityAndArtistPenalty(t *testing.T) {
	engine := New(DefaultConfig(), nil, nil)
	candidates := []candidate{
		{
			Name:       "artist-a - track-1",
			Score:      1.0,
			BaseScore:  1.0,
			Embeddings: map[string][]float64{ModelLyrics: []float64{1, 0}},
		},
		{
			Name:       "artist-a - track-2",
			Score:      0.98,
			BaseScore:  0.98,
			Embeddings: map[string][]float64{ModelLyrics: []float64{0.99, 0.01}},
		},
		{
			Name:       "artist-b - track-3",
			Score:      0.92,
			BaseScore:  0.92,
			Embeddings: map[string][]float64{ModelLyrics: []float64{0, 1}},
		},
	}

	diverse := engine.rerankCandidates(candidates, RecommendationRequest{
		Limit:     3,
		Diversity: 0.7,
		Models:    []string{ModelLyrics},
	})
	nonDiverse := engine.rerankCandidates(candidates, RecommendationRequest{
		Limit:     3,
		Diversity: 0,
		Models:    []string{ModelLyrics},
	})

	if assert.Len(t, diverse, 3) {
		assert.Equal(t, "artist-a - track-1", diverse[0].Name)
		assert.Equal(t, "artist-b - track-3", diverse[1].Name)
	}
	if assert.Len(t, nonDiverse, 3) {
		assert.Equal(t, "artist-a - track-2", nonDiverse[1].Name)
	}
}

func TestApplyNegativePenaltiesUsesAllActiveModels(t *testing.T) {
	engine := New(DefaultConfig(), nil, nil)
	candidates := []candidate{
		{
			Name:       "track-1",
			BaseScore:  1,
			Embeddings: map[string][]float64{ModelLyrics: []float64{1, 0}, ModelDescription: []float64{1, 0}},
		},
		{
			Name:       "track-2",
			BaseScore:  1,
			Embeddings: map[string][]float64{ModelLyrics: []float64{1, 0}, ModelDescription: []float64{0, 1}},
		},
	}

	engine.applyNegativePenalties(context.Background(), candidates, RecommendationRequest{
		Models: []string{ModelLyrics, ModelDescription},
		NegativeEmbeddings: map[string][][]float64{
			ModelLyrics:      [][]float64{{1, 0}},
			ModelDescription: [][]float64{{1, 0}},
		},
		NegativePromptPenalty: 0.85,
	})

	if assert.NotNil(t, candidates[0].NegativeSimilarity) && assert.NotNil(t, candidates[1].NegativeSimilarity) {
		assert.True(t, candidates[0].NegativePenalty > candidates[1].NegativePenalty)
		assert.InDelta(t, 0.15, candidates[0].NegativePenalty, 0.000001)
		assert.InDelta(t, 0.075, candidates[1].NegativePenalty, 0.000001)
	}
}

func vectorKey(vector []float64) string {
	return fmt.Sprintf("%v", vector)
}
