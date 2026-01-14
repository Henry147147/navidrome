package engine

import (
	"testing"

	"github.com/navidrome/navidrome/recommender/milvus"
	"github.com/stretchr/testify/assert"
)

func TestModelConstants(t *testing.T) {
	assert.Equal(t, "lyrics", ModelLyrics)
	assert.Equal(t, "description", ModelDescription)
	assert.Equal(t, "flamingo", ModelFlamingo)
}

func TestCollectionForModel(t *testing.T) {
	tests := []struct {
		model    string
		expected string
	}{
		{ModelLyrics, milvus.CollectionLyrics},
		{ModelDescription, milvus.CollectionDescription},
		{ModelFlamingo, milvus.CollectionFlamingo},
		{"unknown", milvus.CollectionLyrics}, // defaults to lyrics
	}

	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			result := CollectionForModel(tt.model)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()

	assert.Equal(t, 75, cfg.DefaultTopK)
	assert.Equal(t, []string{ModelLyrics, ModelDescription, ModelFlamingo}, cfg.DefaultModels)
	assert.Equal(t, "union", cfg.DefaultMerge)
	assert.Equal(t, 0.0, cfg.DefaultDiversity)
}

func TestConfig(t *testing.T) {
	cfg := Config{
		DefaultTopK:      100,
		DefaultModels:    []string{ModelLyrics},
		DefaultMerge:     "intersection",
		DefaultDiversity: 0.5,
	}

	assert.Equal(t, 100, cfg.DefaultTopK)
	assert.Equal(t, []string{ModelLyrics}, cfg.DefaultModels)
	assert.Equal(t, "intersection", cfg.DefaultMerge)
	assert.Equal(t, 0.5, cfg.DefaultDiversity)
}

func TestSeedTrack(t *testing.T) {
	seed := SeedTrack{
		TrackID:   "track123",
		Embedding: []float64{0.1, 0.2, 0.3},
		Weight:    1.0,
	}

	assert.Equal(t, "track123", seed.TrackID)
	assert.Equal(t, []float64{0.1, 0.2, 0.3}, seed.Embedding)
	assert.Equal(t, 1.0, seed.Weight)
}

func TestRecommendationRequest(t *testing.T) {
	req := RecommendationRequest{
		Seeds: []SeedTrack{
			{TrackID: "seed1", Weight: 1.0},
			{TrackID: "seed2", Weight: 0.5},
		},
		Models:            []string{ModelLyrics, ModelDescription},
		MergeStrategy:     "union",
		Limit:             25,
		ExcludeTrackIDs:   []string{"exclude1"},
		DislikedTrackIDs:  []string{"dislike1"},
		NegativePrompts:   []string{"sad", "slow"},
		Diversity:         0.3,
		ModelPriorities:   map[string]int{ModelLyrics: 2, ModelDescription: 1},
		MinModelAgreement: 1,
	}

	assert.Len(t, req.Seeds, 2)
	assert.Equal(t, []string{ModelLyrics, ModelDescription}, req.Models)
	assert.Equal(t, "union", req.MergeStrategy)
	assert.Equal(t, 25, req.Limit)
	assert.Equal(t, []string{"exclude1"}, req.ExcludeTrackIDs)
	assert.Equal(t, []string{"dislike1"}, req.DislikedTrackIDs)
	assert.Equal(t, []string{"sad", "slow"}, req.NegativePrompts)
	assert.Equal(t, 0.3, req.Diversity)
	assert.Equal(t, 2, req.ModelPriorities[ModelLyrics])
	assert.Equal(t, 1, req.MinModelAgreement)
}

func TestRecommendationItem(t *testing.T) {
	negSim := 0.15
	item := RecommendationItem{
		TrackID:            "track123",
		Score:              0.95,
		Models:             []string{ModelLyrics, ModelDescription},
		NegativeSimilarity: &negSim,
	}

	assert.Equal(t, "track123", item.TrackID)
	assert.Equal(t, 0.95, item.Score)
	assert.Equal(t, []string{ModelLyrics, ModelDescription}, item.Models)
	assert.NotNil(t, item.NegativeSimilarity)
	assert.Equal(t, 0.15, *item.NegativeSimilarity)
}

func TestRecommendationResponse(t *testing.T) {
	resp := RecommendationResponse{
		Tracks: []RecommendationItem{
			{TrackID: "track1", Score: 0.9},
			{TrackID: "track2", Score: 0.8},
		},
		Warnings: []string{"warning1"},
	}

	assert.Len(t, resp.Tracks, 2)
	assert.Len(t, resp.Warnings, 1)
}

func TestNewEngineDefaults(t *testing.T) {
	cfg := Config{} // Zero values
	e := New(cfg, nil, nil)

	assert.NotNil(t, e)
	assert.Equal(t, []string{ModelLyrics, ModelDescription, ModelFlamingo}, e.config.DefaultModels)
	assert.Equal(t, "union", e.config.DefaultMerge)
	assert.Equal(t, 75, e.config.DefaultTopK)
}

func TestNewEngineWithConfig(t *testing.T) {
	cfg := Config{
		DefaultTopK:      100,
		DefaultModels:    []string{ModelLyrics},
		DefaultMerge:     "intersection",
		DefaultDiversity: 0.5,
	}
	e := New(cfg, nil, nil)

	assert.NotNil(t, e)
	assert.Equal(t, 100, e.config.DefaultTopK)
	assert.Equal(t, []string{ModelLyrics}, e.config.DefaultModels)
	assert.Equal(t, "intersection", e.config.DefaultMerge)
	assert.Equal(t, 0.5, e.config.DefaultDiversity)
}

func TestBuildExcludeSet(t *testing.T) {
	e := New(DefaultConfig(), nil, nil)

	req := RecommendationRequest{
		Seeds: []SeedTrack{
			{TrackID: "seed1"},
			{TrackID: "seed2"},
		},
		ExcludeTrackIDs:  []string{"exclude1", "exclude2"},
		DislikedTrackIDs: []string{"dislike1"},
	}

	excludeSet := e.buildExcludeSet(req)

	// Should contain all seed, excluded, and disliked tracks
	assert.Contains(t, excludeSet, "seed1")
	assert.Contains(t, excludeSet, "seed2")
	assert.Contains(t, excludeSet, "exclude1")
	assert.Contains(t, excludeSet, "exclude2")
	assert.Contains(t, excludeSet, "dislike1")
}
