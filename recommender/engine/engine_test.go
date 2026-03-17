package engine

import (
	"testing"

	"github.com/navidrome/navidrome/recommender/milvus"
	"github.com/stretchr/testify/assert"
)

func TestModelConstants(t *testing.T) {
	assert.Equal(t, "muq_audio", ModelMuQAudio)
	assert.Equal(t, "muq_mulan", ModelMuQMulan)
}

func TestCollectionForModel(t *testing.T) {
	tests := []struct {
		model    string
		expected string
	}{
		{ModelMuQAudio, milvus.CollectionMuQAudio},
		{ModelMuQMulan, milvus.CollectionMuQMulan},
		{"unknown", milvus.CollectionMuQMulan},
	}

	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			assert.Equal(t, tt.expected, CollectionForModel(tt.model))
		})
	}
}

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()

	assert.Equal(t, 75, cfg.DefaultTopK)
	assert.Equal(t, []string{ModelMuQAudio, ModelMuQMulan}, cfg.DefaultModels)
	assert.Equal(t, "union", cfg.DefaultMerge)
	assert.Equal(t, 0.0, cfg.DefaultDiversity)
}

func TestConfig(t *testing.T) {
	cfg := Config{
		DefaultTopK:      100,
		DefaultModels:    []string{ModelMuQMulan},
		DefaultMerge:     "intersection",
		DefaultDiversity: 0.5,
	}

	assert.Equal(t, 100, cfg.DefaultTopK)
	assert.Equal(t, []string{ModelMuQMulan}, cfg.DefaultModels)
	assert.Equal(t, "intersection", cfg.DefaultMerge)
	assert.Equal(t, 0.5, cfg.DefaultDiversity)
}

func TestSeedTrack(t *testing.T) {
	seed := SeedTrack{
		TrackID:     "track123",
		LookupNames: []string{"Artist - Title"},
		Embedding:   []float64{0.1, 0.2, 0.3},
		Embeddings: map[string][]float64{
			ModelMuQMulan: {0.4, 0.5, 0.6},
		},
		Weight: 1.0,
	}

	assert.Equal(t, "track123", seed.TrackID)
	assert.Equal(t, []string{"Artist - Title"}, seed.LookupNames)
	assert.Equal(t, []float64{0.1, 0.2, 0.3}, seed.Embedding)
	assert.Equal(t, []float64{0.4, 0.5, 0.6}, seed.Embeddings[ModelMuQMulan])
	assert.Equal(t, 1.0, seed.Weight)
}

func TestRecommendationRequest(t *testing.T) {
	req := RecommendationRequest{
		Seeds: []SeedTrack{
			{TrackID: "seed1", Weight: 1.0},
			{TrackID: "seed2", Weight: 0.5},
		},
		Models:            []string{ModelMuQAudio, ModelMuQMulan},
		MergeStrategy:     "union",
		Limit:             25,
		ExcludeTrackIDs:   []string{"exclude1"},
		DislikedTrackIDs:  []string{"dislike1"},
		NegativePrompts:   []string{"sad", "slow"},
		Diversity:         0.3,
		ModelPriorities:   map[string]int{ModelMuQAudio: 2, ModelMuQMulan: 1},
		MinModelAgreement: 1,
	}

	assert.Len(t, req.Seeds, 2)
	assert.Equal(t, []string{ModelMuQAudio, ModelMuQMulan}, req.Models)
	assert.Equal(t, "union", req.MergeStrategy)
	assert.Equal(t, 25, req.Limit)
	assert.Equal(t, []string{"exclude1"}, req.ExcludeTrackIDs)
	assert.Equal(t, []string{"dislike1"}, req.DislikedTrackIDs)
	assert.Equal(t, []string{"sad", "slow"}, req.NegativePrompts)
	assert.Equal(t, 0.3, req.Diversity)
	assert.Equal(t, 2, req.ModelPriorities[ModelMuQAudio])
	assert.Equal(t, 1, req.MinModelAgreement)
}

func TestRecommendationItem(t *testing.T) {
	negSim := 0.15
	item := RecommendationItem{
		TrackID:            "track123",
		Score:              0.95,
		Models:             []string{ModelMuQAudio, ModelMuQMulan},
		NegativeSimilarity: &negSim,
	}

	assert.Equal(t, "track123", item.TrackID)
	assert.Equal(t, 0.95, item.Score)
	assert.Equal(t, []string{ModelMuQAudio, ModelMuQMulan}, item.Models)
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
	e := New(Config{}, nil, nil)

	assert.NotNil(t, e)
	assert.Equal(t, []string{ModelMuQAudio, ModelMuQMulan}, e.config.DefaultModels)
	assert.Equal(t, "union", e.config.DefaultMerge)
	assert.Equal(t, 75, e.config.DefaultTopK)
}

func TestNewEngineWithConfig(t *testing.T) {
	cfg := Config{
		DefaultTopK:      100,
		DefaultModels:    []string{ModelMuQMulan},
		DefaultMerge:     "intersection",
		DefaultDiversity: 0.5,
	}
	e := New(cfg, nil, nil)

	assert.NotNil(t, e)
	assert.Equal(t, 100, e.config.DefaultTopK)
	assert.Equal(t, []string{ModelMuQMulan}, e.config.DefaultModels)
	assert.Equal(t, "intersection", e.config.DefaultMerge)
	assert.Equal(t, 0.5, e.config.DefaultDiversity)
}

func TestBuildExcludeSet(t *testing.T) {
	e := New(DefaultConfig(), nil, nil)

	req := RecommendationRequest{
		Seeds: []SeedTrack{
			{TrackID: "seed1", LookupNames: []string{"Artist 1 - Song 1"}},
			{TrackID: "seed2", LookupNames: []string{"Artist 2 - Song 2"}},
		},
		ExcludeTrackIDs:  []string{"exclude1", "exclude2"},
		DislikedTrackIDs: []string{"dislike1"},
	}

	excludeSet := e.buildExcludeSet(req)

	assert.Contains(t, excludeSet, "seed1")
	assert.Contains(t, excludeSet, "seed2")
	assert.Contains(t, excludeSet, "Artist 1 - Song 1")
	assert.Contains(t, excludeSet, "Artist 2 - Song 2")
	assert.Contains(t, excludeSet, "exclude1")
	assert.Contains(t, excludeSet, "exclude2")
	assert.Contains(t, excludeSet, "dislike1")
}

func TestSeedLookupNamesDeDupsAndOrders(t *testing.T) {
	got := seedLookupNames(SeedTrack{
		TrackID:     "seed-1",
		LookupNames: []string{"Artist - Song", "seed-1", "  ", "Artist - Song", "Song"},
	})

	assert.Equal(t, []string{"seed-1", "Artist - Song", "Song"}, got)
}
