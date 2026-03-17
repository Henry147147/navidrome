package recommender

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestCanonicalName(t *testing.T) {
	tests := []struct {
		name     string
		artist   string
		title    string
		expected string
	}{
		{"both artist and title", "The Beatles", "Hey Jude", "The Beatles - Hey Jude"},
		{"only artist", "The Beatles", "", "The Beatles"},
		{"only title", "", "Hey Jude", "Hey Jude"},
		{"neither", "", "", ""},
		{"unicode characters", "日本語", "タイトル", "日本語 - タイトル"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := CanonicalName(tt.artist, tt.title)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestEmbedRequest(t *testing.T) {
	req := EmbedRequest{
		FilePath:  "/music/artist/album/track.flac",
		TrackName: "Artist - Track",
		TrackID:   "track-uuid-123",
		Artist:    "Artist",
		Title:     "Track",
		Album:     "Album",
	}

	assert.Equal(t, "/music/artist/album/track.flac", req.FilePath)
	assert.Equal(t, "Artist - Track", req.TrackName)
	assert.Equal(t, "track-uuid-123", req.TrackID)
	assert.Equal(t, "Artist", req.Artist)
	assert.Equal(t, "Track", req.Title)
	assert.Equal(t, "Album", req.Album)
}

func TestEmbedResult(t *testing.T) {
	result := EmbedResult{
		TrackName:         "Artist - Track",
		MuQAudioEmbedding: []float64{0.1, 0.2, 0.3},
		MuQMulanEmbedding: []float64{0.4, 0.5, 0.6},
	}

	assert.Equal(t, "Artist - Track", result.TrackName)
	assert.Len(t, result.MuQAudioEmbedding, 3)
	assert.Len(t, result.MuQMulanEmbedding, 3)
}

func TestEmbedResultPartialEmbeddings(t *testing.T) {
	result := EmbedResult{
		TrackName:         "Track",
		MuQMulanEmbedding: []float64{0.1, 0.2},
	}

	assert.NotNil(t, result.MuQMulanEmbedding)
	assert.Nil(t, result.MuQAudioEmbedding)
}

func TestStatusRequest(t *testing.T) {
	req := StatusRequest{
		TrackID:        "track-uuid-123",
		Artist:         "Artist",
		Title:          "Title",
		Album:          "Album",
		AlternateNames: []string{"alt1", "alt2", "alt3"},
	}

	assert.Equal(t, "track-uuid-123", req.TrackID)
	assert.Equal(t, "Artist", req.Artist)
	assert.Equal(t, "Title", req.Title)
	assert.Equal(t, "Album", req.Album)
	assert.Len(t, req.AlternateNames, 3)
}

func TestStatusResult(t *testing.T) {
	result := StatusResult{
		Embedded:           true,
		HasSharedEmbedding: true,
		HasAudioEmbedding:  true,
		CanonicalName:      "Artist - Title",
	}

	assert.True(t, result.Embedded)
	assert.True(t, result.HasSharedEmbedding)
	assert.True(t, result.HasAudioEmbedding)
	assert.Equal(t, "Artist - Title", result.CanonicalName)
}

func TestStatusResultNotEmbedded(t *testing.T) {
	result := StatusResult{}

	assert.False(t, result.Embedded)
	assert.False(t, result.HasSharedEmbedding)
	assert.False(t, result.HasAudioEmbedding)
	assert.Empty(t, result.CanonicalName)
}

func TestEmbeddingData(t *testing.T) {
	data := EmbeddingData{
		Name:      "Artist - Track",
		Embedding: []float64{0.1, 0.2, 0.3, 0.4},
		Offset:    0.5,
		ModelID:   "muq_mulan",
	}

	assert.Equal(t, "Artist - Track", data.Name)
	assert.Len(t, data.Embedding, 4)
	assert.Equal(t, 0.5, data.Offset)
	assert.Equal(t, "muq_mulan", data.ModelID)
}

func TestSearchOptions(t *testing.T) {
	opts := SearchOptions{
		TopK:         100,
		ExcludeNames: []string{"exclude1", "exclude2"},
	}

	assert.Equal(t, 100, opts.TopK)
	assert.Len(t, opts.ExcludeNames, 2)
}

func TestSearchResult(t *testing.T) {
	result := SearchResult{
		Name:     "Found Track",
		Distance: 0.123,
		TrackID:  "track-id",
	}

	assert.Equal(t, "Found Track", result.Name)
	assert.InDelta(t, 0.123, result.Distance, 0.0001)
	assert.Equal(t, "track-id", result.TrackID)
}

func TestAudioEmbedRequest(t *testing.T) {
	req := AudioEmbedRequest{
		AudioPath:  "/path/to/audio.mp3",
		SampleRate: 24000,
		BatchID:    "batch-123",
	}

	assert.Equal(t, "/path/to/audio.mp3", req.AudioPath)
	assert.Equal(t, 24000, req.SampleRate)
	assert.Equal(t, "batch-123", req.BatchID)
}

func TestAudioEmbedResponse(t *testing.T) {
	resp := AudioEmbedResponse{
		Embedding: []float64{0.1, 0.2, 0.3},
		ModelID:   "muq_audio",
		Duration:  180.5,
		Error:     "",
	}

	assert.Len(t, resp.Embedding, 3)
	assert.Equal(t, "muq_audio", resp.ModelID)
	assert.Equal(t, 180.5, resp.Duration)
	assert.Empty(t, resp.Error)
}

func TestTextEmbedRequest(t *testing.T) {
	req := TextEmbedRequest{
		Text:    "shared semantic query",
		ModelID: "muq_mulan",
	}

	assert.Equal(t, "shared semantic query", req.Text)
	assert.Equal(t, "muq_mulan", req.ModelID)
}

func TestTextEmbedResponse(t *testing.T) {
	resp := TextEmbedResponse{
		Embedding: make([]float64, 512),
		ModelID:   "muq_mulan",
		Dimension: 512,
		Error:     "",
	}

	assert.Len(t, resp.Embedding, 512)
	assert.Equal(t, "muq_mulan", resp.ModelID)
	assert.Equal(t, 512, resp.Dimension)
	assert.Empty(t, resp.Error)
}

func TestRecommendationRequest(t *testing.T) {
	now := time.Now()
	req := RecommendationRequest{
		UserID:   "user-123",
		UserName: "testuser",
		Limit:    25,
		Mode:     "similar",
		Seeds: []RecommendationSeed{
			{TrackID: "seed1", Weight: 1.0, Source: "history"},
			{TrackID: "seed2", Weight: 0.5, Source: "liked", PlayedAt: &now},
		},
		Diversity:             0.3,
		ExcludeTrackIDs:       []string{"exclude1"},
		LibraryIDs:            []int{1, 2},
		DislikedTrackIDs:      []string{"dislike1"},
		DislikedArtistIDs:     []string{"artist1"},
		DislikeStrength:       0.5,
		Models:                []string{"muq_mulan", "muq_audio"},
		MergeStrategy:         "union",
		ModelPriorities:       map[string]int{"muq_mulan": 1, "muq_audio": 2},
		MinModelAgreement:     1,
		NegativePrompts:       []string{"sad", "slow"},
		NegativePromptPenalty: 0.2,
		NegativeEmbeddings:    map[string][][]float64{"muq_mulan": {{0.1, 0.2}}},
	}

	assert.Equal(t, "user-123", req.UserID)
	assert.Equal(t, "testuser", req.UserName)
	assert.Equal(t, 25, req.Limit)
	assert.Equal(t, "similar", req.Mode)
	assert.Len(t, req.Seeds, 2)
	assert.Equal(t, 0.3, req.Diversity)
	assert.Len(t, req.ExcludeTrackIDs, 1)
	assert.Len(t, req.LibraryIDs, 2)
	assert.Len(t, req.DislikedTrackIDs, 1)
	assert.Len(t, req.DislikedArtistIDs, 1)
	assert.Equal(t, 0.5, req.DislikeStrength)
	assert.Len(t, req.Models, 2)
	assert.Equal(t, "union", req.MergeStrategy)
	assert.Len(t, req.ModelPriorities, 2)
	assert.Equal(t, 1, req.MinModelAgreement)
	assert.Len(t, req.NegativePrompts, 2)
	assert.Equal(t, 0.2, req.NegativePromptPenalty)
	assert.Len(t, req.NegativeEmbeddings, 1)
}

func TestRecommendationSeed(t *testing.T) {
	now := time.Now()
	seed := RecommendationSeed{
		TrackID:   "track-uuid",
		Weight:    1.5,
		Source:    "history",
		PlayedAt:  &now,
		Embedding: []float64{0.1, 0.2, 0.3},
		Embeddings: map[string][]float64{
			"muq_mulan": {0.4, 0.5},
			"muq_audio": {0.6, 0.7},
		},
	}

	assert.Equal(t, "track-uuid", seed.TrackID)
	assert.Equal(t, 1.5, seed.Weight)
	assert.Equal(t, "history", seed.Source)
	assert.NotNil(t, seed.PlayedAt)
	assert.Len(t, seed.Embedding, 3)
	assert.Len(t, seed.Embeddings, 2)
}

func TestRecommendationItem(t *testing.T) {
	negSim := 0.15
	item := RecommendationItem{
		TrackID:            "track-123",
		Score:              0.95,
		Reason:             "Shared semantic match",
		Models:             []string{"muq_mulan", "muq_audio"},
		NegativeSimilarity: &negSim,
	}

	assert.Equal(t, "track-123", item.TrackID)
	assert.Equal(t, 0.95, item.Score)
	assert.Equal(t, "Shared semantic match", item.Reason)
	assert.Len(t, item.Models, 2)
	assert.NotNil(t, item.NegativeSimilarity)
	assert.Equal(t, 0.15, *item.NegativeSimilarity)
}

func TestRecommendationItemWithoutNegativeSimilarity(t *testing.T) {
	item := RecommendationItem{
		TrackID: "track-123",
		Score:   0.9,
	}

	assert.Nil(t, item.NegativeSimilarity)
}

func TestRecommendationResponse(t *testing.T) {
	resp := RecommendationResponse{
		Tracks: []RecommendationItem{
			{TrackID: "track1", Score: 0.9},
			{TrackID: "track2", Score: 0.85},
			{TrackID: "track3", Score: 0.8},
		},
		Warnings: []string{"Using compatibility alias for legacy model name."},
	}

	assert.Len(t, resp.Tracks, 3)
	assert.Len(t, resp.Warnings, 1)
}

func TestRecommendationResponseTrackIDs(t *testing.T) {
	resp := RecommendationResponse{
		Tracks: []RecommendationItem{
			{TrackID: "track1", Score: 0.9},
			{TrackID: "", Score: 0.85},
			{TrackID: "track3", Score: 0.8},
		},
	}

	ids := resp.TrackIDs()
	assert.Len(t, ids, 2)
	assert.Contains(t, ids, "track1")
	assert.Contains(t, ids, "track3")
	assert.NotContains(t, ids, "")
}

func TestRecommendationResponseTrackIDsEmpty(t *testing.T) {
	resp := RecommendationResponse{}

	ids := resp.TrackIDs()
	assert.Empty(t, ids)
}

func TestRecommendationResponseTrackIDsAllEmpty(t *testing.T) {
	resp := RecommendationResponse{
		Tracks: []RecommendationItem{
			{TrackID: "", Score: 0.9},
			{TrackID: "", Score: 0.8},
		},
	}

	ids := resp.TrackIDs()
	assert.Empty(t, ids)
}

func TestMilvusConfigStruct(t *testing.T) {
	cfg := MilvusConfig{
		URI:        "http://localhost:19530",
		Timeout:    30 * time.Second,
		MaxRetries: 3,
		Dimensions: MilvusDimensions{
			MuQAudio: 1024,
			MuQMulan: 512,
		},
	}

	assert.Equal(t, "http://localhost:19530", cfg.URI)
	assert.Equal(t, 30*time.Second, cfg.Timeout)
	assert.Equal(t, 3, cfg.MaxRetries)
	assert.Equal(t, 1024, cfg.Dimensions.MuQAudio)
	assert.Equal(t, 512, cfg.Dimensions.MuQMulan)
}

func TestEngineConfigStruct(t *testing.T) {
	cfg := EngineConfig{
		DefaultTopK:      75,
		DefaultModels:    []string{"muq_audio", "muq_mulan"},
		DefaultMerge:     "union",
		DefaultDiversity: 0.3,
	}

	assert.Equal(t, 75, cfg.DefaultTopK)
	assert.Len(t, cfg.DefaultModels, 2)
	assert.Equal(t, "union", cfg.DefaultMerge)
	assert.Equal(t, 0.3, cfg.DefaultDiversity)
}
