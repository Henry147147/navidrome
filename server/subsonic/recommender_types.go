package subsonic

import (
	"context"
	"time"
)

// RecommendationClient provides playlist recommendations.
type RecommendationClient interface {
	Recommend(ctx context.Context, mode string, payload RecommendationRequest) (*RecommendationResponse, error)
}

// RecommendationSeed represents a seed track for recommendations.
type RecommendationSeed struct {
	TrackID   string     `json:"track_id"`
	Weight    float64    `json:"weight,omitempty"`
	Source    string     `json:"source"`
	PlayedAt  *time.Time `json:"played_at,omitempty"`
	Embedding []float64  `json:"embedding,omitempty"` // Direct embedding for text queries
}

// RecommendationRequest contains all parameters for generating recommendations.
type RecommendationRequest struct {
	UserID             string               `json:"user_id"`
	UserName           string               `json:"user_name"`
	Limit              int                  `json:"limit"`
	Mode               string               `json:"mode"`
	Seeds              []RecommendationSeed `json:"seeds"`
	Diversity          float64              `json:"diversity"`
	ExcludeTrackIDs    []string             `json:"exclude_track_ids,omitempty"`
	LibraryIDs         []int                `json:"library_ids,omitempty"`
	DislikedTrackIDs   []string             `json:"disliked_track_ids,omitempty"`
	DislikedArtistIDs  []string             `json:"disliked_artist_ids,omitempty"`
	DislikeStrength    float64              `json:"dislike_strength,omitempty"`
	ExcludePlaylistIDs []string             `json:"exclude_playlist_ids,omitempty"`

	// Multi-model support
	Models            []string       `json:"models,omitempty"`
	MergeStrategy     string         `json:"merge_strategy,omitempty"`
	ModelPriorities   map[string]int `json:"model_priorities,omitempty"`
	MinModelAgreement int            `json:"min_model_agreement,omitempty"`

	// Negative prompting
	NegativePrompts       []string               `json:"negative_prompts,omitempty"`
	NegativePromptPenalty float64                `json:"negative_prompt_penalty,omitempty"`
	NegativeEmbeddings    map[string][][]float64 `json:"negative_embeddings,omitempty"`
}

// RecommendationItem represents a single recommended track.
type RecommendationItem struct {
	TrackID            string   `json:"track_id"`
	Score              float64  `json:"score"`
	Reason             string   `json:"reason,omitempty"`
	Models             []string `json:"models,omitempty"`              // Models that contributed
	NegativeSimilarity *float64 `json:"negative_similarity,omitempty"` // Similarity to negative prompts
}

// RecommendationResponse contains recommendation results.
type RecommendationResponse struct {
	Tracks   []RecommendationItem `json:"tracks"`
	Warnings []string             `json:"warnings"`
}

// TrackIDs returns the track IDs from the response.
func (r RecommendationResponse) TrackIDs() []string {
	ids := make([]string, 0, len(r.Tracks))
	for _, item := range r.Tracks {
		if item.TrackID != "" {
			ids = append(ids, item.TrackID)
		}
	}
	return ids
}

// noopRecommendationClient is a no-op implementation.
type noopRecommendationClient struct{}

func (noopRecommendationClient) Recommend(context.Context, string, RecommendationRequest) (*RecommendationResponse, error) {
	return &RecommendationResponse{Warnings: []string{"recommendation service disabled"}}, nil
}

// NewNoopRecommendationClient creates a no-op recommendation client.
func NewNoopRecommendationClient() RecommendationClient {
	return noopRecommendationClient{}
}
