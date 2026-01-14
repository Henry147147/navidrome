package subsonic

import (
	"context"

	"github.com/navidrome/navidrome/log"
	"github.com/navidrome/navidrome/recommender/engine"
)

// goRecommendationClient implements RecommendationClient using the Go engine.
type goRecommendationClient struct {
	engine *engine.Engine
}

// NewGoRecommendationClient creates a recommendation client that uses the Go engine.
func NewGoRecommendationClient(eng *engine.Engine) RecommendationClient {
	if eng == nil {
		return noopRecommendationClient{}
	}
	return &goRecommendationClient{engine: eng}
}

// Recommend generates recommendations using the Go engine.
func (c *goRecommendationClient) Recommend(ctx context.Context, mode string, payload RecommendationRequest) (*RecommendationResponse, error) {
	// Convert subsonic request to engine request
	seeds := make([]engine.SeedTrack, len(payload.Seeds))
	for i, s := range payload.Seeds {
		seeds[i] = engine.SeedTrack{
			TrackID:   s.TrackID,
			Embedding: s.Embedding,
			Weight:    s.Weight,
		}
	}

	req := engine.RecommendationRequest{
		Seeds:                 seeds,
		Models:                payload.Models,
		MergeStrategy:         payload.MergeStrategy,
		Limit:                 payload.Limit,
		ExcludeTrackIDs:       payload.ExcludeTrackIDs,
		DislikedTrackIDs:      payload.DislikedTrackIDs,
		NegativePrompts:       payload.NegativePrompts,
		NegativeEmbeddings:    payload.NegativeEmbeddings,
		NegativePromptPenalty: payload.NegativePromptPenalty,
		Diversity:             payload.Diversity,
		ModelPriorities:       payload.ModelPriorities,
		MinModelAgreement:     payload.MinModelAgreement,
	}

	log.Debug(ctx, "Calling Go recommendation engine",
		"mode", mode,
		"seeds", len(seeds),
		"limit", payload.Limit,
	)

	resp, err := c.engine.Recommend(ctx, req)
	if err != nil {
		return nil, err
	}

	// Convert engine response to subsonic response
	tracks := make([]RecommendationItem, len(resp.Tracks))
	for i, t := range resp.Tracks {
		tracks[i] = RecommendationItem{
			TrackID:            t.TrackID,
			Score:              t.Score,
			Models:             t.Models,
			NegativeSimilarity: t.NegativeSimilarity,
		}
	}

	return &RecommendationResponse{
		Tracks:   tracks,
		Warnings: resp.Warnings,
	}, nil
}
