package cmd

import (
	"context"

	"github.com/navidrome/navidrome/log"
	"github.com/navidrome/navidrome/model"
	"github.com/navidrome/navidrome/recommender"
	"github.com/navidrome/navidrome/server/subsonic"
)

var (
	newMilvusClientForRecommendations = recommender.NewMilvusClient
	newResolverForRecommendations     = recommender.NewResolver
	newEngineForRecommendations       = recommender.NewRecommendationEngine
)

// newRecommendationClient creates a Go recommendation client when possible and
// falls back to the no-op implementation when dependencies are unavailable.
func newRecommendationClient(ds model.DataStore) subsonic.RecommendationClient {
	milvusClient, _, err := newMilvusClientForRecommendations()
	if err != nil {
		log.Warn(context.Background(), "Recommendation engine unavailable, using no-op client", "error", err)
		return subsonic.NewNoopRecommendationClient()
	}

	resolver := newResolverForRecommendations(ds)
	engine := newEngineForRecommendations(milvusClient, resolver)
	if engine == nil {
		log.Warn(context.Background(), "Recommendation engine initialization returned nil, using no-op client")
		return subsonic.NewNoopRecommendationClient()
	}

	return subsonic.NewGoRecommendationClient(engine)
}
