package recommender

import (
	"context"
	"time"

	"github.com/google/wire"
	"github.com/navidrome/navidrome/conf"
	"github.com/navidrome/navidrome/model"
	"github.com/navidrome/navidrome/recommender/engine"
	"github.com/navidrome/navidrome/recommender/milvus"
	"github.com/navidrome/navidrome/recommender/resolver"
)

// Set provides all recommender dependencies for Wire.
var Set = wire.NewSet(
	NewMilvusClient,
	NewRecommendationEngine,
	NewResolver,
	wire.Bind(new(engine.TrackNameResolver), new(*resolver.Resolver)),
)

// NewMilvusClient creates a Milvus client from configuration.
func NewMilvusClient() (*milvus.Client, func(), error) {
	ctx := context.Background()
	cfg := milvus.Config{
		URI:        conf.Server.Recommendations.Milvus.URI,
		Timeout:    conf.Server.Recommendations.Milvus.Timeout,
		MaxRetries: conf.Server.Recommendations.Milvus.MaxRetries,
		Dimensions: milvus.Dimensions{
			Lyrics:      conf.Server.Recommendations.Milvus.Dimensions.Lyrics,
			Description: conf.Server.Recommendations.Milvus.Dimensions.Description,
			Flamingo:    conf.Server.Recommendations.Milvus.Dimensions.Flamingo,
		},
	}

	// Use defaults if not configured
	if cfg.URI == "" {
		cfg.URI = "http://localhost:19530"
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 30 * time.Second
	}
	if cfg.MaxRetries <= 0 {
		cfg.MaxRetries = 3
	}

	client, err := milvus.NewClient(ctx, cfg)
	if err != nil {
		return nil, nil, err
	}

	// Ensure collections exist
	if err := client.EnsureCollections(ctx); err != nil {
		client.Close()
		return nil, nil, err
	}

	cleanup := func() {
		client.Close()
	}

	return client, cleanup, nil
}

// NewRecommendationEngine creates the recommendation engine.
func NewRecommendationEngine(milvusClient *milvus.Client, res *resolver.Resolver) *engine.Engine {
	cfg := engine.Config{
		DefaultTopK:      conf.Server.Recommendations.DefaultLimit * 3,
		DefaultModels:    []string{engine.ModelLyrics, engine.ModelDescription, engine.ModelFlamingo},
		DefaultMerge:     "union",
		DefaultDiversity: conf.Server.Recommendations.Diversity,
	}

	if cfg.DefaultTopK <= 0 {
		cfg.DefaultTopK = 75
	}

	return engine.New(cfg, milvusClient, res)
}

// NewResolver creates a track name resolver.
func NewResolver(ds model.DataStore) *resolver.Resolver {
	return resolver.NewResolver(ds)
}
