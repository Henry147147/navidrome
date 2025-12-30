package recommender

import (
	"context"
	"time"

	"github.com/google/wire"
	"github.com/navidrome/navidrome/conf"
	"github.com/navidrome/navidrome/model"
	"github.com/navidrome/navidrome/recommender/embedder"
	"github.com/navidrome/navidrome/recommender/engine"
	"github.com/navidrome/navidrome/recommender/llamacpp"
	"github.com/navidrome/navidrome/recommender/milvus"
	"github.com/navidrome/navidrome/recommender/resolver"
)

// Set provides all recommender dependencies for Wire.
var Set = wire.NewSet(
	NewMilvusClient,
	NewLlamaCppClient,
	NewEmbedder,
	NewRecommendationEngine,
	NewResolver,
	wire.Bind(new(Embedder), new(*embedder.Embedder)),
	wire.Bind(new(engine.TrackNameResolver), new(*resolver.Resolver)),
)

// NewMilvusClient creates a Milvus client from configuration.
func NewMilvusClient() (*milvus.Client, func(), error) {
	ctx := context.Background()
	cfg := milvus.Config{
		URI:        conf.Server.Recommendations.Milvus.URI,
		Timeout:    conf.Server.Recommendations.Milvus.Timeout,
		MaxRetries: conf.Server.Recommendations.Milvus.MaxRetries,
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

// NewLlamaCppClient creates a llama.cpp client from configuration.
func NewLlamaCppClient() *llamacpp.Client {
	cfg := llamacpp.Config{
		AudioEmbedURL:    conf.Server.Recommendations.Embedder.LlamaCppAudioURL,
		AudioDescribeURL: conf.Server.Recommendations.Embedder.LlamaCppDescribeURL,
		TextEmbedURL:     conf.Server.Recommendations.Embedder.LlamaCppTextURL,
		Timeout:          conf.Server.Recommendations.EmbedTimeout,
		MaxRetries:       3,
		RetryBackoff:     2 * time.Second,
	}

	// Use defaults if not configured
	if cfg.AudioEmbedURL == "" {
		cfg.AudioEmbedURL = "http://localhost:8080/embed/audio"
	}
	if cfg.AudioDescribeURL == "" {
		cfg.AudioDescribeURL = "http://localhost:8081/describe"
	}
	if cfg.TextEmbedURL == "" {
		cfg.TextEmbedURL = "http://localhost:8082/embed/text"
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 10 * time.Minute
	}

	return llamacpp.NewClient(cfg)
}

// NewEmbedder creates the main embedder.
func NewEmbedder(llama *llamacpp.Client, milvusClient *milvus.Client) *embedder.Embedder {
	cfg := embedder.Config{
		BatchTimeout:       conf.Server.Recommendations.Embedder.BatchTimeout,
		BatchSize:          conf.Server.Recommendations.Embedder.BatchSize,
		EnableDescriptions: conf.Server.Recommendations.Embedder.EnableDescriptions,
		EnableTextEmbed:    conf.Server.Recommendations.Embedder.EnableTextEmbed,
	}

	// Use defaults if not configured
	if cfg.BatchTimeout <= 0 {
		cfg.BatchTimeout = 5 * time.Second
	}
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 50
	}

	return embedder.New(cfg, llama, milvusClient)
}

// NewRecommendationEngine creates the recommendation engine.
func NewRecommendationEngine(milvusClient *milvus.Client, res *resolver.Resolver) *engine.Engine {
	cfg := engine.Config{
		DefaultTopK:      conf.Server.Recommendations.DefaultLimit * 3,
		DefaultModels:    []string{engine.ModelMuQ, engine.ModelQwen3},
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
