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
func NewLlamaCppClient() (*llamacpp.Client, func(), error) {
	cfg := llamacpp.Config{
		LibraryPath:        conf.Server.Recommendations.Embedder.Llama.LibraryPath,
		TextModelPath:      conf.Server.Recommendations.Embedder.Llama.TextModelPath,
		AudioModelPath:     conf.Server.Recommendations.Embedder.Llama.AudioModelPath,
		AudioProjectorPath: conf.Server.Recommendations.Embedder.Llama.AudioProjectorPath,
		ContextSize:        uint32(conf.Server.Recommendations.Embedder.Llama.ContextSize),
		BatchSize:          uint32(conf.Server.Recommendations.Embedder.Llama.BatchSize),
		UBatchSize:         uint32(conf.Server.Recommendations.Embedder.Llama.UBatchSize),
		Threads:            conf.Server.Recommendations.Embedder.Llama.Threads,
		ThreadsBatch:       conf.Server.Recommendations.Embedder.Llama.ThreadsBatch,
		GPULayers:          conf.Server.Recommendations.Embedder.Llama.GPULayers,
		MainGPU:            conf.Server.Recommendations.Embedder.Llama.MainGPU,
		Timeout:            conf.Server.Recommendations.EmbedTimeout,
		MaxRetries:         3,
		RetryBackoff:       2 * time.Second,
	}

	client, err := llamacpp.NewClient(cfg)
	if err != nil {
		return nil, nil, err
	}

	cleanup := func() {
		_ = client.Close()
	}

	return client, cleanup, nil
}

// NewEmbedder creates the main embedder.
func NewEmbedder(llama *llamacpp.Client, milvusClient *milvus.Client) *embedder.Embedder {
	cfg := embedder.Config{
		BatchTimeout:      conf.Server.Recommendations.Embedder.BatchTimeout,
		BatchSize:         conf.Server.Recommendations.Embedder.BatchSize,
		EnableLyrics:      conf.Server.Recommendations.Embedder.EnableLyrics,
		EnableDescription: conf.Server.Recommendations.Embedder.EnableDescription,
		EnableFlamingo:    conf.Server.Recommendations.Embedder.EnableFlamingo,
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
