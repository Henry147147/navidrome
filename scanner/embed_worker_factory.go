package scanner

import (
	"context"

	"github.com/navidrome/navidrome/conf"
	"github.com/navidrome/navidrome/log"
	"github.com/navidrome/navidrome/recommender"
)

type embedWorkerFactory func(ctx context.Context) (*embeddingWorker, func(), error)

var scanEmbedWorkerFactory embedWorkerFactory = defaultEmbedWorkerFactory

func defaultEmbedWorkerFactory(ctx context.Context) (*embeddingWorker, func(), error) {
	// If all embedding stages are disabled, skip initialization.
	if !embeddingEnabled() {
		log.Info(ctx, "Embedding stages disabled; skipping embedder initialization")
		return nil, nil, nil
	}

	musicClient, musicCleanup, err := recommender.NewMusicEmbedClient()
	if err != nil {
		return nil, nil, err
	}

	milvusClient, milvusCleanup, err := recommender.NewMilvusClient()
	if err != nil {
		musicCleanup()
		return nil, nil, err
	}

	emb := recommender.NewEmbedder(musicClient, milvusClient)
	worker := newEmbeddingWorker(newGoEmbeddingClient(emb))
	cleanup := func() {
		_ = emb.Close()
		milvusCleanup()
		musicCleanup()
	}

	return worker, cleanup, nil
}

func embeddingEnabled() bool {
	cfg := conf.Server.Recommendations.Embedder
	return cfg.EnableLyrics || cfg.EnableDescription || cfg.EnableFlamingo
}
