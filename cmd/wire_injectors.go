//go:build wireinject

package cmd

import (
	"context"
	"time"

	"github.com/google/wire"
	"github.com/navidrome/navidrome/core"
	"github.com/navidrome/navidrome/core/agents"
	"github.com/navidrome/navidrome/core/agents/lastfm"
	"github.com/navidrome/navidrome/core/agents/listenbrainz"
	"github.com/navidrome/navidrome/core/artwork"
	"github.com/navidrome/navidrome/core/metrics"
	"github.com/navidrome/navidrome/core/playback"
	"github.com/navidrome/navidrome/core/scrobbler"
	"github.com/navidrome/navidrome/db"
	"github.com/navidrome/navidrome/log"
	"github.com/navidrome/navidrome/model"
	"github.com/navidrome/navidrome/persistence"
	"github.com/navidrome/navidrome/plugins"
	"github.com/navidrome/navidrome/recommender/embedder"
	"github.com/navidrome/navidrome/recommender/llamacpp"
	"github.com/navidrome/navidrome/recommender/milvus"
	"github.com/navidrome/navidrome/scanner"
	"github.com/navidrome/navidrome/server"
	"github.com/navidrome/navidrome/server/events"
	"github.com/navidrome/navidrome/server/nativeapi"
	"github.com/navidrome/navidrome/server/public"
	"github.com/navidrome/navidrome/server/subsonic"
)

var allProviders = wire.NewSet(
	core.Set,
	artwork.Set,
	server.New,
	subsonic.New,
	nativeapi.New,
	public.New,
	subsonic.NewNoopRecommendationClient,
	persistence.New,
	lastfm.NewRouter,
	listenbrainz.NewRouter,
	events.GetBroker,
	newScanner,
	scanner.GetWatcher,
	plugins.GetManager,
	metrics.GetPrometheusInstance,
	db.Db,
	wire.Bind(new(agents.PluginLoader), new(plugins.Manager)),
	wire.Bind(new(scrobbler.PluginLoader), new(plugins.Manager)),
	wire.Bind(new(metrics.PluginLoader), new(plugins.Manager)),
	wire.Bind(new(core.Watcher), new(scanner.Watcher)),
)

// newScanner creates a scanner with embedder configured.
func newScanner(ctx context.Context, ds model.DataStore, cw artwork.CacheWarmer, broker events.Broker, pls core.Playlists, m metrics.Metrics) model.Scanner {
	// Initialize embedder if configured
	emb := initializeEmbedder(ctx)
	if emb != nil {
		return scanner.New(ctx, ds, cw, broker, pls, m, scanner.WithGoEmbedder(emb))
	}
	return scanner.New(ctx, ds, cw, broker, pls, m)
}

// initializeEmbedder creates and initializes the embedder service.
func initializeEmbedder(ctx context.Context) *embedder.Embedder {
	// Check if embedder should be enabled (using default paths as indicator)
	// In production, you might want to add a config flag to explicitly enable/disable
	llamaCfg := llamacpp.DefaultConfig()

	// Try to initialize llama.cpp client
	llamaClient, err := llamacpp.NewClient(llamaCfg)
	if err != nil {
		log.Warn(ctx, "Failed to initialize llama.cpp client, embedder disabled", err)
		return nil
	}

	// Try to initialize Milvus client
	milvusCfg := milvus.DefaultConfig()
	milvusClient, err := milvus.NewClient(ctx, milvusCfg)
	if err != nil {
		log.Warn(ctx, "Failed to initialize Milvus client, embedder disabled", err)
		_ = llamaClient.Close()
		return nil
	}

	// Create embedder with default configuration
	embedCfg := embedder.Config{
		BatchTimeout:      5 * time.Second,
		BatchSize:         50,
		EnableLyrics:      true,
		EnableDescription: true,
		EnableFlamingo:    true,
	}

	emb := embedder.New(embedCfg, llamaClient, milvusClient)
	log.Info(ctx, "Embedder service initialized successfully")
	return emb
}

func CreateDataStore() model.DataStore {
	panic(wire.Build(
		allProviders,
	))
}

func CreateServer() *server.Server {
	panic(wire.Build(
		allProviders,
	))
}

func CreateNativeAPIRouter(ctx context.Context) *nativeapi.Router {
	panic(wire.Build(
		allProviders,
	))
}

func CreateSubsonicAPIRouter(ctx context.Context) *subsonic.Router {
	panic(wire.Build(
		allProviders,
	))
}

func CreatePublicRouter() *public.Router {
	panic(wire.Build(
		allProviders,
	))
}

func CreateLastFMRouter() *lastfm.Router {
	panic(wire.Build(
		allProviders,
	))
}

func CreateListenBrainzRouter() *listenbrainz.Router {
	panic(wire.Build(
		allProviders,
	))
}

func CreateInsights() metrics.Insights {
	panic(wire.Build(
		allProviders,
	))
}

func CreatePrometheus() metrics.Metrics {
	panic(wire.Build(
		allProviders,
	))
}

func CreateScanner(ctx context.Context) model.Scanner {
	panic(wire.Build(
		allProviders,
	))
}

func CreateScanWatcher(ctx context.Context) scanner.Watcher {
	panic(wire.Build(
		allProviders,
	))
}

func GetPlaybackServer() playback.PlaybackServer {
	panic(wire.Build(
		allProviders,
	))
}

func getPluginManager() plugins.Manager {
	panic(wire.Build(
		allProviders,
	))
}

func GetPluginManager(ctx context.Context) plugins.Manager {
	manager := getPluginManager()
	manager.SetSubsonicRouter(CreateSubsonicAPIRouter(ctx))
	return manager
}
