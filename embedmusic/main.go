package main

import (
	"context"
	"database/sql"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"test/llama/musicembed"

	"github.com/navidrome/navidrome/log"
	"github.com/navidrome/navidrome/recommender/milvus"
)

const version = "1.0.0"

func main() {
	// Set up context with cancellation
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Handle signals gracefully
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigChan
		fmt.Fprintln(os.Stderr, "\nReceived interrupt signal, shutting down gracefully...")
		cancel()
	}()

	// Parse command-line flags
	cfg := parseFlags()

	// Initialize logging
	initLogging(cfg.Logging)

	log.Info(ctx, "Navidrome Embedder", "version", version)

	// Validate configuration
	if err := cfg.Validate(); err != nil {
		log.Error(ctx, "Invalid configuration", err)
		os.Exit(2)
	}

	// Print configuration in dry-run mode
	if cfg.CLI.DryRun {
		printConfig(ctx, cfg)
	}

	// Run the embedder
	if err := run(ctx, cfg); err != nil {
		log.Error(ctx, "Embedder failed", err)
		os.Exit(1)
	}

	log.Info(ctx, "Embedder completed successfully")
}

func parseFlags() Config {
	cfg := DefaultConfig()

	// Configuration file
	flag.StringVar(&cfg.CLI.ConfigFile, "config", "", "Path to TOML config file")

	// Database flags
	flag.StringVar(&cfg.Database.Path, "db-path", cfg.Database.Path, "Path to Navidrome database")

	// Milvus flags
	flag.StringVar(&cfg.Milvus.URI, "milvus-uri", cfg.Milvus.URI, "Milvus connection URI")

	// Model flags
	flag.StringVar(&cfg.Models.LibraryPath, "library-path", cfg.Models.LibraryPath, "Path to llama.cpp libraries")
	flag.StringVar(&cfg.Models.TextModel, "text-model", cfg.Models.TextModel, "Path to text embedding model")
	flag.StringVar(&cfg.Models.AudioModel, "audio-model", cfg.Models.AudioModel, "Path to audio model")
	flag.StringVar(&cfg.Models.Projector, "projector", cfg.Models.Projector, "Path to audio projector (mmproj)")

	// Embedder flags
	flag.IntVar(&cfg.Embedder.BatchSize, "batch-size", cfg.Embedder.BatchSize, "Batch size for processing")
	flag.IntVar(&cfg.Embedder.GPULayers, "gpu-layers", cfg.Embedder.GPULayers, "GPU layers to offload")
	flag.IntVar(&cfg.Embedder.Threads, "threads", cfg.Embedder.Threads, "CPU threads for inference (0 = auto)")
	flag.BoolVar(&cfg.Embedder.EnableLyrics, "enable-lyrics", cfg.Embedder.EnableLyrics, "Enable lyrics generation")
	flag.BoolVar(&cfg.Embedder.EnableDescription, "enable-description", cfg.Embedder.EnableDescription, "Enable audio description")
	flag.BoolVar(&cfg.Embedder.EnableFlamingo, "enable-flamingo", cfg.Embedder.EnableFlamingo, "Enable audio embedding")

	// Logging flags
	flag.StringVar(&cfg.Logging.Level, "log-level", cfg.Logging.Level, "Log level (debug, info, warn, error)")
	flag.StringVar(&cfg.Logging.File, "log-file", cfg.Logging.File, "Log file path (empty = stderr)")

	// CLI flags
	flag.BoolVar(&cfg.CLI.DryRun, "dry-run", false, "Show what would be embedded without actually doing it")
	flag.BoolVar(&cfg.CLI.Force, "force", false, "Re-embed tracks even if embeddings exist")
	flag.IntVar(&cfg.CLI.Limit, "limit", 0, "Max tracks to process (0 = all)")
	flag.StringVar(&cfg.CLI.Filter, "filter", "", "SQL WHERE clause to filter tracks")

	// Version flag
	showVersion := flag.Bool("version", false, "Show version and exit")

	flag.Parse()

	if *showVersion {
		fmt.Printf("navidrome-embedder version %s\n", version)
		os.Exit(0)
	}

	// Load config file if specified
	if cfg.CLI.ConfigFile != "" {
		if err := LoadConfigFile(cfg.CLI.ConfigFile, &cfg); err != nil {
			fmt.Fprintf(os.Stderr, "Error loading config file: %v\n", err)
			os.Exit(2)
		}
	}

	return cfg
}

func initLogging(cfg LoggingConfig) {
	log.SetLevelString(cfg.Level)

	if cfg.File != "" {
		f, err := os.OpenFile(cfg.File, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Failed to open log file: %v\n", err)
			os.Exit(2)
		}
		log.SetOutput(f)
	}
}

func printConfig(ctx context.Context, cfg Config) {
	log.Info(ctx, "Configuration:")
	log.Info(ctx, "  Database:", "path", cfg.Database.Path, "type", cfg.Database.Type)
	log.Info(ctx, "  Milvus:", "uri", cfg.Milvus.URI)
	log.Info(ctx, "  Models:",
		"library", cfg.Models.LibraryPath,
		"text", cfg.Models.TextModel,
		"audio", cfg.Models.AudioModel,
		"projector", cfg.Models.Projector)
	log.Info(ctx, "  Embedder:",
		"batchSize", cfg.Embedder.BatchSize,
		"gpuLayers", cfg.Embedder.GPULayers,
		"threads", cfg.Embedder.Threads,
		"lyrics", cfg.Embedder.EnableLyrics,
		"description", cfg.Embedder.EnableDescription,
		"flamingo", cfg.Embedder.EnableFlamingo)
	log.Info(ctx, "  CLI:",
		"dryRun", cfg.CLI.DryRun,
		"force", cfg.CLI.Force,
		"limit", cfg.CLI.Limit,
		"filter", cfg.CLI.Filter)
}

func run(ctx context.Context, cfg Config) error {
	// Connect to database
	log.Info(ctx, "Connecting to database", "path", cfg.Database.Path, "type", cfg.Database.Type)
	db, err := OpenDatabase(ctx, cfg.Database)
	if err != nil {
		return fmt.Errorf("failed to open database: %w", err)
	}
	defer db.Close()

	// Get total track count
	totalTracks, err := GetTotalTrackCount(ctx, db, cfg.CLI.Filter)
	if err != nil {
		return fmt.Errorf("failed to get track count: %w", err)
	}

	if totalTracks == 0 {
		log.Info(ctx, "No tracks found in database")
		return nil
	}

	// Apply limit if specified
	tracksToProcess := totalTracks
	if cfg.CLI.Limit > 0 && cfg.CLI.Limit < totalTracks {
		tracksToProcess = cfg.CLI.Limit
	}

	log.Info(ctx, "Starting embedder",
		"totalTracks", totalTracks,
		"toProcess", tracksToProcess,
		"batchSize", cfg.Embedder.BatchSize)

	// In dry-run mode, just list tracks and exit
	if cfg.CLI.DryRun {
		return dryRun(ctx, db, cfg, tracksToProcess)
	}

	// Initialize musicembed client
	log.Info(ctx, "Initializing music embedding client")
	musicClient, err := initMusicClient(cfg)
	if err != nil {
		return fmt.Errorf("failed to initialize music client: %w", err)
	}
	defer musicClient.Close()

	// Initialize Milvus client
	log.Info(ctx, "Connecting to Milvus", "uri", cfg.Milvus.URI)
	milvusClient, err := initMilvusClient(ctx, cfg)
	if err != nil {
		return fmt.Errorf("failed to initialize Milvus client: %w", err)
	}
	defer milvusClient.Close()

	// Load checkpoint (resume from previous run)
	checkpoint, err := LoadCheckpoint()
	if err != nil {
		log.Warn(ctx, "Failed to load checkpoint", err)
		checkpoint = nil
	}

	startOffset := 0
	if checkpoint != nil {
		log.Info(ctx, "Resuming from checkpoint",
			"lastProcessedID", checkpoint.LastProcessedID,
			"totalProcessed", checkpoint.TotalProcessed,
			"timestamp", checkpoint.Timestamp)
		startOffset = checkpoint.TotalProcessed
	}

	// Process tracks
	stats := &EmbeddingStats{
		StartTime: time.Now(),
	}

	err = processAllTracks(ctx, db, musicClient, milvusClient, cfg, startOffset, tracksToProcess, stats)

	// Delete checkpoint on successful completion
	if err == nil && stats.Failed == 0 {
		_ = DeleteCheckpoint()
	}

	// Print final statistics
	printFinalStats(ctx, stats, tracksToProcess)

	if stats.Failed > 0 {
		return fmt.Errorf("completed with %d failures", stats.Failed)
	}

	return err
}

func dryRun(ctx context.Context, db *sql.DB, cfg Config, limit int) error {
	log.Info(ctx, "Dry run mode - listing tracks to be embedded")

	offset := 0
	batchSize := 100
	count := 0

	for offset < limit {
		tracks, err := FetchTrackBatch(ctx, db, offset, batchSize, cfg.CLI.Filter)
		if err != nil {
			return err
		}

		if len(tracks) == 0 {
			break
		}

		for _, track := range tracks {
			if count >= limit {
				break
			}
			log.Info(ctx, "Would embed",
				"track", track.Path,
				"artist", track.Artist,
				"title", track.Title,
				"album", track.Album)
			count++
		}

		offset += len(tracks)
	}

	log.Info(ctx, "Dry run complete", "tracks", count)
	return nil
}

func processAllTracks(
	ctx context.Context,
	db *sql.DB,
	musicClient *musicembed.Client,
	milvusClient *milvus.Client,
	cfg Config,
	startOffset, limit int,
	stats *EmbeddingStats,
) error {
	offset := startOffset
	lastProgressTime := time.Now()

	for offset < limit {
		// Check for cancellation
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		// Fetch batch of tracks
		tracks, err := FetchTrackBatch(ctx, db, offset, cfg.Embedder.BatchSize, cfg.CLI.Filter)
		if err != nil {
			return fmt.Errorf("failed to fetch tracks: %w", err)
		}

		if len(tracks) == 0 {
			break
		}

		// Process each track in the batch
		for _, track := range tracks {
			if offset+stats.Processed >= limit {
				break
			}

			result, err := ProcessTrackWithRetry(ctx, track, musicClient, milvusClient, cfg, cfg.CLI.Force)
			if err != nil {
				// Check if it's a "skip" error (already exists)
				if !cfg.CLI.Force && err.Error() == "embeddings already exist (use --force to re-embed)" {
					stats.Skipped++
					log.Debug(ctx, "Skipped track (already embedded)",
						"track", track.Path)
				} else {
					stats.Failed++
					stats.AddError(fmt.Sprintf("%s: %v", track.Path, err))
					log.Error(ctx, "Failed to embed track", err,
						"track", track.Path,
						"attempt", "final")
				}
			} else {
				stats.Processed++
				log.Info(ctx, "Embedded track",
					"track", track.Path,
					"artist", track.Artist,
					"title", track.Title,
					"hasLyrics", result.LyricsEmbedding != nil,
					"hasDescription", result.DescriptionEmbedding != nil,
					"hasAudio", result.FlamingoEmbedding != nil,
					"duration", result.Duration.Round(time.Millisecond))
			}

			// Save checkpoint periodically
			if (stats.Processed+stats.Skipped)%checkpointInterval == 0 {
				checkpoint := CheckpointState{
					LastProcessedID: track.ID,
					Timestamp:       time.Now(),
					TotalProcessed:  offset + stats.Processed + stats.Skipped,
				}
				if err := SaveCheckpoint(checkpoint); err != nil {
					log.Warn(ctx, "Failed to save checkpoint", err)
				}
			}

			// Print progress periodically
			if time.Since(lastProgressTime) >= progressInterval {
				printProgress(ctx, stats, offset, limit)
				lastProgressTime = time.Now()
			}
		}

		offset += len(tracks)
	}

	return nil
}

func printProgress(ctx context.Context, stats *EmbeddingStats, offset, total int) {
	percent := float64(stats.Processed+stats.Skipped) / float64(total) * 100
	rate := stats.Rate()
	eta := stats.ETA(total)

	log.Info(ctx, "Progress",
		"processed", stats.Processed,
		"skipped", stats.Skipped,
		"failed", stats.Failed,
		"total", total,
		"percent", fmt.Sprintf("%.1f%%", percent),
		"rate", fmt.Sprintf("%.2f tracks/sec", rate),
		"eta", eta.Round(time.Second))
}

func printFinalStats(ctx context.Context, stats *EmbeddingStats, total int) {
	elapsed := time.Since(stats.StartTime)

	log.Info(ctx, "Embedding complete",
		"processed", stats.Processed,
		"skipped", stats.Skipped,
		"failed", stats.Failed,
		"total", total,
		"elapsed", elapsed.Round(time.Second),
		"avgRate", fmt.Sprintf("%.2f tracks/sec", stats.Rate()))

	if stats.Failed > 0 && len(stats.Errors) > 0 {
		log.Warn(ctx, "Recent errors:", "count", len(stats.Errors))
		for i, errMsg := range stats.Errors {
			if i >= 10 {
				log.Warn(ctx, fmt.Sprintf("  ... and %d more errors", len(stats.Errors)-10))
				break
			}
			log.Warn(ctx, fmt.Sprintf("  %s", errMsg))
		}
	}
}

func initMusicClient(cfg Config) (*musicembed.Client, error) {
	musicCfg := musicembed.Config{
		LibraryPath:        cfg.Models.LibraryPath,
		EmbeddingModelFile: cfg.Models.TextModel,
		MmprojFile:         cfg.Models.Projector,
		ModelFile:          cfg.Models.AudioModel,
		GPULayers:          cfg.Embedder.GPULayers,
		UseGPU:             cfg.Embedder.GPULayers > 0,
		Threads:            cfg.Embedder.Threads,
		ContextSize:        8192,
		BatchSize:          2048,
		GenerationMargin:   512,
		MaxOutputTokens:    8192,
		DescriptionPrompt:  "Describe this track in full detail - tell me the genre, tempo, and key, then dive into the instruments, production style, and overall mood it creates.",
		LyricsPrompt:       "Please provide the structured lyric sheet for this song.",
	}

	client, err := musicembed.New(musicCfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create music client: %w", err)
	}

	return client, nil
}

func initMilvusClient(ctx context.Context, cfg Config) (*milvus.Client, error) {
	milvusCfg := milvus.Config{
		URI:        cfg.Milvus.URI,
		Timeout:    cfg.Milvus.Timeout,
		MaxRetries: 3,
		Dimensions: milvus.Dimensions{
			Lyrics:      cfg.Milvus.Dimensions.Lyrics,
			Description: cfg.Milvus.Dimensions.Description,
			Flamingo:    cfg.Milvus.Dimensions.Flamingo,
		},
	}

	client, err := milvus.NewClient(ctx, milvusCfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create Milvus client: %w", err)
	}

	// Ensure collections exist
	if err := client.EnsureCollections(ctx); err != nil {
		client.Close()
		return nil, fmt.Errorf("failed to ensure Milvus collections: %w", err)
	}

	return client, nil
}
