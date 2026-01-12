package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"test/llama/musicembed"

	"github.com/navidrome/navidrome/log"
	"github.com/navidrome/navidrome/recommender/milvus"
)

// EmbeddingResult holds the results of embedding a single track.
type EmbeddingResult struct {
	TrackID              string
	TrackName            string
	LyricsEmbedding      []float64
	DescriptionEmbedding []float64
	FlamingoEmbedding    []float64
	Description          string
	GeneratedLyrics      string
	Duration             time.Duration
}

// CheckpointState stores progress for resuming.
type CheckpointState struct {
	LastProcessedID string    `json:"lastProcessedID"`
	Timestamp       time.Time `json:"timestamp"`
	TotalProcessed  int       `json:"totalProcessed"`
}

// EmbeddingStats tracks embedding statistics.
type EmbeddingStats struct {
	Processed int
	Skipped   int
	Failed    int
	StartTime time.Time
	Errors    []string // Keep last 100 errors
}

const (
	checkpointFile     = ".embedder-state.json"
	maxRetries         = 3
	initialBackoff     = 2 * time.Second
	maxBackoff         = 30 * time.Second
	progressInterval   = 30 * time.Second
	checkpointInterval = 100 // Save checkpoint every N tracks
	maxErrorsToKeep    = 100
)

// ProcessTrack processes a single track and generates embeddings.
func ProcessTrack(
	ctx context.Context,
	track Track,
	musicClient *musicembed.Client,
	milvusClient *milvus.Client,
	cfg Config,
	force bool,
) (*EmbeddingResult, error) {
	startTime := time.Now()
	trackName := CanonicalName(track)

	// Construct full path to audio file
	fullPath := filepath.Join(cfg.CLI.MusicDir, track.Path)

	// Check if embeddings already exist (unless --force)
	if !force {
		exists, err := checkEmbeddingsExist(ctx, milvusClient, trackName, cfg.Embedder)
		if err != nil {
			return nil, fmt.Errorf("failed to check existing embeddings: %w", err)
		}
		if exists {
			return nil, fmt.Errorf("embeddings already exist (use --force to re-embed)")
		}
	}

	result := &EmbeddingResult{
		TrackID:   track.ID,
		TrackName: trackName,
	}

	// Stage 1: Generate lyrics text embedding (if enabled)
	if cfg.Embedder.EnableLyrics {
		var lyricsText string

		log.Debug(ctx, "Processing lyrics",
			"track", track.Path,
			"hasExistingLyrics", track.Lyrics != "",
			"lyricsLength", len(track.Lyrics))

		// Use existing lyrics if available and meaningful (not just whitespace/short placeholder)
		trimmedLyrics := strings.TrimSpace(track.Lyrics)
		if len(trimmedLyrics) > 10 {
			log.Debug(ctx, "Using existing lyrics from database", "track", track.Path)
			lyricsText = track.Lyrics
		} else {
			// Note: GenerateLyricsWithCheck has CUDA cleanup issues
			// Using GenerateLyrics for now until CUDA issue is resolved
			generated, err := musicClient.GenerateLyrics(fullPath)
			if err != nil {
				return nil, fmt.Errorf("failed to generate lyrics: %w", err)
			}
			lyricsText = generated
			result.GeneratedLyrics = generated
		}

		if lyricsText != "" {
			embedding, err := musicClient.EmbedText(lyricsText)
			if err != nil {
				return nil, fmt.Errorf("failed to embed lyrics text: %w", err)
			}
			result.LyricsEmbedding = float32sToFloat64s(embedding)

			// Store lyrics embedding in Milvus
			err = storeLyricsEmbedding(ctx, milvusClient, trackName, result.LyricsEmbedding, lyricsText)
			if err != nil {
				return nil, fmt.Errorf("failed to store lyrics embedding: %w", err)
			}
		}
	}

	// Stage 2: Generate audio description and embed (if enabled)
	if cfg.Embedder.EnableDescription {
		description, err := musicClient.GenerateDescription(fullPath)
		if err != nil {
			return nil, fmt.Errorf("failed to generate description: %w", err)
		}
		result.Description = description

		if description != "" {
			embedding, err := musicClient.EmbedText(description)
			if err != nil {
				return nil, fmt.Errorf("failed to embed description text: %w", err)
			}
			result.DescriptionEmbedding = float32sToFloat64s(embedding)

			// Store description embedding in Milvus
			err = storeDescriptionEmbedding(ctx, milvusClient, trackName, result.DescriptionEmbedding, description)
			if err != nil {
				return nil, fmt.Errorf("failed to store description embedding: %w", err)
			}
		}
	}

	// Stage 3: Generate direct audio embedding (if enabled)
	if cfg.Embedder.EnableFlamingo {
		embedding, err := musicClient.EmbedAudio(fullPath)
		if err != nil {
			return nil, fmt.Errorf("failed to embed audio: %w", err)
		}
		result.FlamingoEmbedding = float32sToFloat64s(embedding)

		// Store audio embedding in Milvus
		err = storeAudioEmbedding(ctx, milvusClient, trackName, result.FlamingoEmbedding)
		if err != nil {
			return nil, fmt.Errorf("failed to store audio embedding: %w", err)
		}
	}

	result.Duration = time.Since(startTime)
	return result, nil
}

// ProcessTrackWithRetry processes a track with retry logic.
func ProcessTrackWithRetry(
	ctx context.Context,
	track Track,
	musicClient *musicembed.Client,
	milvusClient *milvus.Client,
	cfg Config,
	force bool,
) (*EmbeddingResult, error) {
	backoff := initialBackoff

	for attempt := 0; attempt <= maxRetries; attempt++ {
		if attempt > 0 {
			// Wait before retry
			select {
			case <-time.After(backoff):
			case <-ctx.Done():
				return nil, ctx.Err()
			}

			// Exponential backoff
			backoff *= 2
			if backoff > maxBackoff {
				backoff = maxBackoff
			}

			log.Debug(ctx, "Retrying track embedding",
				"track", track.Path,
				"attempt", attempt+1,
				"maxRetries", maxRetries)
		}

		result, err := ProcessTrack(ctx, track, musicClient, milvusClient, cfg, force)
		if err == nil {
			return result, nil
		}

		// Check if error is "already exists"
		if !force && err.Error() == "embeddings already exist (use --force to re-embed)" {
			return nil, err // Don't retry for "already exists"
		}

		if attempt == maxRetries {
			return nil, fmt.Errorf("failed after %d retries: %w", maxRetries, err)
		}

		log.Warn(ctx, "Track embedding failed, will retry",
			"track", track.Path,
			"attempt", attempt+1,
			"error", err.Error())
	}

	return nil, fmt.Errorf("failed to process track")
}

// checkEmbeddingsExist checks if embeddings already exist for a track.
func checkEmbeddingsExist(ctx context.Context, milvusClient *milvus.Client, trackName string, cfg EmbedderConfig) (bool, error) {
	names := []string{trackName}

	// Check each enabled collection
	if cfg.EnableLyrics {
		exists, err := milvusClient.Exists(ctx, milvus.CollectionLyrics, names)
		if err != nil {
			return false, err
		}
		if exists[trackName] {
			return true, nil
		}
	}

	if cfg.EnableDescription {
		exists, err := milvusClient.Exists(ctx, milvus.CollectionDescription, names)
		if err != nil {
			return false, err
		}
		if exists[trackName] {
			return true, nil
		}
	}

	if cfg.EnableFlamingo {
		exists, err := milvusClient.Exists(ctx, milvus.CollectionFlamingo, names)
		if err != nil {
			return false, err
		}
		if exists[trackName] {
			return true, nil
		}
	}

	return false, nil
}

// storeLyricsEmbedding stores a lyrics embedding in Milvus.
func storeLyricsEmbedding(ctx context.Context, client *milvus.Client, name string, embedding []float64, lyrics string) error {
	data := []milvus.EmbeddingData{
		{
			Name:      name,
			Embedding: embedding,
			Lyrics:    lyrics,
		},
	}
	return client.Upsert(ctx, milvus.CollectionLyrics, data)
}

// storeDescriptionEmbedding stores a description embedding in Milvus.
func storeDescriptionEmbedding(ctx context.Context, client *milvus.Client, name string, embedding []float64, description string) error {
	data := []milvus.EmbeddingData{
		{
			Name:        name,
			Embedding:   embedding,
			Description: description,
		},
	}
	return client.Upsert(ctx, milvus.CollectionDescription, data)
}

// storeAudioEmbedding stores an audio embedding in Milvus.
func storeAudioEmbedding(ctx context.Context, client *milvus.Client, name string, embedding []float64) error {
	data := []milvus.EmbeddingData{
		{
			Name:      name,
			Embedding: embedding,
		},
	}
	return client.Upsert(ctx, milvus.CollectionFlamingo, data)
}

// LoadCheckpoint loads the checkpoint state from disk.
func LoadCheckpoint() (*CheckpointState, error) {
	data, err := os.ReadFile(checkpointFile)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil // No checkpoint exists
		}
		return nil, fmt.Errorf("failed to read checkpoint: %w", err)
	}

	var state CheckpointState
	if err := json.Unmarshal(data, &state); err != nil {
		return nil, fmt.Errorf("failed to parse checkpoint: %w", err)
	}

	return &state, nil
}

// SaveCheckpoint saves the checkpoint state to disk.
func SaveCheckpoint(state CheckpointState) error {
	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal checkpoint: %w", err)
	}

	if err := os.WriteFile(checkpointFile, data, 0600); err != nil {
		return fmt.Errorf("failed to write checkpoint: %w", err)
	}

	return nil
}

// DeleteCheckpoint removes the checkpoint file.
func DeleteCheckpoint() error {
	if err := os.Remove(checkpointFile); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("failed to delete checkpoint: %w", err)
	}
	return nil
}

// float32sToFloat64s converts []float32 to []float64.
func float32sToFloat64s(f32s []float32) []float64 {
	f64s := make([]float64, len(f32s))
	for i, v := range f32s {
		f64s[i] = float64(v)
	}
	return f64s
}

// AddError adds an error to the stats, keeping only the most recent maxErrorsToKeep errors.
func (s *EmbeddingStats) AddError(err string) {
	s.Errors = append(s.Errors, err)
	if len(s.Errors) > maxErrorsToKeep {
		s.Errors = s.Errors[len(s.Errors)-maxErrorsToKeep:]
	}
}

// Rate returns the processing rate in tracks per second.
func (s *EmbeddingStats) Rate() float64 {
	elapsed := time.Since(s.StartTime).Seconds()
	if elapsed == 0 {
		return 0
	}
	return float64(s.Processed) / elapsed
}

// ETA returns the estimated time to completion.
func (s *EmbeddingStats) ETA(total int) time.Duration {
	rate := s.Rate()
	if rate == 0 {
		return 0
	}
	remaining := total - s.Processed - s.Skipped
	if remaining <= 0 {
		return 0
	}
	return time.Duration(float64(remaining)/rate) * time.Second
}
