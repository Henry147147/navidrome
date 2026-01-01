// Package embedder provides the three-stage embedding pipeline for audio files.
package embedder

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/navidrome/navidrome/log"
	"github.com/navidrome/navidrome/recommender/llamacpp"
	"github.com/navidrome/navidrome/recommender/milvus"
)

// Config holds embedder configuration.
type Config struct {
	BatchTimeout      time.Duration
	BatchSize         int
	EnableLyrics      bool // Enable lyrics text embedding
	EnableDescription bool // Enable audio description embedding
	EnableFlamingo    bool // Enable flamingo audio embedding
}

// EmbedRequest contains all information needed to embed a track.
type EmbedRequest struct {
	FilePath  string
	TrackName string
	TrackID   string
	Artist    string
	Title     string
	Album     string
}

// EmbedResult contains the results of embedding a track.
type EmbedResult struct {
	TrackName            string
	LyricsEmbedding      []float64
	DescriptionEmbedding []float64
	FlamingoEmbedding    []float64
	Description          string
}

// StatusRequest contains information for checking embedding status.
type StatusRequest struct {
	TrackID        string
	Artist         string
	Title          string
	Album          string
	AlternateNames []string
}

// StatusResult contains embedding status information.
type StatusResult struct {
	Embedded          bool
	HasDescription    bool
	HasAudioEmbedding bool
	CanonicalName     string
}

// Embedder implements the embedding pipeline using llama.cpp and Milvus.
type Embedder struct {
	config      Config
	llmClient   *llamacpp.Client
	vectorStore *milvus.Client
	pipeline    *Pipeline
	mu          sync.RWMutex
	closed      bool
}

// New creates a new Embedder with the given configuration and clients.
func New(cfg Config, llm *llamacpp.Client, milvus *milvus.Client) *Embedder {
	e := &Embedder{
		config:      cfg,
		llmClient:   llm,
		vectorStore: milvus,
	}

	e.pipeline = NewPipeline(PipelineConfig{
		BatchTimeout:      cfg.BatchTimeout,
		BatchSize:         cfg.BatchSize,
		EnableLyrics:      cfg.EnableLyrics,
		EnableDescription: cfg.EnableDescription,
		EnableFlamingo:    cfg.EnableFlamingo,
	}, llm, milvus)

	return e
}

// EmbedAudio processes an audio file through all enabled stages.
func (e *Embedder) EmbedAudio(ctx context.Context, req EmbedRequest) (*EmbedResult, error) {
	e.mu.RLock()
	if e.closed {
		e.mu.RUnlock()
		return nil, fmt.Errorf("embedder is closed")
	}
	e.mu.RUnlock()

	if req.FilePath == "" {
		return nil, fmt.Errorf("file path is required")
	}

	trackName := req.TrackName
	if trackName == "" {
		trackName = canonicalName(req.Artist, req.Title)
	}

	log.Debug(ctx, "Queueing audio for embedding",
		"file", req.FilePath,
		"name", trackName,
		"trackId", req.TrackID,
	)

	// Create track context and enqueue
	trackCtx := &TrackContext{
		FilePath:  req.FilePath,
		TrackName: trackName,
		TrackID:   req.TrackID,
		Artist:    req.Artist,
		Title:     req.Title,
		Album:     req.Album,
		Done:      make(chan struct{}),
	}

	if err := e.pipeline.Enqueue(ctx, trackCtx); err != nil {
		return nil, fmt.Errorf("enqueue track: %w", err)
	}

	// Wait for processing to complete
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-trackCtx.Done:
	}

	if trackCtx.Error != nil {
		return nil, trackCtx.Error
	}

	return &EmbedResult{
		TrackName:            trackCtx.TrackName,
		LyricsEmbedding:      trackCtx.LyricsEmbedding,
		DescriptionEmbedding: trackCtx.DescriptionEmbedding,
		FlamingoEmbedding:    trackCtx.FlamingoEmbedding,
		Description:          trackCtx.Description,
	}, nil
}

// EmbedText generates text embeddings directly (for query-time use).
func (e *Embedder) EmbedText(ctx context.Context, text string) ([]float64, error) {
	e.mu.RLock()
	if e.closed {
		e.mu.RUnlock()
		return nil, fmt.Errorf("embedder is closed")
	}
	e.mu.RUnlock()

	if text == "" {
		return nil, fmt.Errorf("text is required")
	}

	// For query-time text embedding, we bypass the batch pipeline
	// and call llama.cpp directly for lower latency
	resp, err := e.llmClient.EmbedText(ctx, llamacpp.TextEmbedRequest{
		Text: text,
	})
	if err != nil {
		return nil, fmt.Errorf("embed text: %w", err)
	}

	return resp.Embedding, nil
}

// CheckStatus checks if embeddings already exist for a track.
func (e *Embedder) CheckStatus(ctx context.Context, req StatusRequest) (*StatusResult, error) {
	e.mu.RLock()
	if e.closed {
		e.mu.RUnlock()
		return nil, fmt.Errorf("embedder is closed")
	}
	e.mu.RUnlock()

	// Build list of possible names to check
	names := buildPossibleNames(req)
	if len(names) == 0 {
		return &StatusResult{}, nil
	}

	// Check existence in each collection
	lyricsExists, err := e.vectorStore.Exists(ctx, milvus.CollectionLyrics, names)
	if err != nil {
		return nil, fmt.Errorf("check lyrics embeddings: %w", err)
	}

	descExists, err := e.vectorStore.Exists(ctx, milvus.CollectionDescription, names)
	if err != nil {
		return nil, fmt.Errorf("check description embeddings: %w", err)
	}

	flamingoExists, err := e.vectorStore.Exists(ctx, milvus.CollectionFlamingo, names)
	if err != nil {
		return nil, fmt.Errorf("check flamingo embeddings: %w", err)
	}

	// Find the canonical name (first match found)
	var canonicalNameResult string
	var hasLyrics, hasDesc, hasFlamingo bool
	for _, name := range names {
		if lyricsExists[name] {
			canonicalNameResult = name
			hasLyrics = true
			break
		}
	}
	for _, name := range names {
		if descExists[name] {
			if canonicalNameResult == "" {
				canonicalNameResult = name
			}
			hasDesc = true
			break
		}
	}
	for _, name := range names {
		if flamingoExists[name] {
			if canonicalNameResult == "" {
				canonicalNameResult = name
			}
			hasFlamingo = true
			break
		}
	}

	return &StatusResult{
		Embedded:          hasLyrics || hasDesc || hasFlamingo,
		HasAudioEmbedding: hasFlamingo,
		HasDescription:    hasDesc,
		CanonicalName:     canonicalNameResult,
	}, nil
}

// FlushBatch forces processing of any pending batch items.
func (e *Embedder) FlushBatch(ctx context.Context) error {
	e.mu.RLock()
	if e.closed {
		e.mu.RUnlock()
		return fmt.Errorf("embedder is closed")
	}
	e.mu.RUnlock()

	return e.pipeline.Flush(ctx)
}

// Close releases resources.
func (e *Embedder) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.closed {
		return nil
	}

	e.closed = true
	e.pipeline.Close()
	return nil
}


// buildPossibleNames generates all possible track names from a status request.
func buildPossibleNames(req StatusRequest) []string {
	names := make([]string, 0, 3+len(req.AlternateNames))

	// Canonical name from artist + title
	if canonical := canonicalName(req.Artist, req.Title); canonical != "" {
		names = append(names, canonical)
	}

	// Track ID as name
	if req.TrackID != "" {
		names = append(names, req.TrackID)
	}

	// Alternate names (e.g., filename)
	names = append(names, req.AlternateNames...)

	return names
}

// canonicalName generates a canonical name from artist and title.
func canonicalName(artist, title string) string {
	if artist == "" && title == "" {
		return ""
	}
	if artist == "" {
		return title
	}
	if title == "" {
		return artist
	}
	return artist + " - " + title
}
