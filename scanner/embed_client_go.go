package scanner

import (
	"context"
	"path/filepath"

	"github.com/navidrome/navidrome/log"
	"github.com/navidrome/navidrome/recommender/embedder"
	"github.com/navidrome/navidrome/recommender/resolver"
)

// WithGoEmbedder returns a ScannerOption that configures the scanner to use
// the Go-based embedder instead of the Python socket client.
func WithGoEmbedder(e *embedder.Embedder) ScannerOption {
	return func(c *controller) {
		if e != nil {
			client := newGoEmbeddingClient(e)
			c.embedWorker = newEmbeddingWorker(client)
			log.Info(c.rootCtx, "Scanner configured with Go embedder")
		}
	}
}

// goEmbeddingClient implements embeddingClient using the Go embedder package.
type goEmbeddingClient struct {
	embedder *embedder.Embedder
}

// newGoEmbeddingClient creates a new Go-based embedding client.
func newGoEmbeddingClient(e *embedder.Embedder) embeddingClient {
	return &goEmbeddingClient{
		embedder: e,
	}
}

// HealthCheck verifies the embedder is ready.
func (c *goEmbeddingClient) HealthCheck(ctx context.Context) error {
	// The Go embedder is always "healthy" if it exists.
	// Actual health depends on llama.cpp server availability which is checked on demand.
	return nil
}

// CheckEmbedding checks if embeddings exist for a track.
func (c *goEmbeddingClient) CheckEmbedding(ctx context.Context, candidate embeddingCandidate) (embeddingStatus, error) {
	req := embedder.StatusRequest{
		TrackID:        candidate.key(),
		Artist:         candidate.Artist,
		Title:          candidate.Title,
		Album:          candidate.Album,
		AlternateNames: []string{filepath.Base(candidate.TrackPath)},
	}

	result, err := c.embedder.CheckStatus(ctx, req)
	if err != nil {
		return embeddingStatus{}, err
	}

	return embeddingStatus{
		Embedded:       result.Embedded,
		HasDescription: result.HasDescription,
		Name:           result.CanonicalName,
	}, nil
}

// EmbedSong embeds an audio file.
func (c *goEmbeddingClient) EmbedSong(ctx context.Context, candidate embeddingCandidate) error {
	trackName := resolver.CanonicalName(candidate.Artist, candidate.Title)
	if trackName == "" {
		trackName = filepath.Base(candidate.TrackPath)
	}

	req := embedder.EmbedRequest{
		FilePath:  candidate.absolutePath(),
		TrackName: trackName,
		TrackID:   candidate.key(),
		Artist:    candidate.Artist,
		Title:     candidate.Title,
		Album:     candidate.Album,
	}

	_, err := c.embedder.EmbedAudio(ctx, req)
	return err
}

// FlushBatch forces immediate processing of pending embeddings.
func (c *goEmbeddingClient) FlushBatch(ctx context.Context) error {
	return c.embedder.FlushBatch(ctx)
}
