package scanner

import "context"

// embeddingStatus represents the status of embeddings for a track.
type embeddingStatus struct {
	Embedded       bool   `json:"embedded"`
	HasDescription bool   `json:"hasDescription"`
	Name           string `json:"name"`
}

// embeddingClient is the interface for embedding services.
type embeddingClient interface {
	HealthCheck(ctx context.Context) error
	CheckEmbedding(ctx context.Context, candidate embeddingCandidate) (embeddingStatus, error)
	EmbedSong(ctx context.Context, candidate embeddingCandidate) error
	FlushBatch(ctx context.Context) error
}
