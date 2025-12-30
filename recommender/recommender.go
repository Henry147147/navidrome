// Package recommender provides music recommendation and embedding services.
// It replaces the Python-based embedding services with a pure Go implementation
// using llama.cpp for model inference and Milvus for vector storage.
package recommender

import (
	"context"
	"time"
)

// Embedder generates embeddings for audio files using a three-stage pipeline:
// 1. Audio -> Audio embedding
// 2. Audio -> Text description
// 3. Text description -> Text embedding
type Embedder interface {
	// EmbedAudio processes an audio file through all enabled stages.
	EmbedAudio(ctx context.Context, req EmbedRequest) (*EmbedResult, error)

	// EmbedText generates text embeddings directly (for query-time use).
	EmbedText(ctx context.Context, text string) ([]float64, error)

	// CheckStatus checks if embeddings already exist for a track.
	CheckStatus(ctx context.Context, req StatusRequest) (*StatusResult, error)

	// FlushBatch forces processing of any pending batch items.
	FlushBatch(ctx context.Context) error

	// Close releases resources.
	Close() error
}

// RecommendationEngine provides playlist recommendations based on embeddings.
type RecommendationEngine interface {
	// Recommend generates track recommendations based on seeds.
	Recommend(ctx context.Context, req RecommendationRequest) (*RecommendationResponse, error)
}

// VectorStore abstracts Milvus vector database operations.
type VectorStore interface {
	// EnsureCollections creates collections if they don't exist.
	EnsureCollections(ctx context.Context) error

	// Upsert inserts or updates embedding data.
	Upsert(ctx context.Context, collection string, data []EmbeddingData) error

	// Search performs similarity search.
	Search(ctx context.Context, collection string, vector []float64, opts SearchOptions) ([]SearchResult, error)

	// GetByNames retrieves embeddings by track names.
	GetByNames(ctx context.Context, collection string, names []string) (map[string][]float64, error)

	// Exists checks if a name exists in a collection.
	Exists(ctx context.Context, collection string, names []string) (map[string]bool, error)

	// Delete removes embeddings by names.
	Delete(ctx context.Context, collection string, names []string) error

	// Close releases the connection.
	Close() error
}

// LLMClient handles communication with the llama.cpp inference server.
type LLMClient interface {
	// EmbedAudioBatch generates audio embeddings for multiple files.
	EmbedAudioBatch(ctx context.Context, reqs []AudioEmbedRequest) ([]AudioEmbedResponse, error)

	// DescribeAudioBatch generates text descriptions for multiple audio files.
	DescribeAudioBatch(ctx context.Context, reqs []AudioDescribeRequest) ([]AudioDescribeResponse, error)

	// EmbedTextBatch generates text embeddings for multiple strings.
	EmbedTextBatch(ctx context.Context, reqs []TextEmbedRequest) ([]TextEmbedResponse, error)

	// HealthCheck verifies the llama.cpp server is reachable.
	HealthCheck(ctx context.Context) error
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
	TrackName       string
	AudioEmbedding  []float64
	TextEmbedding   []float64
	FlamingoEmbedding []float64
	Description     string
	ModelID         string
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

// EmbeddingData represents a single embedding record for storage.
type EmbeddingData struct {
	Name        string
	Embedding   []float64
	Offset      float64
	ModelID     string
	Description string // Only for description_embedding collection
}

// SearchOptions configures similarity search behavior.
type SearchOptions struct {
	TopK         int
	ExcludeNames []string
}

// SearchResult represents a single search hit.
type SearchResult struct {
	Name     string
	Distance float64
	TrackID  string
}

// AudioEmbedRequest for llama.cpp audio embedding endpoint.
type AudioEmbedRequest struct {
	AudioPath  string `json:"audio_path"`
	SampleRate int    `json:"sample_rate,omitempty"`
	BatchID    string `json:"batch_id,omitempty"`
}

// AudioEmbedResponse from llama.cpp audio embedding endpoint.
type AudioEmbedResponse struct {
	Embedding []float64 `json:"embedding"`
	ModelID   string    `json:"model_id"`
	Duration  float64   `json:"duration_seconds"`
	Error     string    `json:"error,omitempty"`
}

// AudioDescribeRequest for llama.cpp audio description endpoint.
type AudioDescribeRequest struct {
	AudioPath string `json:"audio_path"`
	Prompt    string `json:"prompt,omitempty"`
}

// AudioDescribeResponse from llama.cpp audio description endpoint.
type AudioDescribeResponse struct {
	Description    string    `json:"description"`
	AudioEmbedding []float64 `json:"audio_embedding"` // Flamingo audio features
	ModelID        string    `json:"model_id"`
	Error          string    `json:"error,omitempty"`
}

// TextEmbedRequest for llama.cpp text embedding endpoint.
type TextEmbedRequest struct {
	Text    string `json:"text"`
	ModelID string `json:"model_id,omitempty"`
}

// TextEmbedResponse from llama.cpp text embedding endpoint.
type TextEmbedResponse struct {
	Embedding []float64 `json:"embedding"`
	ModelID   string    `json:"model_id"`
	Dimension int       `json:"dimension"`
	Error     string    `json:"error,omitempty"`
}

// RecommendationRequest contains all parameters for generating recommendations.
type RecommendationRequest struct {
	UserID            string
	UserName          string
	Limit             int
	Mode              string
	Seeds             []RecommendationSeed
	Diversity         float64
	ExcludeTrackIDs   []string
	LibraryIDs        []int
	DislikedTrackIDs  []string
	DislikedArtistIDs []string
	DislikeStrength   float64

	// Multi-model support
	Models            []string
	MergeStrategy     string // "union", "intersection", "priority"
	ModelPriorities   map[string]int
	MinModelAgreement int

	// Negative prompting
	NegativePrompts       []string
	NegativePromptPenalty float64
	NegativeEmbeddings    map[string][][]float64
}

// RecommendationSeed represents a seed track for recommendations.
type RecommendationSeed struct {
	TrackID   string
	Weight    float64
	Source    string
	PlayedAt  *time.Time
	Embedding []float64 // Direct embedding for text queries
}

// RecommendationItem represents a single recommended track.
type RecommendationItem struct {
	TrackID            string
	Score              float64
	Reason             string
	Models             []string // Models that contributed
	NegativeSimilarity *float64 // Similarity to negative prompts
}

// RecommendationResponse contains recommendation results.
type RecommendationResponse struct {
	Tracks   []RecommendationItem
	Warnings []string
}

// TrackIDs returns the track IDs from the response.
func (r RecommendationResponse) TrackIDs() []string {
	ids := make([]string, 0, len(r.Tracks))
	for _, item := range r.Tracks {
		if item.TrackID != "" {
			ids = append(ids, item.TrackID)
		}
	}
	return ids
}

// CanonicalName generates a canonical name from artist and title.
func CanonicalName(artist, title string) string {
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
