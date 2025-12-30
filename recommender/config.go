package recommender

import "time"

// MilvusConfig holds Milvus connection settings.
// Note: This is re-exported from milvus subpackage for external use.
type MilvusConfig struct {
	URI        string        // Server URI or file path for Milvus Lite
	Timeout    time.Duration // Connection/operation timeout
	MaxRetries int           // Max retry attempts
}

// LLMConfig holds llama.cpp server configuration.
// Note: This is re-exported from llamacpp subpackage for external use.
type LLMConfig struct {
	AudioEmbedURL    string        // Audio embedding endpoint
	AudioDescribeURL string        // Audio description endpoint
	TextEmbedURL     string        // Text embedding endpoint
	Timeout          time.Duration // Request timeout
	MaxRetries       int           // Max retry attempts
	RetryBackoff     time.Duration // Initial backoff between retries
}

// EmbedderConfig holds embedder pipeline configuration.
// Note: This is re-exported from embedder subpackage for external use.
type EmbedderConfig struct {
	BatchTimeout       time.Duration // Time to wait before processing partial batch
	BatchSize          int           // Maximum items per batch
	EnableDescriptions bool          // Enable stage 2 (audio -> description)
	EnableTextEmbed    bool          // Enable stage 3 (description -> text embedding)
}

// EngineConfig holds recommendation engine configuration.
// Note: This is re-exported from engine subpackage for external use.
type EngineConfig struct {
	DefaultTopK      int
	DefaultModels    []string
	DefaultMerge     string
	DefaultDiversity float64
}
