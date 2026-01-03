package recommender

import "time"

// MilvusConfig holds Milvus connection settings.
// Note: This is re-exported from milvus subpackage for external use.
type MilvusConfig struct {
	URI        string        // Server URI or file path for Milvus Lite
	Timeout    time.Duration // Connection/operation timeout
	MaxRetries int           // Max retry attempts
}

// LLMConfig holds llama.cpp configuration.
// Note: This is re-exported from llamacpp subpackage for external use.
type LLMConfig struct {
	LibraryPath        string        // Path to llama.cpp shared libraries
	TextModelPath      string        // Path to text embedding model
	AudioModelPath     string        // Path to audio model
	AudioProjectorPath string        // Path to audio projector (mmproj)
	ContextSize        uint32        // Context size override (0 = default)
	BatchSize          uint32        // Batch size override (0 = default)
	UBatchSize         uint32        // Micro batch size override (0 = default)
	Threads            int           // Threads for inference (0 = default)
	ThreadsBatch       int           // Threads for batch processing (0 = default)
	GPULayers          int           // Layers to offload to GPU (0 = default)
	MainGPU            int           // Main GPU index (0 = default)
	Timeout            time.Duration // Request timeout
	MaxRetries         int           // Max retry attempts
	RetryBackoff       time.Duration // Initial backoff between retries
}

// EmbedderConfig holds embedder pipeline configuration.
// Note: This is re-exported from embedder subpackage for external use.
type EmbedderConfig struct {
	BatchTimeout      time.Duration // Time to wait before processing partial batch
	BatchSize         int           // Maximum items per batch
	EnableLyrics      bool          // Enable lyrics text embedding
	EnableDescription bool          // Enable audio description embedding
	EnableFlamingo    bool          // Enable flamingo audio embedding
}

// EngineConfig holds recommendation engine configuration.
// Note: This is re-exported from engine subpackage for external use.
type EngineConfig struct {
	DefaultTopK      int
	DefaultModels    []string
	DefaultMerge     string
	DefaultDiversity float64
}
