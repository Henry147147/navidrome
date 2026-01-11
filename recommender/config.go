package recommender

import "time"

// MilvusConfig holds Milvus connection settings.
// Note: This is re-exported from milvus subpackage for external use.
type MilvusConfig struct {
	URI        string           // Server URI or file path for Milvus Lite
	Timeout    time.Duration    // Connection/operation timeout
	MaxRetries int              // Max retry attempts
	Dimensions MilvusDimensions // Embedding dimensions by collection
}

// MilvusDimensions holds embedding dimensions for Milvus collections.
// Note: This is re-exported from milvus subpackage for external use.
type MilvusDimensions struct {
	Lyrics      int
	Description int
	Flamingo    int
}

// EngineConfig holds recommendation engine configuration.
// Note: This is re-exported from engine subpackage for external use.
type EngineConfig struct {
	DefaultTopK      int
	DefaultModels    []string
	DefaultMerge     string
	DefaultDiversity float64
}
