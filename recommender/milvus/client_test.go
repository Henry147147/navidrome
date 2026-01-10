package milvus

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestCollectionConstants(t *testing.T) {
	assert.Equal(t, "lyrics_embedding", CollectionLyrics)
	assert.Equal(t, "description_embedding", CollectionDescription)
	assert.Equal(t, "flamingo_audio_embedding", CollectionFlamingo)
}

func TestDimensionConstants(t *testing.T) {
	assert.Equal(t, 2560, DimLyrics)
	assert.Equal(t, 2560, DimDescription)
	assert.Equal(t, 3584, DimFlamingo)
}

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()

	assert.Equal(t, "http://localhost:19530", cfg.URI)
	assert.Equal(t, 30*time.Second, cfg.Timeout)
	assert.Equal(t, 3, cfg.MaxRetries)
	assert.Equal(t, DefaultDimensions(), cfg.Dimensions)
}

func TestConfig(t *testing.T) {
	cfg := Config{
		URI:        "http://custom:19530",
		Timeout:    60 * time.Second,
		MaxRetries: 5,
		Dimensions: Dimensions{
			Lyrics:      111,
			Description: 222,
			Flamingo:    333,
		},
	}

	assert.Equal(t, "http://custom:19530", cfg.URI)
	assert.Equal(t, 60*time.Second, cfg.Timeout)
	assert.Equal(t, 5, cfg.MaxRetries)
	assert.Equal(t, 111, cfg.Dimensions.Lyrics)
	assert.Equal(t, 222, cfg.Dimensions.Description)
	assert.Equal(t, 333, cfg.Dimensions.Flamingo)
}

func TestEmbeddingData(t *testing.T) {
	data := EmbeddingData{
		Name:        "Test Track",
		Embedding:   []float64{0.1, 0.2, 0.3},
		Offset:      0.5,
		ModelID:     "lyrics",
		Description: "A test description",
	}

	assert.Equal(t, "Test Track", data.Name)
	assert.Equal(t, []float64{0.1, 0.2, 0.3}, data.Embedding)
	assert.Equal(t, 0.5, data.Offset)
	assert.Equal(t, "lyrics", data.ModelID)
	assert.Equal(t, "A test description", data.Description)
}

func TestSearchOptions(t *testing.T) {
	opts := SearchOptions{
		TopK:         100,
		ExcludeNames: []string{"exclude1", "exclude2"},
	}

	assert.Equal(t, 100, opts.TopK)
	assert.Equal(t, []string{"exclude1", "exclude2"}, opts.ExcludeNames)
}

func TestSearchResult(t *testing.T) {
	result := SearchResult{
		Name:     "Test Track",
		Distance: 0.15,
	}

	assert.Equal(t, "Test Track", result.Name)
	assert.Equal(t, 0.15, result.Distance)
}

func TestNormalizeDimensions(t *testing.T) {
	dims := normalizeDimensions(Dimensions{
		Lyrics:      0,
		Description: 1024,
		Flamingo:    -1,
	})

	assert.Equal(t, DimLyrics, dims.Lyrics)
	assert.Equal(t, 1024, dims.Description)
	assert.Equal(t, DimFlamingo, dims.Flamingo)
}
