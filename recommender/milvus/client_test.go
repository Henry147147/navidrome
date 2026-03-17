package milvus

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestCollectionConstants(t *testing.T) {
	assert.Equal(t, "muq_audio_embedding", CollectionMuQAudio)
	assert.Equal(t, "muq_mulan_embedding", CollectionMuQMulan)
	assert.Equal(t, CollectionMuQMulan, CollectionLyrics)
	assert.Equal(t, CollectionMuQMulan, CollectionDescription)
	assert.Equal(t, CollectionMuQAudio, CollectionFlamingo)
}

func TestDimensionConstants(t *testing.T) {
	assert.Equal(t, 1024, DimMuQAudio)
	assert.Equal(t, 512, DimMuQMulan)
	assert.Equal(t, DimMuQMulan, DimLyrics)
	assert.Equal(t, DimMuQMulan, DimDescription)
	assert.Equal(t, DimMuQAudio, DimFlamingo)
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
			MuQAudio: 111,
			MuQMulan: 222,
		},
	}

	assert.Equal(t, "http://custom:19530", cfg.URI)
	assert.Equal(t, 60*time.Second, cfg.Timeout)
	assert.Equal(t, 5, cfg.MaxRetries)
	assert.Equal(t, 111, cfg.Dimensions.MuQAudio)
	assert.Equal(t, 222, cfg.Dimensions.MuQMulan)
}

func TestEmbeddingData(t *testing.T) {
	data := EmbeddingData{
		Name:      "Test Track",
		Embedding: []float64{0.1, 0.2, 0.3},
		Offset:    0.5,
		ModelID:   "muq_mulan",
	}

	assert.Equal(t, "Test Track", data.Name)
	assert.Equal(t, []float64{0.1, 0.2, 0.3}, data.Embedding)
	assert.Equal(t, 0.5, data.Offset)
	assert.Equal(t, "muq_mulan", data.ModelID)
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
		MuQAudio: 0,
		MuQMulan: -1,
	})

	assert.Equal(t, DimMuQAudio, dims.MuQAudio)
	assert.Equal(t, DimMuQMulan, dims.MuQMulan)
}
