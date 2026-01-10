package milvus

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestDimensionForCollection(t *testing.T) {
	dims := Dimensions{
		Lyrics:      11,
		Description: 22,
		Flamingo:    33,
	}

	tests := []struct {
		name       string
		collection string
		expected   int
	}{
		{"lyrics collection", CollectionLyrics, dims.Lyrics},
		{"description collection", CollectionDescription, dims.Description},
		{"flamingo collection", CollectionFlamingo, dims.Flamingo},
		{"unknown defaults to lyrics", "unknown_collection", dims.Lyrics},
		{"empty string defaults to lyrics", "", dims.Lyrics},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := dimensionForCollection(dims, tt.collection)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestBuildInFilter(t *testing.T) {
	tests := []struct {
		name     string
		field    string
		values   []string
		expected string
	}{
		{
			"single value",
			"name",
			[]string{"track1"},
			"name in ['track1']",
		},
		{
			"multiple values",
			"name",
			[]string{"track1", "track2", "track3"},
			"name in ['track1', 'track2', 'track3']",
		},
		{
			"empty values",
			"name",
			[]string{},
			"",
		},
		{
			"value with single quote",
			"name",
			[]string{"It's a test"},
			"name in ['It\\'s a test']",
		},
		{
			"multiple values with quotes",
			"name",
			[]string{"Track's One", "Track's Two"},
			"name in ['Track\\'s One', 'Track\\'s Two']",
		},
		{
			"different field name",
			"model_id",
			[]string{"lyrics", "description"},
			"model_id in ['lyrics', 'description']",
		},
		{
			"value with special characters",
			"name",
			[]string{"Artist - Track (Remix)"},
			"name in ['Artist - Track (Remix)']",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := buildInFilter(tt.field, tt.values)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestBuildNotInFilter(t *testing.T) {
	tests := []struct {
		name     string
		field    string
		values   []string
		expected string
	}{
		{
			"single value",
			"name",
			[]string{"track1"},
			"name not in ['track1']",
		},
		{
			"multiple values",
			"name",
			[]string{"track1", "track2"},
			"name not in ['track1', 'track2']",
		},
		{
			"empty values",
			"name",
			[]string{},
			"",
		},
		{
			"value with single quote",
			"name",
			[]string{"It's excluded"},
			"name not in ['It\\'s excluded']",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := buildNotInFilter(tt.field, tt.values)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestFloat64sToFloat32s(t *testing.T) {
	tests := []struct {
		name     string
		input    []float64
		expected []float32
	}{
		{
			"empty slice",
			[]float64{},
			[]float32{},
		},
		{
			"single value",
			[]float64{1.5},
			[]float32{1.5},
		},
		{
			"multiple values",
			[]float64{0.1, 0.2, 0.3, 0.4, 0.5},
			[]float32{0.1, 0.2, 0.3, 0.4, 0.5},
		},
		{
			"large values",
			[]float64{1000.5, 2000.25, 3000.125},
			[]float32{1000.5, 2000.25, 3000.125},
		},
		{
			"negative values",
			[]float64{-1.0, -2.5, -3.75},
			[]float32{-1.0, -2.5, -3.75},
		},
		{
			"zero",
			[]float64{0.0},
			[]float32{0.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := float64sToFloat32s(tt.input)
			assert.Len(t, result, len(tt.expected))
			for i := range tt.expected {
				assert.InDelta(t, tt.expected[i], result[i], 0.0001)
			}
		})
	}
}

func TestFloat32sToFloat64s(t *testing.T) {
	tests := []struct {
		name     string
		input    []float32
		expected []float64
	}{
		{
			"empty slice",
			[]float32{},
			[]float64{},
		},
		{
			"single value",
			[]float32{1.5},
			[]float64{1.5},
		},
		{
			"multiple values",
			[]float32{0.1, 0.2, 0.3, 0.4, 0.5},
			[]float64{0.1, 0.2, 0.3, 0.4, 0.5},
		},
		{
			"large values",
			[]float32{1000.5, 2000.25, 3000.125},
			[]float64{1000.5, 2000.25, 3000.125},
		},
		{
			"negative values",
			[]float32{-1.0, -2.5, -3.75},
			[]float64{-1.0, -2.5, -3.75},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := float32sToFloat64s(tt.input)
			assert.Len(t, result, len(tt.expected))
			for i := range tt.expected {
				assert.InDelta(t, tt.expected[i], result[i], 0.0001)
			}
		})
	}
}

func TestFloatConversionRoundTrip(t *testing.T) {
	// Test that converting from float64 -> float32 -> float64 preserves reasonable precision
	original := []float64{0.123456789, 0.987654321, 0.555555555}

	asFloat32 := float64sToFloat32s(original)
	backToFloat64 := float32sToFloat64s(asFloat32)

	assert.Len(t, backToFloat64, len(original))
	for i := range original {
		// float32 has less precision, so we accept some loss
		assert.InDelta(t, original[i], backToFloat64[i], 0.0001)
	}
}

func TestEmbeddingDataStruct(t *testing.T) {
	data := EmbeddingData{
		Name:        "Test Track - Artist",
		Embedding:   []float64{0.1, 0.2, 0.3, 0.4},
		Offset:      0.5,
		ModelID:     "lyrics",
		Lyrics:      "Some lyrics",
		Description: "A beautiful test track",
	}

	assert.Equal(t, "Test Track - Artist", data.Name)
	assert.Equal(t, []float64{0.1, 0.2, 0.3, 0.4}, data.Embedding)
	assert.Equal(t, 0.5, data.Offset)
	assert.Equal(t, "lyrics", data.ModelID)
	assert.Equal(t, "Some lyrics", data.Lyrics)
	assert.Equal(t, "A beautiful test track", data.Description)
}

func TestEmbeddingDataWithEmptyFields(t *testing.T) {
	// Lyrics/Description are optional (collection-specific)
	data := EmbeddingData{
		Name:      "Track Name",
		Embedding: []float64{0.1},
		ModelID:   "flamingo",
	}

	assert.Empty(t, data.Lyrics)
	assert.Empty(t, data.Description)
	assert.Equal(t, 0.0, data.Offset)
}

func TestSearchOptionsWithExcludes(t *testing.T) {
	opts := SearchOptions{
		TopK:         100,
		ExcludeNames: []string{"exclude1", "exclude2", "exclude3"},
	}

	assert.Equal(t, 100, opts.TopK)
	assert.Len(t, opts.ExcludeNames, 3)
}

func TestSearchOptionsZeroValues(t *testing.T) {
	opts := SearchOptions{}

	assert.Equal(t, 0, opts.TopK)
	assert.Nil(t, opts.ExcludeNames)
}

func TestSearchResultWithDistance(t *testing.T) {
	result := SearchResult{
		Name:     "Found Track",
		Distance: 0.123,
	}

	assert.Equal(t, "Found Track", result.Name)
	assert.InDelta(t, 0.123, result.Distance, 0.0001)
}

func TestBuildFilterWithUnicodeCharacters(t *testing.T) {
	// Test Unicode handling in filters
	values := []string{
		"日本語トラック",         // Japanese
		"Трек на русском", // Russian
		"Piste française", // French with accent
	}

	filter := buildInFilter("name", values)
	assert.Contains(t, filter, "日本語トラック")
	assert.Contains(t, filter, "Трек на русском")
	assert.Contains(t, filter, "Piste française")
}

func TestBuildFilterWithMultipleQuotes(t *testing.T) {
	values := []string{"It's 'quoted'"}
	filter := buildInFilter("name", values)
	// Both single quotes should be escaped
	assert.Contains(t, filter, "It\\'s \\'quoted\\'")
}
