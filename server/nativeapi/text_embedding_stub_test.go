package nativeapi

import (
	"math"
	"testing"

	"github.com/navidrome/navidrome/conf"
	"github.com/navidrome/navidrome/recommender/engine"
)

func TestDeterministicTextEmbedding(t *testing.T) {
	vecA := deterministicTextEmbedding("test prompt", "qwen3", "lyrics", 16)
	vecB := deterministicTextEmbedding("test prompt", "qwen3", "lyrics", 16)

	if len(vecA) != 16 || len(vecB) != 16 {
		t.Fatalf("expected vectors to have dimension 16, got %d and %d", len(vecA), len(vecB))
	}
	for i := range vecA {
		if vecA[i] != vecB[i] {
			t.Fatalf("expected deterministic vectors to match at index %d", i)
		}
	}
}

func TestDeterministicTextEmbeddingVariesByInput(t *testing.T) {
	lyricsVec := deterministicTextEmbedding("same prompt", "qwen3", "lyrics", 16)
	descVec := deterministicTextEmbedding("same prompt", "qwen3", "description", 16)
	otherPrompt := deterministicTextEmbedding("different prompt", "qwen3", "lyrics", 16)

	if vectorsEqual(lyricsVec, descVec) {
		t.Fatalf("expected different targets to produce different embeddings")
	}
	if vectorsEqual(lyricsVec, otherPrompt) {
		t.Fatalf("expected different text to produce different embeddings")
	}
}

func TestDeterministicTextEmbeddingNormalizesVector(t *testing.T) {
	vec := deterministicTextEmbedding("another prompt", "stub", "description", 32)
	var norm float64
	for _, v := range vec {
		norm += v * v
	}
	norm = math.Sqrt(norm)
	if math.Abs(norm-1.0) > 1e-9 {
		t.Fatalf("expected normalized vector norm ~= 1.0, got %.12f", norm)
	}
}

func TestDeterministicTextEmbeddingUsesDefaultDimension(t *testing.T) {
	vec := deterministicTextEmbedding("fallback dim", "stub", "lyrics", 0)
	if len(vec) != defaultTextEmbeddingDim {
		t.Fatalf("expected default dimension %d, got %d", defaultTextEmbeddingDim, len(vec))
	}
}

func TestEmbeddingDimensionForModel(t *testing.T) {
	previous := conf.Server.Recommendations.Milvus.Dimensions
	t.Cleanup(func() {
		conf.Server.Recommendations.Milvus.Dimensions = previous
	})

	conf.Server.Recommendations.Milvus.Dimensions.Lyrics = 111
	conf.Server.Recommendations.Milvus.Dimensions.Description = 222
	conf.Server.Recommendations.Milvus.Dimensions.Flamingo = 333

	if got := embeddingDimensionForModel(engine.ModelLyrics); got != 111 {
		t.Fatalf("expected lyrics dimension 111, got %d", got)
	}
	if got := embeddingDimensionForModel(engine.ModelDescription); got != 222 {
		t.Fatalf("expected description dimension 222, got %d", got)
	}
	if got := embeddingDimensionForModel(engine.ModelFlamingo); got != 333 {
		t.Fatalf("expected flamingo dimension 333, got %d", got)
	}
	if got := embeddingDimensionForModel("unknown"); got != defaultTextEmbeddingDim {
		t.Fatalf("expected default dimension %d, got %d", defaultTextEmbeddingDim, got)
	}
}

func TestHashToSeedDeterministic(t *testing.T) {
	a := hashToSeed("abc", "def", "ghi")
	b := hashToSeed("abc", "def", "ghi")
	c := hashToSeed("abc", "def", "xyz")

	if a != b {
		t.Fatalf("expected deterministic seed for equal input, got %d and %d", a, b)
	}
	if a == c {
		t.Fatalf("expected different inputs to produce different seeds")
	}
}

func vectorsEqual(a []float64, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
