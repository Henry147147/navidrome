package nativeapi

import (
	"math"
	"testing"
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
