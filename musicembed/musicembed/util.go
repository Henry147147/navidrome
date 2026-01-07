package musicembed

import (
	"fmt"
	"math"
)

// CosineSimilarity computes the cosine similarity between two embedding vectors.
// Returns a value between -1 and 1, where 1 means identical direction.
func CosineSimilarity(a, b []float32) float64 {
	if len(a) == 0 || len(b) == 0 || len(a) != len(b) {
		return 0
	}

	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}

	if math.IsNaN(dotProduct) {
		return 0
	}

	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 || math.IsNaN(denom) {
		return 0
	}

	return dotProduct / denom
}

// validateEmbedding checks that an embedding vector is valid.
func validateEmbedding(vec []float32) error {
	var norm float64
	for _, v := range vec {
		fv := float64(v)
		if math.IsNaN(fv) || math.IsInf(fv, 0) {
			return fmt.Errorf("embedding contains invalid value")
		}
		norm += fv * fv
	}
	if norm == 0 || math.IsNaN(norm) {
		return fmt.Errorf("embedding norm invalid: %f", norm)
	}
	return nil
}

// meanPool averages token embeddings into a single vector.
func meanPool(embeddings []float32, nTokens, nEmbd int) []float32 {
	result := make([]float32, nEmbd)
	if nTokens == 0 {
		return result
	}
	for t := 0; t < nTokens; t++ {
		offset := t * nEmbd
		for d := 0; d < nEmbd; d++ {
			result[d] += embeddings[offset+d]
		}
	}
	for d := 0; d < nEmbd; d++ {
		result[d] /= float32(nTokens)
	}
	return result
}

// maxInt returns the larger of two integers.
func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
