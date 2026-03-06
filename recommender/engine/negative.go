package engine

import (
	"context"
	"math"

	"github.com/navidrome/navidrome/log"
)

// applyNegativePenalties reduces relevance for tracks similar to negative prompts across all active models.
func (e *Engine) applyNegativePenalties(ctx context.Context, candidates []candidate, req RecommendationRequest) {
	if len(candidates) == 0 || len(req.NegativeEmbeddings) == 0 {
		return
	}

	modelWeights := unitModelWeights(map[string][]candidate{})
	if req.MergeStrategy == "priority" {
		modelWeights = priorityModelWeights(req.Models, req.ModelPriorities)
	}

	penaltyFactor := req.NegativePromptPenalty
	if penaltyFactor <= 0 {
		penaltyFactor = 0.85
	}

	applied := 0
	for i := range candidates {
		totalWeight := 0.0
		weightedSimilarity := 0.0
		maxSimilarity := 0.0

		for _, model := range req.Models {
			negEmbeddings := req.NegativeEmbeddings[model]
			if len(negEmbeddings) == 0 {
				continue
			}
			trackEmbedding, ok := candidates[i].Embeddings[model]
			if !ok || len(trackEmbedding) == 0 {
				continue
			}

			modelMax := 0.0
			for _, negEmb := range negEmbeddings {
				sim := cosineSimilarity(trackEmbedding, negEmb)
				if sim > modelMax {
					modelMax = sim
				}
			}
			if modelMax > maxSimilarity {
				maxSimilarity = modelMax
			}

			weight := modelWeights[model]
			if weight <= 0 {
				weight = 1
			}
			totalWeight += weight
			weightedSimilarity += modelMax * weight
		}

		if totalWeight == 0 {
			continue
		}

		combinedSimilarity := weightedSimilarity / totalWeight
		penalty := combinedSimilarity * (1 - penaltyFactor)
		if penalty <= 0 {
			continue
		}

		candidates[i].NegativePenalty = penalty
		candidates[i].NegativeSimilarity = &combinedSimilarity
		if maxSimilarity > 0 && maxSimilarity > combinedSimilarity {
			candidates[i].NegativeSimilarity = &maxSimilarity
		}
		applied++
	}

	log.Debug(ctx, "Applied negative penalties",
		"candidates", len(candidates),
		"affected", applied,
		"models", len(req.NegativeEmbeddings),
	)
}

// cosineSimilarity computes the cosine similarity between two vectors.
func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

// euclideanDistance computes the Euclidean distance between two vectors.
//
//nolint:unused // reserved for future alternative distance metrics.
func euclideanDistance(a, b []float64) float64 {
	if len(a) != len(b) {
		return math.MaxFloat64
	}

	var sum float64
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}

	return math.Sqrt(sum)
}
