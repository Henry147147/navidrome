package engine

import (
	"context"
	"math"

	"github.com/navidrome/navidrome/log"
)

// applyNegativePenalties reduces scores for tracks similar to negative prompts.
func (e *Engine) applyNegativePenalties(ctx context.Context, candidates []candidate, req RecommendationRequest) {
	if len(candidates) == 0 {
		return
	}

	// Get negative embeddings for the primary model
	primaryModel := req.Models[0]
	negEmbeddings, ok := req.NegativeEmbeddings[primaryModel]
	if !ok || len(negEmbeddings) == 0 {
		return
	}

	// Get embeddings for candidates
	names := make([]string, len(candidates))
	for i, c := range candidates {
		names[i] = c.Name
	}

	collection := CollectionForModel(primaryModel)
	trackEmbeddings, err := e.milvus.GetByNames(ctx, collection, names)
	if err != nil {
		log.Warn(ctx, "Failed to get candidate embeddings for negative penalty", "error", err)
		return
	}

	// Default penalty factor
	penaltyFactor := req.NegativePromptPenalty
	if penaltyFactor <= 0 {
		penaltyFactor = 0.85
	}

	// Apply penalties
	for i := range candidates {
		trackEmb, ok := trackEmbeddings[candidates[i].Name]
		if !ok {
			continue
		}

		// Find maximum similarity to any negative embedding
		maxSim := 0.0
		for _, negEmb := range negEmbeddings {
			sim := cosineSimilarity(trackEmb, negEmb)
			if sim > maxSim {
				maxSim = sim
			}
		}

		if maxSim > 0 {
			// Apply penalty: higher similarity = more penalty
			// penalty = 1 - (maxSim * (1 - penaltyFactor))
			// e.g., with penaltyFactor=0.85 and maxSim=1.0, penalty=0.85
			penalty := 1.0 - (maxSim * (1.0 - penaltyFactor))
			candidates[i].Score *= penalty
			candidates[i].NegativeSimilarity = &maxSim
		}
	}

	log.Debug(ctx, "Applied negative penalties",
		"candidates", len(candidates),
		"negativeEmbeddings", len(negEmbeddings),
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
