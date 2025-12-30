package engine

import (
	"context"

	"github.com/navidrome/navidrome/log"
	"github.com/navidrome/navidrome/recommender/milvus"
)

// searchMultiModel performs similarity search across multiple models and merges results.
func (e *Engine) searchMultiModel(ctx context.Context, seedEmbeddings map[string]map[string][]float64, req RecommendationRequest, excludeNames []string) ([]candidate, error) {
	// Results per model: model -> []candidate
	modelResults := make(map[string][]candidate)

	// Search each model with its seed embeddings
	for _, model := range req.Models {
		seeds := seedEmbeddings[model]
		if len(seeds) == 0 {
			continue
		}

		collection := CollectionForModel(model)
		topK := e.config.DefaultTopK
		if topK < req.Limit*3 {
			topK = req.Limit * 3 // Search for more than we need to allow for filtering
		}

		// Collect all seed vectors for this model
		vectors := make([][]float64, 0, len(seeds))
		weights := make([]float64, 0, len(seeds))
		for seedID, emb := range seeds {
			vectors = append(vectors, emb)
			// Find weight for this seed
			weight := 1.0
			for _, seed := range req.Seeds {
				if seed.TrackID == seedID || (len(seed.Embedding) > 0 && seedID[:7] == "direct_") {
					if seed.Weight > 0 {
						weight = seed.Weight
					}
					break
				}
			}
			weights = append(weights, weight)
		}

		// Perform search
		hits, err := e.milvus.SearchMultiple(ctx, collection, vectors, milvus.SearchOptions{
			TopK:         topK,
			ExcludeNames: excludeNames,
		})
		if err != nil {
			log.Warn(ctx, "Search failed for model", "model", model, "error", err)
			continue
		}

		// Convert hits to candidates with weighted scores
		candidates := make([]candidate, 0, len(hits))
		for _, hit := range hits {
			// Apply diversity if configured
			score := hit.Distance
			if req.Diversity > 0 {
				score = score * (1 - req.Diversity)
			}

			candidates = append(candidates, candidate{
				Name:   hit.Name,
				Score:  score,
				Scores: []float64{score},
				Models: []string{model},
			})
		}

		modelResults[model] = candidates
		log.Debug(ctx, "Model search complete", "model", model, "hits", len(hits))
	}

	// Merge results based on strategy
	var merged []candidate
	switch req.MergeStrategy {
	case "intersection":
		merged = e.mergeIntersection(modelResults)
	case "priority":
		merged = e.mergePriority(modelResults, req.ModelPriorities, req.Limit)
	default: // "union"
		minAgreement := req.MinModelAgreement
		if minAgreement <= 0 {
			minAgreement = 1
		}
		merged = e.mergeUnion(modelResults, minAgreement)
	}

	return merged, nil
}

// searchSingleModel performs similarity search with a single model.
func (e *Engine) searchSingleModel(ctx context.Context, model string, vectors [][]float64, excludeNames []string, topK int) ([]candidate, error) {
	collection := CollectionForModel(model)

	hits, err := e.milvus.SearchMultiple(ctx, collection, vectors, milvus.SearchOptions{
		TopK:         topK,
		ExcludeNames: excludeNames,
	})
	if err != nil {
		return nil, err
	}

	candidates := make([]candidate, 0, len(hits))
	for _, hit := range hits {
		candidates = append(candidates, candidate{
			Name:   hit.Name,
			Score:  hit.Distance,
			Scores: []float64{hit.Distance},
			Models: []string{model},
		})
	}

	return candidates, nil
}
