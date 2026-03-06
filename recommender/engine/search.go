package engine

import (
	"context"
	"sort"

	"github.com/navidrome/navidrome/log"
	"github.com/navidrome/navidrome/recommender/milvus"
)

const (
	rrfRankConstant      = 60
	minSearchCandidates  = 100
	maxSearchCandidates  = 500
	defaultPriorityValue = 100
)

// searchMultiModel performs similarity search across multiple models and merges results.
func (e *Engine) searchMultiModel(ctx context.Context, seedEmbeddings map[string][]seedEmbedding, req RecommendationRequest, excludeNames []string) ([]candidate, error) {
	modelResults := make(map[string][]candidate)
	topK := expandedSearchLimit(req.Limit)

	for _, model := range req.Models {
		seeds := seedEmbeddings[model]
		if len(seeds) == 0 {
			continue
		}

		candidates, err := e.searchSingleModel(ctx, model, seeds, excludeNames, topK)
		if err != nil {
			log.Warn(ctx, "Search failed for model", "model", model, "error", err)
			continue
		}
		modelResults[model] = candidates
		log.Debug(ctx, "Model search complete", "model", model, "hits", len(candidates))
	}

	if len(modelResults) == 0 {
		return nil, nil
	}

	minAgreement := req.MinModelAgreement
	if minAgreement <= 0 {
		minAgreement = 1
	}

	var merged []candidate
	switch req.MergeStrategy {
	case "intersection":
		merged = e.mergeIntersection(modelResults, req.Models)
	case "priority":
		merged = e.mergePriority(modelResults, req.Models, req.ModelPriorities, req.Limit, minAgreement)
	default:
		merged = e.mergeUnion(modelResults, req.Models, minAgreement)
	}

	if len(merged) == 0 {
		return nil, nil
	}

	needsEmbeddings := req.Diversity > 0 || len(req.NegativeEmbeddings) > 0
	if needsEmbeddings {
		e.attachCandidateEmbeddings(ctx, merged, req.Models)
	}

	if len(req.NegativeEmbeddings) > 0 {
		e.applyNegativePenalties(ctx, merged, req)
	}

	return e.rerankCandidates(merged, req), nil
}

// searchSingleModel performs similarity search with a single model using weighted per-seed RRF.
func (e *Engine) searchSingleModel(ctx context.Context, model string, seeds []seedEmbedding, excludeNames []string, topK int) ([]candidate, error) {
	if len(seeds) == 0 {
		return nil, nil
	}

	collection := CollectionForModel(model)
	weights := normalizeSeedWeights(seeds)
	aggregated := make(map[string]*candidate)

	for idx, seed := range seeds {
		hits, err := e.milvus.Search(ctx, collection, seed.Embedding, milvus.SearchOptions{
			TopK:         topK,
			ExcludeNames: excludeNames,
		})
		if err != nil {
			return nil, err
		}

		for rank, hit := range hits {
			existing, ok := aggregated[hit.Name]
			if !ok {
				existing = &candidate{
					Name:        hit.Name,
					Models:      []string{model},
					ModelScores: map[string]float64{},
					ModelRanks:  map[string]int{},
				}
				aggregated[hit.Name] = existing
			}
			existing.Score += reciprocalRankContribution(rank+1, weights[idx])
		}
	}

	candidates := make([]candidate, 0, len(aggregated))
	for _, item := range aggregated {
		candidates = append(candidates, *item)
	}
	sortCandidates(candidates)
	for idx := range candidates {
		candidates[idx].BaseScore = candidates[idx].Score
		if candidates[idx].ModelScores == nil {
			candidates[idx].ModelScores = map[string]float64{}
		}
		if candidates[idx].ModelRanks == nil {
			candidates[idx].ModelRanks = map[string]int{}
		}
		candidates[idx].ModelScores[model] = candidates[idx].Score
		candidates[idx].ModelRanks[model] = idx + 1
	}

	return candidates, nil
}

func expandedSearchLimit(limit int) int {
	target := limit * 5
	if target < minSearchCandidates {
		target = minSearchCandidates
	}
	if target > maxSearchCandidates {
		target = maxSearchCandidates
	}
	return target
}

func normalizeSeedWeights(seeds []seedEmbedding) []float64 {
	if len(seeds) == 0 {
		return nil
	}
	weights := make([]float64, len(seeds))
	total := 0.0
	for i, seed := range seeds {
		if seed.Weight > 0 {
			weights[i] = seed.Weight
			total += seed.Weight
		}
	}
	if total == 0 {
		equal := 1.0 / float64(len(seeds))
		for i := range weights {
			weights[i] = equal
		}
		return weights
	}
	for i := range weights {
		weights[i] /= total
	}
	return weights
}

func reciprocalRankContribution(rank int, weight float64) float64 {
	if rank <= 0 {
		return 0
	}
	if weight <= 0 {
		weight = 1
	}
	return weight / float64(rrfRankConstant+rank)
}

func sortCandidates(candidates []candidate) {
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].Score == candidates[j].Score {
			return candidates[i].Name < candidates[j].Name
		}
		return candidates[i].Score > candidates[j].Score
	})
}

func uniqueModelsInOrder(preferred []string, results map[string][]candidate) []string {
	seen := make(map[string]struct{}, len(results))
	models := make([]string, 0, len(results))
	for _, model := range preferred {
		if _, ok := results[model]; !ok {
			continue
		}
		if _, dup := seen[model]; dup {
			continue
		}
		seen[model] = struct{}{}
		models = append(models, model)
	}
	if len(models) == len(results) {
		return models
	}

	extra := make([]string, 0, len(results)-len(models))
	for model := range results {
		if _, dup := seen[model]; dup {
			continue
		}
		extra = append(extra, model)
	}
	sort.Strings(extra)
	return append(models, extra...)
}

func appendModelOnce(models []string, model string) []string {
	for _, existing := range models {
		if existing == model {
			return models
		}
	}
	return append(models, model)
}
