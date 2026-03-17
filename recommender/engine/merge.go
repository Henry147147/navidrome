package engine

// mergeUnion combines all results, keeping tracks that appear in at least minAgreement models.
func (e *Engine) mergeUnion(results map[string][]candidate, preferredModels []string, minAgreement int) []candidate {
	if minAgreement <= 0 {
		minAgreement = 1
	}
	merged := mergeCandidates(results, preferredModels, unitModelWeights(results), minAgreement)
	sortCandidates(merged)
	return merged
}

// mergeIntersection returns only tracks found in all active models.
func (e *Engine) mergeIntersection(results map[string][]candidate, preferredModels []string) []candidate {
	if len(results) == 0 {
		return nil
	}
	activeModels := uniqueModelsInOrder(preferredModels, results)
	merged := mergeCandidates(results, activeModels, unitModelWeights(results), len(activeModels))
	sortCandidates(merged)
	return merged
}

// mergePriority requires presence in the highest priority model, then backfills from the weighted union.
func (e *Engine) mergePriority(results map[string][]candidate, preferredModels []string, priorities map[string]int, topK int, minAgreement int) []candidate {
	if len(results) == 0 {
		return nil
	}
	if minAgreement <= 0 {
		minAgreement = 1
	}

	activeModels := uniqueModelsInOrder(preferredModels, results)
	weights := priorityModelWeights(activeModels, priorities)
	union := mergeCandidates(results, activeModels, weights, minAgreement)
	sortCandidates(union)
	if len(union) == 0 {
		return nil
	}

	primaryModel := highestPriorityModel(activeModels, priorities)
	primaryOnly := make([]candidate, 0, len(union))
	seen := make(map[string]struct{}, len(union))
	for _, candidate := range union {
		if _, ok := candidate.ModelRanks[primaryModel]; !ok {
			continue
		}
		primaryOnly = append(primaryOnly, candidate)
		seen[candidate.Name] = struct{}{}
	}
	if topK > 0 && len(primaryOnly) >= topK {
		return primaryOnly
	}
	for _, candidate := range union {
		if _, ok := seen[candidate.Name]; ok {
			continue
		}
		primaryOnly = append(primaryOnly, candidate)
	}
	return primaryOnly
}

func mergeCandidates(results map[string][]candidate, preferredModels []string, modelWeights map[string]float64, minAgreement int) []candidate {
	if minAgreement <= 0 {
		minAgreement = 1
	}

	activeModels := uniqueModelsInOrder(preferredModels, results)
	trackScores := make(map[string]*candidate)

	for _, model := range activeModels {
		weight := modelWeights[model]
		if weight <= 0 {
			weight = 1
		}
		for _, c := range results[model] {
			rank := c.ModelRanks[model]
			if rank <= 0 {
				continue
			}
			existing, ok := trackScores[c.Name]
			if !ok {
				existing = &candidate{
					Name:        c.Name,
					ModelScores: map[string]float64{},
					ModelRanks:  map[string]int{},
				}
				trackScores[c.Name] = existing
			}
			existing.Score += reciprocalRankContribution(rank, weight)
			existing.BaseScore = existing.Score
			existing.ModelScores[model] = c.ModelScores[model]
			existing.ModelRanks[model] = rank
			existing.Models = appendModelOnce(existing.Models, model)
		}
	}

	merged := make([]candidate, 0, len(trackScores))
	for _, c := range trackScores {
		if len(c.Models) < minAgreement {
			continue
		}
		merged = append(merged, *c)
	}
	return merged
}

func unitModelWeights(results map[string][]candidate) map[string]float64 {
	weights := make(map[string]float64, len(results))
	for model := range results {
		weights[model] = 1
	}
	return weights
}

func priorityModelWeights(models []string, priorities map[string]int) map[string]float64 {
	weights := make(map[string]float64, len(models))
	for _, model := range models {
		priority := priorities[model]
		if priority <= 0 {
			priority = defaultPriorityValue
		}
		weights[model] = 1.0 / float64(priority)
	}
	return weights
}

func highestPriorityModel(models []string, priorities map[string]int) string {
	var bestModel string
	bestPriority := int(^uint(0) >> 1)
	for _, model := range models {
		priority := priorities[model]
		if priority <= 0 {
			priority = defaultPriorityValue
		}
		if priority < bestPriority {
			bestPriority = priority
			bestModel = model
		}
	}
	if bestModel != "" {
		return bestModel
	}
	for _, model := range models {
		return model
	}
	return ""
}
