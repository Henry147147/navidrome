package engine

// mergeUnion combines all results, keeping tracks that appear in at least minAgreement models.
func (e *Engine) mergeUnion(results map[string][]candidate, minAgreement int) []candidate {
	// Track: name -> aggregated candidate
	trackScores := make(map[string]*candidate)

	for model, candidates := range results {
		for _, c := range candidates {
			if existing, ok := trackScores[c.Name]; ok {
				existing.Scores = append(existing.Scores, c.Score)
				existing.Models = append(existing.Models, model)
			} else {
				trackScores[c.Name] = &candidate{
					Name:   c.Name,
					Scores: []float64{c.Score},
					Models: []string{model},
				}
			}
		}
	}

	// Filter by minimum agreement and compute final scores
	merged := make([]candidate, 0, len(trackScores))
	for _, c := range trackScores {
		if len(c.Models) >= minAgreement {
			c.Score = average(c.Scores)
			merged = append(merged, *c)
		}
	}

	return merged
}

// mergeIntersection returns only tracks found in ALL models.
func (e *Engine) mergeIntersection(results map[string][]candidate) []candidate {
	if len(results) == 0 {
		return nil
	}

	// Build sets of names for each model
	modelSets := make([]map[string]float64, 0, len(results))
	for _, candidates := range results {
		nameSet := make(map[string]float64)
		for _, c := range candidates {
			nameSet[c.Name] = c.Score
		}
		modelSets = append(modelSets, nameSet)
	}

	if len(modelSets) == 0 {
		return nil
	}

	// Find intersection
	common := make(map[string]bool)
	for name := range modelSets[0] {
		inAll := true
		for i := 1; i < len(modelSets); i++ {
			if _, ok := modelSets[i][name]; !ok {
				inAll = false
				break
			}
		}
		if inAll {
			common[name] = true
		}
	}

	// Build candidates with combined scores
	var modelNames []string
	for model := range results {
		modelNames = append(modelNames, model)
	}

	merged := make([]candidate, 0, len(common))
	for name := range common {
		var scores []float64
		for _, candidates := range results {
			for _, c := range candidates {
				if c.Name == name {
					scores = append(scores, c.Score)
					break
				}
			}
		}
		merged = append(merged, candidate{
			Name:   name,
			Score:  average(scores),
			Scores: scores,
			Models: modelNames,
		})
	}

	return merged
}

// mergePriority tries intersection first, then falls back to highest priority model.
func (e *Engine) mergePriority(results map[string][]candidate, priorities map[string]int, topK int) []candidate {
	// Try intersection first
	intersection := e.mergeIntersection(results)
	if len(intersection) >= topK {
		return intersection
	}

	// Find primary model (lowest priority number = highest priority)
	var primaryModel string
	minPriority := 999999
	for model := range results {
		priority, ok := priorities[model]
		if !ok {
			priority = 100 // Default priority
		}
		if priority < minPriority {
			minPriority = priority
			primaryModel = model
		}
	}

	// If no primary model found, use first available
	if primaryModel == "" {
		for model := range results {
			primaryModel = model
			break
		}
	}

	// Fall back to primary model results
	if candidates, ok := results[primaryModel]; ok {
		// Add intersection results first, then primary model results
		seen := make(map[string]bool)
		merged := make([]candidate, 0, len(intersection)+len(candidates))

		for _, c := range intersection {
			seen[c.Name] = true
			merged = append(merged, c)
		}

		for _, c := range candidates {
			if !seen[c.Name] {
				seen[c.Name] = true
				merged = append(merged, c)
			}
		}

		return merged
	}

	return intersection
}

// average computes the arithmetic mean of a slice.
func average(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	var sum float64
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}
