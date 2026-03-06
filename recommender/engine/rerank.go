package engine

import (
	"context"
	"math"
	"strings"
)

func (e *Engine) attachCandidateEmbeddings(ctx context.Context, candidates []candidate, models []string) {
	if len(candidates) == 0 {
		return
	}

	names := make([]string, 0, len(candidates))
	for _, candidate := range candidates {
		names = append(names, candidate.Name)
	}

	for _, model := range models {
		embeddings, err := e.milvus.GetByNames(ctx, CollectionForModel(model), names)
		if err != nil {
			continue
		}
		for idx := range candidates {
			embedding, ok := embeddings[candidates[idx].Name]
			if !ok || len(embedding) == 0 {
				continue
			}
			if candidates[idx].Embeddings == nil {
				candidates[idx].Embeddings = make(map[string][]float64)
			}
			candidates[idx].Embeddings[model] = embedding
		}
	}
}

func (e *Engine) rerankCandidates(candidates []candidate, req RecommendationRequest) []candidate {
	if len(candidates) == 0 {
		return nil
	}

	limit := req.Limit
	if limit <= 0 || limit > len(candidates) {
		limit = len(candidates)
	}

	diversity := req.Diversity
	if diversity < 0 {
		diversity = 0
	}
	if diversity > 1 {
		diversity = 1
	}

	relevance := normalizeCandidateScores(candidates)
	selected := make([]candidate, 0, limit)
	used := make([]bool, len(candidates))
	artistCounts := make(map[string]int)

	for len(selected) < limit {
		bestIdx := -1
		bestScore := math.Inf(-1)
		bestRelevance := math.Inf(-1)
		bestName := ""

		for idx, candidate := range candidates {
			if used[idx] {
				continue
			}

			adjustedRelevance := relevance[idx] - candidate.NegativePenalty
			if adjustedRelevance < 0 {
				adjustedRelevance = 0
			}

			redundancy := 0.0
			if diversity > 0 && len(selected) > 0 {
				redundancy = maxRedundancy(candidate, selected, req.Models)
			}

			score := (1-diversity)*adjustedRelevance - diversity*redundancy
			score -= artistRepeatPenalty(candidateArtist(candidate.Name), artistCounts)

			if score > bestScore ||
				(score == bestScore && adjustedRelevance > bestRelevance) ||
				(score == bestScore && adjustedRelevance == bestRelevance && (bestName == "" || candidate.Name < bestName)) {
				bestIdx = idx
				bestScore = score
				bestRelevance = adjustedRelevance
				bestName = candidate.Name
			}
		}

		if bestIdx < 0 {
			break
		}

		chosen := candidates[bestIdx]
		chosen.Score = bestScore
		selected = append(selected, chosen)
		used[bestIdx] = true

		artist := candidateArtist(chosen.Name)
		if artist != "" {
			artistCounts[artist]++
		}
	}

	return selected
}

func normalizeCandidateScores(candidates []candidate) []float64 {
	values := make([]float64, len(candidates))
	if len(candidates) == 0 {
		return values
	}

	minScore := candidates[0].BaseScore
	maxScore := candidates[0].BaseScore
	for _, candidate := range candidates[1:] {
		if candidate.BaseScore < minScore {
			minScore = candidate.BaseScore
		}
		if candidate.BaseScore > maxScore {
			maxScore = candidate.BaseScore
		}
	}

	if maxScore == minScore {
		for idx := range values {
			values[idx] = 1
		}
		return values
	}

	rangeScore := maxScore - minScore
	for idx, candidate := range candidates {
		values[idx] = (candidate.BaseScore - minScore) / rangeScore
	}
	return values
}

func maxRedundancy(candidate candidate, selected []candidate, models []string) float64 {
	maxSimilarity := 0.0
	for _, chosen := range selected {
		similarity := candidateSimilarity(candidate, chosen, models)
		if similarity > maxSimilarity {
			maxSimilarity = similarity
		}
	}
	return maxSimilarity
}

func candidateSimilarity(left candidate, right candidate, models []string) float64 {
	total := 0.0
	count := 0
	for _, model := range models {
		leftEmbedding, leftOK := left.Embeddings[model]
		rightEmbedding, rightOK := right.Embeddings[model]
		if !leftOK || !rightOK {
			continue
		}
		total += cosineSimilarity(leftEmbedding, rightEmbedding)
		count++
	}
	if count == 0 {
		return 0
	}
	return total / float64(count)
}

func candidateArtist(name string) string {
	parts := strings.SplitN(name, " - ", 2)
	if len(parts) == 0 {
		return ""
	}
	return strings.TrimSpace(parts[0])
}

func artistRepeatPenalty(artist string, artistCounts map[string]int) float64 {
	if artist == "" {
		return 0
	}
	switch artistCounts[artist] {
	case 0:
		return 0
	case 1:
		return 0.05
	default:
		return 0.10
	}
}
