package nativeapi

import (
	"strings"

	"github.com/navidrome/navidrome/recommender/engine"
)

const defaultRecommendationModelAudio = engine.ModelFlamingo

func canonicalModelsFromValue(value string) []string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "", "none":
		return nil
	case engine.ModelFlamingo, "audio", "music_flamingo", "music-flamingo":
		return []string{engine.ModelFlamingo}
	case engine.ModelLyrics, "lyric":
		return []string{engine.ModelLyrics}
	case engine.ModelDescription, "desc":
		return []string{engine.ModelDescription}
	case "qwen3", "qwen8b", "qwen-8b":
		return []string{engine.ModelLyrics, engine.ModelDescription}
	default:
		return nil
	}
}

func normalizeModelList(models []string) []string {
	if len(models) == 0 {
		return nil
	}

	result := make([]string, 0, len(models))
	seen := make(map[string]struct{}, len(models))
	for _, model := range models {
		for _, canonical := range canonicalModelsFromValue(model) {
			if _, ok := seen[canonical]; ok {
				continue
			}
			seen[canonical] = struct{}{}
			result = append(result, canonical)
		}
	}
	return result
}

func normalizeRecommendationModels(models []string, fallback []string) []string {
	normalized := normalizeModelList(models)
	if len(normalized) > 0 {
		return normalized
	}

	normalizedFallback := normalizeModelList(fallback)
	if len(normalizedFallback) > 0 {
		return normalizedFallback
	}
	return []string{defaultRecommendationModelAudio}
}

func normalizeModelPriorities(priorities map[string]int) map[string]int {
	if len(priorities) == 0 {
		return nil
	}

	result := make(map[string]int)
	for key, value := range priorities {
		for _, canonical := range canonicalModelsFromValue(key) {
			existing, ok := result[canonical]
			if !ok || value < existing {
				result[canonical] = value
			}
		}
	}

	if len(result) == 0 {
		return nil
	}
	return result
}

func normalizeNegativeEmbeddings(raw map[string][][]float64) map[string][][]float64 {
	if len(raw) == 0 {
		return nil
	}

	result := make(map[string][][]float64)
	for key, embeddings := range raw {
		canonical := canonicalModelsFromValue(key)
		if len(canonical) == 0 || len(embeddings) == 0 {
			continue
		}
		for _, model := range canonical {
			result[model] = append(result[model], embeddings...)
		}
	}

	if len(result) == 0 {
		return nil
	}
	return result
}

func normalizeTextTargets(targets []string, legacyModel string) []string {
	raw := targets
	if len(raw) == 0 && strings.TrimSpace(legacyModel) != "" {
		raw = []string{legacyModel}
	}
	if len(raw) == 0 {
		raw = []string{engine.ModelLyrics, engine.ModelDescription}
	}

	normalized := normalizeModelList(raw)
	if len(normalized) == 0 {
		return []string{engine.ModelLyrics, engine.ModelDescription}
	}

	filtered := make([]string, 0, len(normalized))
	for _, model := range normalized {
		if model == engine.ModelLyrics || model == engine.ModelDescription {
			filtered = append(filtered, model)
		}
	}

	if len(filtered) == 0 {
		return []string{engine.ModelLyrics, engine.ModelDescription}
	}
	return filtered
}

func ensureTextRecommendationModels(models []string, textTargets []string) []string {
	normalizedTargets := normalizeTextTargets(textTargets, "")
	combined := append([]string{}, models...)
	combined = append(combined, normalizedTargets...)
	normalized := normalizeModelList(combined)
	if len(normalized) == 0 {
		return normalizedTargets
	}
	return normalized
}
