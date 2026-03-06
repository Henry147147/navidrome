package nativeapi

import (
	"strings"

	"github.com/navidrome/navidrome/recommender/engine"
)

const defaultRecommendationModelAudio = engine.ModelMuQAudio

func canonicalModelsFromValue(value string) []string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "", "none":
		return nil
	case engine.ModelMuQAudio, "audio", "music_flamingo", "music-flamingo", "flamingo":
		return []string{engine.ModelMuQAudio}
	case engine.ModelMuQMulan, "lyrics", "description", "lyric", "desc", "qwen3", "qwen8b", "qwen-8b":
		return []string{engine.ModelMuQMulan}
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
		return []string{engine.ModelMuQMulan}
	}
	return []string{engine.ModelMuQMulan}
}

func ensureTextRecommendationModels(models []string, textTargets []string) []string {
	_ = textTargets
	normalized := normalizeModelList(models)
	if len(normalized) == 0 {
		return []string{engine.ModelMuQMulan}
	}
	for _, model := range normalized {
		if model == engine.ModelMuQMulan {
			return normalized
		}
	}
	normalized = append(normalized, engine.ModelMuQMulan)
	return normalized
}

func normalizeTextEmbedderModel(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "", engine.ModelMuQMulan, "lyrics", "lyric", "description", "desc", "qwen3", "qwen8b", "qwen-8b":
		return engine.ModelMuQMulan
	default:
		return engine.ModelMuQMulan
	}
}
