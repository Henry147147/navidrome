package nativeapi

import (
	"testing"

	"github.com/navidrome/navidrome/recommender/engine"
)

func TestNormalizeRecommendationModels(t *testing.T) {
	t.Run("defaults to flamingo", func(t *testing.T) {
		got := normalizeRecommendationModels(nil, nil)
		if len(got) != 1 || got[0] != engine.ModelFlamingo {
			t.Fatalf("expected [%s], got %#v", engine.ModelFlamingo, got)
		}
	})

	t.Run("normalizes aliases and qwen3 expansion", func(t *testing.T) {
		got := normalizeRecommendationModels([]string{"audio", "qwen3", "lyric", "desc"}, nil)
		expected := []string{engine.ModelFlamingo, engine.ModelLyrics, engine.ModelDescription}
		if len(got) != len(expected) {
			t.Fatalf("expected %d models, got %d (%#v)", len(expected), len(got), got)
		}
		for i, model := range expected {
			if got[i] != model {
				t.Fatalf("expected model %q at index %d, got %q", model, i, got[i])
			}
		}
	})

	t.Run("uses fallback when raw models are unsupported", func(t *testing.T) {
		got := normalizeRecommendationModels([]string{"unsupported-model"}, []string{"flamingo"})
		if len(got) != 1 || got[0] != engine.ModelFlamingo {
			t.Fatalf("expected fallback [%s], got %#v", engine.ModelFlamingo, got)
		}
	})
}

func TestNormalizeModelPriorities(t *testing.T) {
	got := normalizeModelPriorities(map[string]int{
		"qwen3":    5,
		"lyric":    2,
		"flamingo": 9,
	})

	if got[engine.ModelLyrics] != 2 {
		t.Fatalf("expected lyrics priority 2, got %d", got[engine.ModelLyrics])
	}
	if got[engine.ModelDescription] != 5 {
		t.Fatalf("expected description priority 5, got %d", got[engine.ModelDescription])
	}
	if got[engine.ModelFlamingo] != 9 {
		t.Fatalf("expected flamingo priority 9, got %d", got[engine.ModelFlamingo])
	}
}

func TestNormalizeTextTargets(t *testing.T) {
	t.Run("defaults to lyrics and description", func(t *testing.T) {
		got := normalizeTextTargets(nil, "")
		if len(got) != 2 || got[0] != engine.ModelLyrics || got[1] != engine.ModelDescription {
			t.Fatalf("expected [%s %s], got %#v", engine.ModelLyrics, engine.ModelDescription, got)
		}
	})

	t.Run("legacy model qwen3 expands to lyrics and description", func(t *testing.T) {
		got := normalizeTextTargets(nil, "qwen3")
		if len(got) != 2 || got[0] != engine.ModelLyrics || got[1] != engine.ModelDescription {
			t.Fatalf("expected [%s %s], got %#v", engine.ModelLyrics, engine.ModelDescription, got)
		}
	})

	t.Run("filters out non-text models", func(t *testing.T) {
		got := normalizeTextTargets([]string{"flamingo", "lyrics"}, "")
		if len(got) != 1 || got[0] != engine.ModelLyrics {
			t.Fatalf("expected [%s], got %#v", engine.ModelLyrics, got)
		}
	})
}

func TestEnsureTextRecommendationModels(t *testing.T) {
	t.Run("adds text targets for text recommendation endpoint", func(t *testing.T) {
		got := ensureTextRecommendationModels([]string{engine.ModelFlamingo}, []string{engine.ModelLyrics, engine.ModelDescription})
		expected := []string{engine.ModelFlamingo, engine.ModelLyrics, engine.ModelDescription}
		if len(got) != len(expected) {
			t.Fatalf("expected %d models, got %d (%#v)", len(expected), len(got), got)
		}
		for i, model := range expected {
			if got[i] != model {
				t.Fatalf("expected model %q at index %d, got %q", model, i, got[i])
			}
		}
	})

	t.Run("uses text defaults when all provided models are unsupported", func(t *testing.T) {
		got := ensureTextRecommendationModels([]string{"unsupported"}, nil)
		if len(got) != 2 || got[0] != engine.ModelLyrics || got[1] != engine.ModelDescription {
			t.Fatalf("expected [%s %s], got %#v", engine.ModelLyrics, engine.ModelDescription, got)
		}
	})
}
