package nativeapi

import (
	"testing"

	"github.com/navidrome/navidrome/recommender/engine"
)

func TestCanonicalModelsFromValue(t *testing.T) {
	tests := []struct {
		name  string
		value string
		want  []string
	}{
		{name: "empty ignored", value: "", want: nil},
		{name: "none ignored", value: "none", want: nil},
		{name: "flamingo canonical", value: "flamingo", want: []string{engine.ModelFlamingo}},
		{name: "audio alias", value: "audio", want: []string{engine.ModelFlamingo}},
		{name: "music_flamingo alias", value: "music_flamingo", want: []string{engine.ModelFlamingo}},
		{name: "music-flamingo alias", value: "music-flamingo", want: []string{engine.ModelFlamingo}},
		{name: "lyrics canonical", value: "lyrics", want: []string{engine.ModelLyrics}},
		{name: "lyric alias", value: "lyric", want: []string{engine.ModelLyrics}},
		{name: "description canonical", value: "description", want: []string{engine.ModelDescription}},
		{name: "desc alias", value: "desc", want: []string{engine.ModelDescription}},
		{name: "qwen3 expands", value: "qwen3", want: []string{engine.ModelLyrics, engine.ModelDescription}},
		{name: "qwen8b expands", value: "qwen8b", want: []string{engine.ModelLyrics, engine.ModelDescription}},
		{name: "unknown ignored", value: "unknown", want: nil},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assertStringSliceEqual(t, canonicalModelsFromValue(tt.value), tt.want)
		})
	}
}

func TestNormalizeModelList(t *testing.T) {
	got := normalizeModelList([]string{"audio", "lyrics", "lyric", "qwen3", "desc", "unknown", "flamingo"})
	want := []string{engine.ModelFlamingo, engine.ModelLyrics, engine.ModelDescription}
	assertStringSliceEqual(t, got, want)
}

func TestNormalizeRecommendationModels(t *testing.T) {
	t.Run("defaults to flamingo", func(t *testing.T) {
		got := normalizeRecommendationModels(nil, nil)
		assertStringSliceEqual(t, got, []string{engine.ModelFlamingo})
	})

	t.Run("normalizes aliases and qwen3 expansion", func(t *testing.T) {
		got := normalizeRecommendationModels([]string{"audio", "qwen3", "lyric", "desc"}, nil)
		assertStringSliceEqual(t, got, []string{engine.ModelFlamingo, engine.ModelLyrics, engine.ModelDescription})
	})

	t.Run("uses fallback when raw models are unsupported", func(t *testing.T) {
		got := normalizeRecommendationModels([]string{"unsupported-model"}, []string{"flamingo"})
		assertStringSliceEqual(t, got, []string{engine.ModelFlamingo})
	})

	t.Run("uses normalized fallback list when models omitted", func(t *testing.T) {
		got := normalizeRecommendationModels(nil, []string{"qwen3", "audio"})
		assertStringSliceEqual(t, got, []string{engine.ModelLyrics, engine.ModelDescription, engine.ModelFlamingo})
	})
}

func TestNormalizeModelPriorities(t *testing.T) {
	t.Run("expands aliases and keeps highest priority (lowest value)", func(t *testing.T) {
		got := normalizeModelPriorities(map[string]int{
			"qwen3":    5,
			"lyric":    2,
			"lyrics":   4,
			"desc":     7,
			"flamingo": 9,
			"audio":    6,
			"unknown":  1,
		})
		if got[engine.ModelLyrics] != 2 {
			t.Fatalf("expected lyrics priority 2, got %d", got[engine.ModelLyrics])
		}
		if got[engine.ModelDescription] != 5 {
			t.Fatalf("expected description priority 5, got %d", got[engine.ModelDescription])
		}
		if got[engine.ModelFlamingo] != 6 {
			t.Fatalf("expected flamingo priority 6, got %d", got[engine.ModelFlamingo])
		}
	})

	t.Run("returns nil when no known priorities", func(t *testing.T) {
		got := normalizeModelPriorities(map[string]int{"unknown": 1})
		if got != nil {
			t.Fatalf("expected nil, got %#v", got)
		}
	})
}

func TestNormalizeNegativeEmbeddings(t *testing.T) {
	t.Run("expands aliases and skips unknown models", func(t *testing.T) {
		got := normalizeNegativeEmbeddings(map[string][][]float64{
			"qwen3": {
				{1, 2},
				{3, 4},
			},
			"lyrics": {
				{5, 6},
			},
			"desc": {
				{7, 8},
			},
			"unknown": {
				{9, 10},
			},
		})

		if len(got[engine.ModelLyrics]) != 3 {
			t.Fatalf("expected 3 lyric vectors, got %d", len(got[engine.ModelLyrics]))
		}
		if len(got[engine.ModelDescription]) != 3 {
			t.Fatalf("expected 3 description vectors, got %d", len(got[engine.ModelDescription]))
		}
		if _, ok := got["unknown"]; ok {
			t.Fatalf("did not expect unknown model embeddings")
		}
	})

	t.Run("returns nil when inputs are empty or unsupported", func(t *testing.T) {
		if got := normalizeNegativeEmbeddings(nil); got != nil {
			t.Fatalf("expected nil for nil input, got %#v", got)
		}
		if got := normalizeNegativeEmbeddings(map[string][][]float64{"unknown": {{1}}}); got != nil {
			t.Fatalf("expected nil for unsupported input, got %#v", got)
		}
	})
}

func TestNormalizeTextTargets(t *testing.T) {
	t.Run("defaults to lyrics and description", func(t *testing.T) {
		assertStringSliceEqual(t, normalizeTextTargets(nil, ""), []string{engine.ModelLyrics, engine.ModelDescription})
	})

	t.Run("legacy model qwen3 expands to lyrics and description", func(t *testing.T) {
		assertStringSliceEqual(t, normalizeTextTargets(nil, "qwen3"), []string{engine.ModelLyrics, engine.ModelDescription})
	})

	t.Run("legacy audio model falls back to default text targets", func(t *testing.T) {
		assertStringSliceEqual(t, normalizeTextTargets(nil, "audio"), []string{engine.ModelLyrics, engine.ModelDescription})
	})

	t.Run("filters out non-text models", func(t *testing.T) {
		assertStringSliceEqual(t, normalizeTextTargets([]string{"flamingo", "lyrics", "description"}, ""), []string{engine.ModelLyrics, engine.ModelDescription})
	})

	t.Run("unsupported targets fall back to defaults", func(t *testing.T) {
		assertStringSliceEqual(t, normalizeTextTargets([]string{"unknown"}, ""), []string{engine.ModelLyrics, engine.ModelDescription})
	})
}

func TestEnsureTextRecommendationModels(t *testing.T) {
	t.Run("adds text targets for text recommendation endpoint", func(t *testing.T) {
		got := ensureTextRecommendationModels([]string{engine.ModelFlamingo}, []string{engine.ModelLyrics, engine.ModelDescription})
		assertStringSliceEqual(t, got, []string{engine.ModelFlamingo, engine.ModelLyrics, engine.ModelDescription})
	})

	t.Run("deduplicates repeated text models", func(t *testing.T) {
		got := ensureTextRecommendationModels([]string{engine.ModelLyrics}, []string{engine.ModelLyrics, engine.ModelDescription})
		assertStringSliceEqual(t, got, []string{engine.ModelLyrics, engine.ModelDescription})
	})

	t.Run("uses text defaults when all provided models are unsupported", func(t *testing.T) {
		got := ensureTextRecommendationModels([]string{"unsupported"}, nil)
		assertStringSliceEqual(t, got, []string{engine.ModelLyrics, engine.ModelDescription})
	})
}

func assertStringSliceEqual(t *testing.T, got []string, want []string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("expected length %d, got %d (%#v)", len(want), len(got), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("expected %q at index %d, got %q", want[i], i, got[i])
		}
	}
}
