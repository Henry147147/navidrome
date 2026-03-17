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
		{name: "flamingo alias", value: "flamingo", want: []string{engine.ModelMuQAudio}},
		{name: "audio alias", value: "audio", want: []string{engine.ModelMuQAudio}},
		{name: "music_flamingo alias", value: "music_flamingo", want: []string{engine.ModelMuQAudio}},
		{name: "music-flamingo alias", value: "music-flamingo", want: []string{engine.ModelMuQAudio}},
		{name: "lyrics alias", value: "lyrics", want: []string{engine.ModelMuQMulan}},
		{name: "lyric alias", value: "lyric", want: []string{engine.ModelMuQMulan}},
		{name: "description alias", value: "description", want: []string{engine.ModelMuQMulan}},
		{name: "desc alias", value: "desc", want: []string{engine.ModelMuQMulan}},
		{name: "qwen3 alias", value: "qwen3", want: []string{engine.ModelMuQMulan}},
		{name: "qwen8b alias", value: "qwen8b", want: []string{engine.ModelMuQMulan}},
		{name: "canonical muq audio", value: engine.ModelMuQAudio, want: []string{engine.ModelMuQAudio}},
		{name: "canonical muq mulan", value: engine.ModelMuQMulan, want: []string{engine.ModelMuQMulan}},
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
	want := []string{engine.ModelMuQAudio, engine.ModelMuQMulan}
	assertStringSliceEqual(t, got, want)
}

func TestNormalizeRecommendationModels(t *testing.T) {
	t.Run("defaults to muq audio", func(t *testing.T) {
		got := normalizeRecommendationModels(nil, nil)
		assertStringSliceEqual(t, got, []string{engine.ModelMuQAudio})
	})

	t.Run("normalizes aliases to canonical models", func(t *testing.T) {
		got := normalizeRecommendationModels([]string{"audio", "qwen3", "lyric", "desc"}, nil)
		assertStringSliceEqual(t, got, []string{engine.ModelMuQAudio, engine.ModelMuQMulan})
	})

	t.Run("uses fallback when raw models are unsupported", func(t *testing.T) {
		got := normalizeRecommendationModels([]string{"unsupported-model"}, []string{"flamingo"})
		assertStringSliceEqual(t, got, []string{engine.ModelMuQAudio})
	})

	t.Run("uses normalized fallback list when models omitted", func(t *testing.T) {
		got := normalizeRecommendationModels(nil, []string{"qwen3", "audio"})
		assertStringSliceEqual(t, got, []string{engine.ModelMuQMulan, engine.ModelMuQAudio})
	})
}

func TestNormalizeModelPriorities(t *testing.T) {
	t.Run("canonicalizes aliases and keeps highest priority", func(t *testing.T) {
		got := normalizeModelPriorities(map[string]int{
			"qwen3":    5,
			"lyric":    2,
			"lyrics":   4,
			"desc":     7,
			"flamingo": 9,
			"audio":    6,
			"unknown":  1,
		})
		if got[engine.ModelMuQMulan] != 2 {
			t.Fatalf("expected muq mulan priority 2, got %d", got[engine.ModelMuQMulan])
		}
		if got[engine.ModelMuQAudio] != 6 {
			t.Fatalf("expected muq audio priority 6, got %d", got[engine.ModelMuQAudio])
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
	t.Run("collapses aliases into canonical embedding buckets", func(t *testing.T) {
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

		if len(got[engine.ModelMuQMulan]) != 4 {
			t.Fatalf("expected 4 shared vectors, got %d", len(got[engine.ModelMuQMulan]))
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
	t.Run("defaults to muq mulan", func(t *testing.T) {
		assertStringSliceEqual(t, normalizeTextTargets(nil, ""), []string{engine.ModelMuQMulan})
	})

	t.Run("legacy text models normalize to muq mulan", func(t *testing.T) {
		assertStringSliceEqual(t, normalizeTextTargets(nil, "qwen3"), []string{engine.ModelMuQMulan})
	})

	t.Run("legacy audio model still uses muq mulan for text queries", func(t *testing.T) {
		assertStringSliceEqual(t, normalizeTextTargets(nil, "audio"), []string{engine.ModelMuQMulan})
	})

	t.Run("text targets are accepted but ignored for compatibility", func(t *testing.T) {
		assertStringSliceEqual(t, normalizeTextTargets([]string{"flamingo", "lyrics", "description"}, ""), []string{engine.ModelMuQMulan})
	})

	t.Run("unsupported targets fall back to defaults", func(t *testing.T) {
		assertStringSliceEqual(t, normalizeTextTargets([]string{"unknown"}, ""), []string{engine.ModelMuQMulan})
	})
}

func TestEnsureTextRecommendationModels(t *testing.T) {
	t.Run("adds muq mulan when only audio models are present", func(t *testing.T) {
		got := ensureTextRecommendationModels([]string{engine.ModelMuQAudio}, []string{engine.ModelMuQMulan})
		assertStringSliceEqual(t, got, []string{engine.ModelMuQAudio, engine.ModelMuQMulan})
	})

	t.Run("keeps canonical shared model once", func(t *testing.T) {
		got := ensureTextRecommendationModels([]string{engine.ModelMuQMulan}, []string{engine.ModelMuQMulan})
		assertStringSliceEqual(t, got, []string{engine.ModelMuQMulan})
	})

	t.Run("uses muq mulan when all provided models are unsupported", func(t *testing.T) {
		got := ensureTextRecommendationModels([]string{"unsupported"}, nil)
		assertStringSliceEqual(t, got, []string{engine.ModelMuQMulan})
	})
}

func TestNormalizeTextEmbedderModel(t *testing.T) {
	tests := []struct {
		name  string
		value string
	}{
		{name: "empty", value: ""},
		{name: "canonical", value: engine.ModelMuQMulan},
		{name: "lyrics alias", value: "lyrics"},
		{name: "description alias", value: "description"},
		{name: "qwen alias", value: "qwen8b"},
		{name: "unsupported", value: "something-else"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := normalizeTextEmbedderModel(tt.value); got != engine.ModelMuQMulan {
				t.Fatalf("expected %q, got %q", engine.ModelMuQMulan, got)
			}
		})
	}
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
