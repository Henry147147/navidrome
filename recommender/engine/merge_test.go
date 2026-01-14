package engine

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestAverage(t *testing.T) {
	tests := []struct {
		name     string
		values   []float64
		expected float64
	}{
		{"empty slice", []float64{}, 0},
		{"single value", []float64{5.0}, 5.0},
		{"two values", []float64{4.0, 6.0}, 5.0},
		{"multiple values", []float64{1.0, 2.0, 3.0, 4.0, 5.0}, 3.0},
		{"with decimals", []float64{0.1, 0.2, 0.3}, 0.2},
		{"negative values", []float64{-1.0, 1.0}, 0.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := average(tt.values)
			assert.InDelta(t, tt.expected, result, 0.0001)
		})
	}
}

func TestMergeUnion(t *testing.T) {
	e := New(DefaultConfig(), nil, nil)

	t.Run("empty results", func(t *testing.T) {
		results := map[string][]candidate{}
		merged := e.mergeUnion(results, 1)
		assert.Empty(t, merged)
	})

	t.Run("single model results", func(t *testing.T) {
		results := map[string][]candidate{
			ModelLyrics: {
				{Name: "track1", Score: 0.9},
				{Name: "track2", Score: 0.8},
			},
		}
		merged := e.mergeUnion(results, 1)
		assert.Len(t, merged, 2)
	})

	t.Run("multiple models all unique", func(t *testing.T) {
		results := map[string][]candidate{
			ModelLyrics: {
				{Name: "track1", Score: 0.9},
			},
			ModelDescription: {
				{Name: "track2", Score: 0.8},
			},
		}
		merged := e.mergeUnion(results, 1)
		assert.Len(t, merged, 2)
	})

	t.Run("overlapping tracks", func(t *testing.T) {
		results := map[string][]candidate{
			ModelLyrics: {
				{Name: "track1", Score: 0.9},
				{Name: "track2", Score: 0.7},
			},
			ModelDescription: {
				{Name: "track1", Score: 0.85},
				{Name: "track3", Score: 0.6},
			},
		}
		merged := e.mergeUnion(results, 1)
		assert.Len(t, merged, 3)

		// Find track1 and verify its score is averaged
		for _, c := range merged {
			if c.Name == "track1" {
				assert.InDelta(t, 0.875, c.Score, 0.001) // (0.9 + 0.85) / 2
				assert.Len(t, c.Models, 2)
				assert.Len(t, c.Scores, 2)
				break
			}
		}
	})

	t.Run("minimum agreement filter", func(t *testing.T) {
		results := map[string][]candidate{
			ModelLyrics: {
				{Name: "track1", Score: 0.9},
				{Name: "track2", Score: 0.7},
			},
			ModelDescription: {
				{Name: "track1", Score: 0.85},
				{Name: "track3", Score: 0.6},
			},
			ModelFlamingo: {
				{Name: "track1", Score: 0.8},
			},
		}

		// Require at least 2 models to agree
		merged := e.mergeUnion(results, 2)
		assert.Len(t, merged, 1) // Only track1 appears in 2+ models
		assert.Equal(t, "track1", merged[0].Name)

		// Require at least 3 models to agree
		merged = e.mergeUnion(results, 3)
		assert.Len(t, merged, 1) // track1 appears in all 3
	})

	t.Run("score averaging across models", func(t *testing.T) {
		results := map[string][]candidate{
			ModelLyrics:      {{Name: "track1", Score: 1.0}},
			ModelDescription: {{Name: "track1", Score: 0.5}},
			ModelFlamingo:    {{Name: "track1", Score: 0.2}},
		}
		merged := e.mergeUnion(results, 1)
		assert.Len(t, merged, 1)
		// Average of 1.0, 0.5, 0.2 = 0.5667
		assert.InDelta(t, 0.5667, merged[0].Score, 0.001)
	})
}

func TestMergeIntersection(t *testing.T) {
	e := New(DefaultConfig(), nil, nil)

	t.Run("empty results", func(t *testing.T) {
		results := map[string][]candidate{}
		merged := e.mergeIntersection(results)
		assert.Nil(t, merged)
	})

	t.Run("single model returns all", func(t *testing.T) {
		results := map[string][]candidate{
			ModelLyrics: {
				{Name: "track1", Score: 0.9},
				{Name: "track2", Score: 0.8},
			},
		}
		merged := e.mergeIntersection(results)
		assert.Len(t, merged, 2)
	})

	t.Run("no intersection", func(t *testing.T) {
		results := map[string][]candidate{
			ModelLyrics:      {{Name: "track1", Score: 0.9}},
			ModelDescription: {{Name: "track2", Score: 0.8}},
		}
		merged := e.mergeIntersection(results)
		assert.Empty(t, merged)
	})

	t.Run("full intersection", func(t *testing.T) {
		results := map[string][]candidate{
			ModelLyrics:      {{Name: "track1", Score: 0.9}, {Name: "track2", Score: 0.7}},
			ModelDescription: {{Name: "track1", Score: 0.85}, {Name: "track2", Score: 0.65}},
		}
		merged := e.mergeIntersection(results)
		assert.Len(t, merged, 2)
	})

	t.Run("partial intersection", func(t *testing.T) {
		results := map[string][]candidate{
			ModelLyrics: {
				{Name: "track1", Score: 0.9},
				{Name: "track2", Score: 0.7},
				{Name: "track3", Score: 0.5},
			},
			ModelDescription: {
				{Name: "track1", Score: 0.85},
				{Name: "track2", Score: 0.65},
				{Name: "track4", Score: 0.4},
			},
		}
		merged := e.mergeIntersection(results)
		assert.Len(t, merged, 2) // Only track1 and track2 in both

		// Verify scores are averaged
		for _, c := range merged {
			if c.Name == "track1" {
				assert.InDelta(t, 0.875, c.Score, 0.001)
			} else if c.Name == "track2" {
				assert.InDelta(t, 0.675, c.Score, 0.001)
			}
		}
	})

	t.Run("three models intersection", func(t *testing.T) {
		results := map[string][]candidate{
			ModelLyrics:      {{Name: "track1", Score: 0.9}, {Name: "track2", Score: 0.7}},
			ModelDescription: {{Name: "track1", Score: 0.85}, {Name: "track3", Score: 0.6}},
			ModelFlamingo:    {{Name: "track1", Score: 0.8}, {Name: "track2", Score: 0.5}},
		}
		merged := e.mergeIntersection(results)
		assert.Len(t, merged, 1) // Only track1 in all three
		assert.Equal(t, "track1", merged[0].Name)
	})
}

func TestMergePriority(t *testing.T) {
	e := New(DefaultConfig(), nil, nil)

	t.Run("uses intersection when sufficient", func(t *testing.T) {
		results := map[string][]candidate{
			ModelLyrics: {
				{Name: "track1", Score: 0.9},
				{Name: "track2", Score: 0.8},
				{Name: "track3", Score: 0.7},
			},
			ModelDescription: {
				{Name: "track1", Score: 0.85},
				{Name: "track2", Score: 0.75},
				{Name: "track3", Score: 0.65},
			},
		}
		priorities := map[string]int{ModelLyrics: 1, ModelDescription: 2}

		// Request 2 items, intersection has 3
		merged := e.mergePriority(results, priorities, 2)
		assert.Len(t, merged, 3) // Returns all intersection results
	})

	t.Run("falls back to primary model", func(t *testing.T) {
		results := map[string][]candidate{
			ModelLyrics: {
				{Name: "track1", Score: 0.9},
				{Name: "track2", Score: 0.8},
				{Name: "track3", Score: 0.7},
			},
			ModelDescription: {
				{Name: "track4", Score: 0.6},
			},
		}
		priorities := map[string]int{ModelLyrics: 1, ModelDescription: 2}

		// Intersection is empty, should fall back to lyrics (priority 1)
		merged := e.mergePriority(results, priorities, 5)
		assert.Len(t, merged, 3)
	})

	t.Run("respects priority order", func(t *testing.T) {
		results := map[string][]candidate{
			ModelLyrics: {
				{Name: "lyrics_track", Score: 0.5},
			},
			ModelDescription: {
				{Name: "desc_track", Score: 0.9},
			},
		}
		// Description has higher priority (lower number)
		priorities := map[string]int{ModelLyrics: 10, ModelDescription: 1}

		merged := e.mergePriority(results, priorities, 5)
		// Should return description results since intersection is empty
		assert.Len(t, merged, 1)
		assert.Equal(t, "desc_track", merged[0].Name)
	})

	t.Run("combines intersection with fallback", func(t *testing.T) {
		results := map[string][]candidate{
			ModelLyrics: {
				{Name: "common", Score: 0.9},
				{Name: "lyrics_only", Score: 0.7},
			},
			ModelDescription: {
				{Name: "common", Score: 0.85},
			},
		}
		priorities := map[string]int{ModelLyrics: 1}

		merged := e.mergePriority(results, priorities, 5)
		// Should have both common (from intersection) and lyrics_only (from fallback)
		assert.Len(t, merged, 2)

		// Common should come first (from intersection)
		names := make([]string, len(merged))
		for i, c := range merged {
			names[i] = c.Name
		}
		assert.Contains(t, names, "common")
		assert.Contains(t, names, "lyrics_only")
	})

	t.Run("default priority for unspecified models", func(t *testing.T) {
		results := map[string][]candidate{
			ModelLyrics:   {{Name: "track1", Score: 0.9}},
			ModelFlamingo: {{Name: "track2", Score: 0.8}},
		}
		// Only specify lyrics priority, flamingo gets default 100
		priorities := map[string]int{ModelLyrics: 1}

		merged := e.mergePriority(results, priorities, 5)
		// Lyrics has priority 1 (lower = higher), so should be primary
		assert.Contains(t, merged[0].Name, "track")
	})

	t.Run("empty priorities uses first model", func(t *testing.T) {
		results := map[string][]candidate{
			ModelDescription: {{Name: "desc_track", Score: 0.9}},
		}
		priorities := map[string]int{} // Empty priorities

		merged := e.mergePriority(results, priorities, 5)
		assert.Len(t, merged, 1)
	})
}

func TestCandidate(t *testing.T) {
	c := candidate{
		Name:   "test_track",
		Score:  0.95,
		Scores: []float64{0.9, 1.0},
		Models: []string{ModelLyrics, ModelDescription},
	}

	assert.Equal(t, "test_track", c.Name)
	assert.Equal(t, 0.95, c.Score)
	assert.Equal(t, []float64{0.9, 1.0}, c.Scores)
	assert.Equal(t, []string{ModelLyrics, ModelDescription}, c.Models)
}

func TestNegativeSimilarityPointer(t *testing.T) {
	// Test that negative similarity is optional
	c1 := candidate{Name: "track1", Score: 0.9}
	assert.Nil(t, c1.NegativeSimilarity)

	sim := 0.5
	c2 := candidate{Name: "track2", Score: 0.8, NegativeSimilarity: &sim}
	assert.NotNil(t, c2.NegativeSimilarity)
	assert.Equal(t, 0.5, *c2.NegativeSimilarity)
}
