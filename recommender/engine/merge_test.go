package engine

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestReciprocalRankContribution(t *testing.T) {
	assert.InDelta(t, 1.0/61.0, reciprocalRankContribution(1, 1), 0.000001)
	assert.InDelta(t, 0.5/62.0, reciprocalRankContribution(2, 0.5), 0.000001)
	assert.Zero(t, reciprocalRankContribution(0, 1))
}

func TestMergeUnionUsesModelLevelRRF(t *testing.T) {
	e := New(DefaultConfig(), nil, nil)
	results := map[string][]candidate{
		ModelLyrics: {
			rankedCandidate("track-consensus", ModelLyrics, 2, 0.30),
			rankedCandidate("track-lyrics-only", ModelLyrics, 1, 0.95),
		},
		ModelDescription: {
			rankedCandidate("track-consensus", ModelDescription, 1, 0.20),
			rankedCandidate("track-description-only", ModelDescription, 2, 0.99),
		},
	}

	merged := e.mergeUnion(results, []string{ModelLyrics, ModelDescription}, 1)

	if assert.Len(t, merged, 3) {
		assert.Equal(t, "track-consensus", merged[0].Name)
		assert.ElementsMatch(t, []string{ModelLyrics, ModelDescription}, merged[0].Models)
		assert.InDelta(
			t,
			reciprocalRankContribution(2, 1)+reciprocalRankContribution(1, 1),
			merged[0].Score,
			0.000001,
		)
	}
}

func TestMergeIntersectionRequiresAllActiveModels(t *testing.T) {
	e := New(DefaultConfig(), nil, nil)
	results := map[string][]candidate{
		ModelLyrics: {
			rankedCandidate("track-all", ModelLyrics, 1, 0.9),
			rankedCandidate("lyrics-only", ModelLyrics, 2, 0.8),
		},
		ModelDescription: {
			rankedCandidate("track-all", ModelDescription, 2, 0.7),
			rankedCandidate("description-only", ModelDescription, 1, 0.95),
		},
		ModelFlamingo: {
			rankedCandidate("track-all", ModelFlamingo, 3, 0.6),
		},
	}

	merged := e.mergeIntersection(results, []string{ModelLyrics, ModelDescription, ModelFlamingo})

	if assert.Len(t, merged, 1) {
		assert.Equal(t, "track-all", merged[0].Name)
		assert.ElementsMatch(t, []string{ModelLyrics, ModelDescription, ModelFlamingo}, merged[0].Models)
	}
}

func TestMergePriorityRequiresPrimaryThenBackfills(t *testing.T) {
	e := New(DefaultConfig(), nil, nil)
	results := map[string][]candidate{
		ModelLyrics: {
			rankedCandidate("lyrics-primary", ModelLyrics, 1, 0.8),
			rankedCandidate("shared", ModelLyrics, 2, 0.7),
		},
		ModelDescription: {
			rankedCandidate("shared", ModelDescription, 1, 0.9),
			rankedCandidate("description-only", ModelDescription, 2, 0.95),
		},
	}
	priorities := map[string]int{ModelLyrics: 1, ModelDescription: 5}

	merged := e.mergePriority(results, []string{ModelLyrics, ModelDescription}, priorities, 3, 1)

	if assert.Len(t, merged, 3) {
		assert.Equal(t, "shared", merged[0].Name)
		assert.Equal(t, "lyrics-primary", merged[1].Name)
		assert.Equal(t, "description-only", merged[2].Name)
	}
}

func TestMergeUnionRespectsMinAgreement(t *testing.T) {
	e := New(DefaultConfig(), nil, nil)
	results := map[string][]candidate{
		ModelLyrics: {
			rankedCandidate("track-shared", ModelLyrics, 1, 0.9),
			rankedCandidate("lyrics-only", ModelLyrics, 2, 0.8),
		},
		ModelDescription: {
			rankedCandidate("track-shared", ModelDescription, 1, 0.7),
		},
	}

	merged := e.mergeUnion(results, []string{ModelLyrics, ModelDescription}, 2)

	if assert.Len(t, merged, 1) {
		assert.Equal(t, "track-shared", merged[0].Name)
	}
}

func rankedCandidate(name string, model string, rank int, score float64) candidate {
	return candidate{
		Name:      name,
		Score:     score,
		BaseScore: score,
		Models:    []string{model},
		ModelScores: map[string]float64{
			model: score,
		},
		ModelRanks: map[string]int{
			model: rank,
		},
	}
}
