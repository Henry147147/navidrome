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
		ModelMuQAudio: {
			rankedCandidate("track-consensus", ModelMuQAudio, 2, 0.30),
			rankedCandidate("track-audio-only", ModelMuQAudio, 1, 0.95),
		},
		ModelMuQMulan: {
			rankedCandidate("track-consensus", ModelMuQMulan, 1, 0.20),
			rankedCandidate("track-shared-only", ModelMuQMulan, 2, 0.99),
		},
	}

	merged := e.mergeUnion(results, []string{ModelMuQAudio, ModelMuQMulan}, 1)

	if assert.Len(t, merged, 3) {
		assert.Equal(t, "track-consensus", merged[0].Name)
		assert.ElementsMatch(t, []string{ModelMuQAudio, ModelMuQMulan}, merged[0].Models)
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
		ModelMuQAudio: {
			rankedCandidate("track-all", ModelMuQAudio, 1, 0.9),
			rankedCandidate("audio-only", ModelMuQAudio, 2, 0.8),
		},
		ModelMuQMulan: {
			rankedCandidate("track-all", ModelMuQMulan, 2, 0.7),
			rankedCandidate("shared-only", ModelMuQMulan, 1, 0.95),
		},
	}

	merged := e.mergeIntersection(results, []string{ModelMuQAudio, ModelMuQMulan})

	if assert.Len(t, merged, 1) {
		assert.Equal(t, "track-all", merged[0].Name)
		assert.ElementsMatch(t, []string{ModelMuQAudio, ModelMuQMulan}, merged[0].Models)
	}
}

func TestMergePriorityRequiresPrimaryThenBackfills(t *testing.T) {
	e := New(DefaultConfig(), nil, nil)
	results := map[string][]candidate{
		ModelMuQAudio: {
			rankedCandidate("audio-primary", ModelMuQAudio, 1, 0.8),
			rankedCandidate("shared", ModelMuQAudio, 2, 0.7),
		},
		ModelMuQMulan: {
			rankedCandidate("shared", ModelMuQMulan, 1, 0.9),
			rankedCandidate("text-only", ModelMuQMulan, 2, 0.95),
		},
	}
	priorities := map[string]int{ModelMuQAudio: 1, ModelMuQMulan: 5}

	merged := e.mergePriority(results, []string{ModelMuQAudio, ModelMuQMulan}, priorities, 3, 1)

	if assert.Len(t, merged, 3) {
		assert.Equal(t, "shared", merged[0].Name)
		assert.Equal(t, "audio-primary", merged[1].Name)
		assert.Equal(t, "text-only", merged[2].Name)
	}
}

func TestMergeUnionRespectsMinAgreement(t *testing.T) {
	e := New(DefaultConfig(), nil, nil)
	results := map[string][]candidate{
		ModelMuQAudio: {
			rankedCandidate("track-shared", ModelMuQAudio, 1, 0.9),
			rankedCandidate("audio-only", ModelMuQAudio, 2, 0.8),
		},
		ModelMuQMulan: {
			rankedCandidate("track-shared", ModelMuQMulan, 1, 0.7),
		},
	}

	merged := e.mergeUnion(results, []string{ModelMuQAudio, ModelMuQMulan}, 2)

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
