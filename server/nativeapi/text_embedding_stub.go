package nativeapi

import (
	"hash/fnv"
	"math"
	"strings"

	"github.com/navidrome/navidrome/conf"
	"github.com/navidrome/navidrome/recommender/engine"
)

const (
	defaultTextEmbeddingDim = 512
)

func embeddingDimensionForModel(model string) int {
	switch model {
	case engine.ModelMuQAudio:
		if dim := conf.Server.Recommendations.Milvus.Dimensions.ResolvedMuQAudio(); dim > 0 {
			return dim
		}
	case engine.ModelMuQMulan:
		if dim := conf.Server.Recommendations.Milvus.Dimensions.ResolvedMuQMulan(); dim > 0 {
			return dim
		}
	}
	return defaultTextEmbeddingDim
}

func deterministicTextEmbedding(text string, model string, target string, dim int) []float64 {
	if dim <= 0 {
		dim = defaultTextEmbeddingDim
	}

	seed := hashToSeed(
		strings.TrimSpace(text),
		strings.ToLower(strings.TrimSpace(model)),
		strings.ToLower(strings.TrimSpace(target)),
	)
	state := uint64(seed)
	vector := make([]float64, dim)

	var norm float64
	for i := range vector {
		value := splitmixSignedUnit(&state)
		vector[i] = value
		norm += value * value
	}

	if norm == 0 {
		vector[0] = 1
		return vector
	}

	invNorm := 1.0 / math.Sqrt(norm)
	for i := range vector {
		vector[i] *= invNorm
	}

	return vector
}

func splitmixSignedUnit(state *uint64) float64 {
	*state += 0x9e3779b97f4a7c15
	z := *state
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	z ^= z >> 31

	// Convert to a deterministic float64 in [0, 1), then map to [-1, 1).
	const inv53 = 1.0 / (1 << 53)
	unit := float64(z>>11) * inv53
	return unit*2.0 - 1.0
}

func hashToSeed(parts ...string) int64 {
	h := fnv.New64a()
	for _, part := range parts {
		_, _ = h.Write([]byte(part))
		_, _ = h.Write([]byte{0x00})
	}
	return int64(h.Sum64())
}
