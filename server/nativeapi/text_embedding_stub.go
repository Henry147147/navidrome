package nativeapi

import (
	"hash/fnv"
	"math"
	"math/rand"
	"strings"

	"github.com/navidrome/navidrome/conf"
	"github.com/navidrome/navidrome/recommender/engine"
)

const (
	defaultTextEmbeddingDim = 2560
)

func embeddingDimensionForModel(model string) int {
	switch model {
	case engine.ModelLyrics:
		if conf.Server.Recommendations.Milvus.Dimensions.Lyrics > 0 {
			return conf.Server.Recommendations.Milvus.Dimensions.Lyrics
		}
	case engine.ModelDescription:
		if conf.Server.Recommendations.Milvus.Dimensions.Description > 0 {
			return conf.Server.Recommendations.Milvus.Dimensions.Description
		}
	case engine.ModelFlamingo:
		if conf.Server.Recommendations.Milvus.Dimensions.Flamingo > 0 {
			return conf.Server.Recommendations.Milvus.Dimensions.Flamingo
		}
	}
	return defaultTextEmbeddingDim
}

func deterministicTextEmbedding(text string, model string, target string, dim int) []float64 {
	if dim <= 0 {
		dim = defaultTextEmbeddingDim
	}

	seed := hashToSeed(strings.TrimSpace(text), strings.ToLower(strings.TrimSpace(model)), strings.ToLower(strings.TrimSpace(target)))
	rng := rand.New(rand.NewSource(seed))
	vector := make([]float64, dim)

	var norm float64
	for i := range vector {
		value := rng.Float64()*2.0 - 1.0
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

func hashToSeed(parts ...string) int64 {
	h := fnv.New64a()
	for _, part := range parts {
		_, _ = h.Write([]byte(part))
		_, _ = h.Write([]byte{0x00})
	}
	return int64(h.Sum64())
}
