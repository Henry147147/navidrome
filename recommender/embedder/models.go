package embedder

// Model identifiers stored alongside embeddings.
const (
	ModelLyrics      = "lyrics"
	ModelDescription = "description"
	ModelFlamingo    = "flamingo"
)

func float32sToFloat64s(values []float32) []float64 {
	if len(values) == 0 {
		return nil
	}
	out := make([]float64, len(values))
	for i, v := range values {
		out[i] = float64(v)
	}
	return out
}
