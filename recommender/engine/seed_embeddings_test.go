package engine

import (
	"context"
	"testing"
)

func TestResolveSeedEmbeddingsUsesPerModelEmbeddings(t *testing.T) {
	e := New(DefaultConfig(), nil, nil)
	req := RecommendationRequest{
		Seeds: []SeedTrack{
			{
				TrackID: "text-seed",
				Embeddings: map[string][]float64{
					ModelLyrics:      {0.1, 0.2, 0.3},
					ModelDescription: {0.4, 0.5, 0.6},
				},
			},
		},
		Models: []string{ModelLyrics, ModelDescription, ModelFlamingo},
	}

	got, warnings, err := e.resolveSeedEmbeddings(context.Background(), req)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(warnings) != 0 {
		t.Fatalf("expected no warnings, got %#v", warnings)
	}
	if len(got[ModelLyrics]) != 1 {
		t.Fatalf("expected lyrics embedding to be available, got %#v", got[ModelLyrics])
	}
	if len(got[ModelDescription]) != 1 {
		t.Fatalf("expected description embedding to be available, got %#v", got[ModelDescription])
	}
	if len(got[ModelFlamingo]) != 0 {
		t.Fatalf("expected no flamingo embedding for text seed, got %#v", got[ModelFlamingo])
	}
}

func TestResolveSeedEmbeddingsUsesLegacyDirectEmbedding(t *testing.T) {
	e := New(DefaultConfig(), nil, nil)
	req := RecommendationRequest{
		Seeds: []SeedTrack{
			{
				TrackID:   "legacy-seed",
				Embedding: []float64{0.1, 0.2},
			},
		},
		Models: []string{ModelFlamingo, ModelLyrics},
	}

	got, warnings, err := e.resolveSeedEmbeddings(context.Background(), req)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(warnings) != 0 {
		t.Fatalf("expected no warnings, got %#v", warnings)
	}
	if len(got[ModelFlamingo]) != 1 {
		t.Fatalf("expected legacy direct embedding to use primary model, got %#v", got[ModelFlamingo])
	}
	if len(got[ModelLyrics]) != 0 {
		t.Fatalf("expected no lyrics embedding, got %#v", got[ModelLyrics])
	}
}
