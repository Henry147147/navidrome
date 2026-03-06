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
					ModelMuQMulan: {0.1, 0.2, 0.3},
				},
			},
		},
		Models: []string{ModelMuQMulan, ModelMuQAudio},
	}

	got, warnings, err := e.resolveSeedEmbeddings(context.Background(), req)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(warnings) != 0 {
		t.Fatalf("expected no warnings, got %#v", warnings)
	}
	if len(got[ModelMuQMulan]) != 1 {
		t.Fatalf("expected shared embedding to be available, got %#v", got[ModelMuQMulan])
	}
	if len(got[ModelMuQAudio]) != 0 {
		t.Fatalf("expected no MuQ audio embedding for text seed, got %#v", got[ModelMuQAudio])
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
		Models: []string{ModelMuQAudio, ModelMuQMulan},
	}

	got, warnings, err := e.resolveSeedEmbeddings(context.Background(), req)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(warnings) != 0 {
		t.Fatalf("expected no warnings, got %#v", warnings)
	}
	if len(got[ModelMuQAudio]) != 1 {
		t.Fatalf("expected legacy direct embedding to use primary model, got %#v", got[ModelMuQAudio])
	}
	if len(got[ModelMuQMulan]) != 0 {
		t.Fatalf("expected no MuQ-MuLan embedding, got %#v", got[ModelMuQMulan])
	}
}

func TestResolveSeedEmbeddingsSkipsUnknownAndEmptyPerModelEntries(t *testing.T) {
	e := New(DefaultConfig(), nil, nil)
	req := RecommendationRequest{
		Seeds: []SeedTrack{
			{
				TrackID: "text-seed",
				Embeddings: map[string][]float64{
					ModelMuQMulan: {0.1, 0.2},
					"unknown":     {0.3, 0.4},
					ModelMuQAudio: {},
				},
			},
		},
		Models: []string{ModelMuQMulan, ModelMuQAudio},
	}

	got, warnings, err := e.resolveSeedEmbeddings(context.Background(), req)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(warnings) != 0 {
		t.Fatalf("expected no warnings, got %#v", warnings)
	}
	if len(got[ModelMuQMulan]) != 1 {
		t.Fatalf("expected one MuQ-MuLan embedding, got %#v", got[ModelMuQMulan])
	}
	if len(got[ModelMuQAudio]) != 0 {
		t.Fatalf("expected no MuQ audio embedding, got %#v", got[ModelMuQAudio])
	}
}

func TestResolveSeedEmbeddingsPrefersPerModelMapOverLegacyEmbedding(t *testing.T) {
	e := New(DefaultConfig(), nil, nil)
	req := RecommendationRequest{
		Seeds: []SeedTrack{
			{
				TrackID:   "text-seed",
				Embedding: []float64{9.9, 9.9},
				Embeddings: map[string][]float64{
					ModelMuQMulan: {0.1, 0.2},
				},
			},
		},
		Models: []string{ModelMuQMulan, ModelMuQAudio},
	}

	got, warnings, err := e.resolveSeedEmbeddings(context.Background(), req)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(warnings) != 0 {
		t.Fatalf("expected no warnings, got %#v", warnings)
	}
	if len(got[ModelMuQMulan]) != 1 {
		t.Fatalf("expected one MuQ-MuLan embedding, got %#v", got[ModelMuQMulan])
	}
	if len(got[ModelMuQAudio]) != 0 {
		t.Fatalf("expected no MuQ audio embedding, got %#v", got[ModelMuQAudio])
	}
	if got[ModelMuQMulan][0].Key != "text-seed" {
		t.Fatalf("expected resolved seed key text-seed, got %#v", got[ModelMuQMulan])
	}
}

func TestResolveSeedEmbeddingsWarnsWhenNoEmbeddingsAvailable(t *testing.T) {
	e := New(DefaultConfig(), nil, nil)
	req := RecommendationRequest{
		Seeds: []SeedTrack{
			{TrackID: ""},
		},
		Models: []string{ModelMuQMulan},
	}

	got, warnings, err := e.resolveSeedEmbeddings(context.Background(), req)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(got[ModelMuQMulan]) != 0 {
		t.Fatalf("expected no embeddings, got %#v", got[ModelMuQMulan])
	}
	if len(warnings) != 1 || warnings[0] != "No embeddings found for any seeds" {
		t.Fatalf("expected no-embeddings warning, got %#v", warnings)
	}
}
