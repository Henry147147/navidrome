package cmd

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/navidrome/navidrome/model"
	"github.com/navidrome/navidrome/recommender/engine"
	"github.com/navidrome/navidrome/recommender/milvus"
	"github.com/navidrome/navidrome/recommender/resolver"
	"github.com/navidrome/navidrome/server/subsonic"
)

func TestNewRecommendationClientFallsBackWhenMilvusInitFails(t *testing.T) {
	restoreRecommendationFactories(t)
	newMilvusClientForRecommendations = func() (*milvus.Client, func(), error) {
		return nil, nil, errors.New("milvus unavailable")
	}

	client := newRecommendationClient(nil)
	resp, err := client.Recommend(context.Background(), "test", subsonic.RecommendationRequest{})
	if err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	if len(resp.Warnings) == 0 || !strings.Contains(resp.Warnings[0], "disabled") {
		t.Fatalf("expected noop warning, got %#v", resp.Warnings)
	}
}

func TestNewRecommendationClientFallsBackWhenEngineIsNil(t *testing.T) {
	restoreRecommendationFactories(t)
	newMilvusClientForRecommendations = func() (*milvus.Client, func(), error) {
		return nil, func() {}, nil
	}
	newResolverForRecommendations = func(_ model.DataStore) *resolver.Resolver {
		return nil
	}
	newEngineForRecommendations = func(_ *milvus.Client, _ *resolver.Resolver) *engine.Engine {
		return nil
	}

	client := newRecommendationClient(nil)
	resp, err := client.Recommend(context.Background(), "test", subsonic.RecommendationRequest{})
	if err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	if len(resp.Warnings) == 0 || !strings.Contains(resp.Warnings[0], "disabled") {
		t.Fatalf("expected noop warning, got %#v", resp.Warnings)
	}
}

func TestNewRecommendationClientUsesGoEngineWhenDependenciesExist(t *testing.T) {
	restoreRecommendationFactories(t)
	newMilvusClientForRecommendations = func() (*milvus.Client, func(), error) {
		return nil, func() {}, nil
	}
	newResolverForRecommendations = func(_ model.DataStore) *resolver.Resolver {
		return nil
	}
	newEngineForRecommendations = func(_ *milvus.Client, _ *resolver.Resolver) *engine.Engine {
		return engine.New(engine.DefaultConfig(), nil, nil)
	}

	client := newRecommendationClient(nil)
	resp, err := client.Recommend(context.Background(), "test", subsonic.RecommendationRequest{})
	if err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	if len(resp.Warnings) != 1 || resp.Warnings[0] != "No seeds provided" {
		t.Fatalf("expected go engine warning, got %#v", resp.Warnings)
	}
}

func restoreRecommendationFactories(t *testing.T) {
	t.Helper()
	originalMilvus := newMilvusClientForRecommendations
	originalResolver := newResolverForRecommendations
	originalEngine := newEngineForRecommendations
	t.Cleanup(func() {
		newMilvusClientForRecommendations = originalMilvus
		newResolverForRecommendations = originalResolver
		newEngineForRecommendations = originalEngine
	})
}
