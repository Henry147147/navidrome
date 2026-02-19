package cmd

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/navidrome/navidrome/log"
	"github.com/navidrome/navidrome/model"
	"github.com/navidrome/navidrome/recommender"
	"github.com/navidrome/navidrome/server/subsonic"
)

var (
	newMilvusClientForRecommendations = recommender.NewMilvusClient
	newResolverForRecommendations     = recommender.NewResolver
	newEngineForRecommendations       = recommender.NewRecommendationEngine
)

const recommendationClientRetryInterval = 10 * time.Second

type retryingRecommendationClient struct {
	mu          sync.RWMutex
	client      subsonic.RecommendationClient
	factory     func() (subsonic.RecommendationClient, error)
	retryAfter  time.Duration
	lastAttempt time.Time
	lastErr     error
}

// newRecommendationClient creates a Go recommendation client when possible and
// falls back to a retrying client when dependencies are temporarily unavailable.
func newRecommendationClient(ds model.DataStore) subsonic.RecommendationClient {
	factory := func() (subsonic.RecommendationClient, error) {
		return buildRecommendationClient(ds)
	}

	client, err := factory()
	if err == nil {
		return client
	}
	log.Warn(context.Background(), "Recommendation engine unavailable at startup, enabling retrying client", "error", err)
	return &retryingRecommendationClient{
		factory:     factory,
		retryAfter:  recommendationClientRetryInterval,
		lastAttempt: time.Now(),
		lastErr:     err,
	}
}

func buildRecommendationClient(ds model.DataStore) (subsonic.RecommendationClient, error) {
	milvusClient, _, err := newMilvusClientForRecommendations()
	if err != nil {
		return nil, fmt.Errorf("milvus init failed: %w", err)
	}

	resolver := newResolverForRecommendations(ds)
	engine := newEngineForRecommendations(milvusClient, resolver)
	if engine == nil {
		return nil, fmt.Errorf("recommendation engine initialization returned nil")
	}

	return subsonic.NewGoRecommendationClient(engine), nil
}

func (c *retryingRecommendationClient) Recommend(ctx context.Context, mode string, payload subsonic.RecommendationRequest) (*subsonic.RecommendationResponse, error) {
	if client := c.getClient(); client != nil {
		return client.Recommend(ctx, mode, payload)
	}
	if err := c.ensureClient(); err != nil {
		log.Warn(ctx, "Recommendation engine unavailable", "error", err)
		return &subsonic.RecommendationResponse{
			Warnings: []string{fmt.Sprintf("recommendation service unavailable: %v", err)},
		}, nil
	}
	client := c.getClient()
	if client == nil {
		return &subsonic.RecommendationResponse{
			Warnings: []string{"recommendation service unavailable"},
		}, nil
	}
	return client.Recommend(ctx, mode, payload)
}

func (c *retryingRecommendationClient) getClient() subsonic.RecommendationClient {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.client
}

func (c *retryingRecommendationClient) ensureClient() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.client != nil {
		return nil
	}

	now := time.Now()
	if c.lastErr != nil && c.retryAfter > 0 && now.Sub(c.lastAttempt) < c.retryAfter {
		return c.lastErr
	}

	c.lastAttempt = now
	client, err := c.factory()
	if err != nil {
		c.lastErr = err
		return err
	}

	c.client = client
	c.lastErr = nil
	log.Info(context.Background(), "Recommendation engine became available")
	return nil
}
