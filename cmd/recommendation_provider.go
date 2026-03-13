package cmd

import (
	"context"
	"fmt"
	"strings"
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

func (c *retryingRecommendationClient) RecommendationHealth(ctx context.Context) subsonic.RecommendationHealth {
	if client := c.getClient(); client != nil {
		if provider, ok := client.(subsonic.RecommendationHealthProvider); ok {
			return provider.RecommendationHealth(ctx)
		}
		return subsonic.RecommendationHealth{Ready: true}
	}
	if err := c.ensureClient(); err != nil {
		return recommendationHealthFromError(err)
	}
	client := c.getClient()
	if provider, ok := client.(subsonic.RecommendationHealthProvider); ok {
		return provider.RecommendationHealth(ctx)
	}
	return subsonic.RecommendationHealth{Ready: client != nil}
}

func recommendationHealthFromError(err error) subsonic.RecommendationHealth {
	if err == nil {
		return subsonic.RecommendationHealth{Ready: true}
	}
	msg := strings.ToLower(err.Error())
	health := subsonic.RecommendationHealth{
		Ready:      false,
		ReasonCode: "milvus_unreachable",
		Message:    "Semantic recommendations are temporarily unavailable because Milvus is not ready.",
		Retryable:  true,
		Dependency: "engine",
	}
	if strings.Contains(msg, "schema mismatch") || strings.Contains(msg, "expected dim=") {
		health.ReasonCode = "milvus_schema_mismatch"
		health.Message = "Semantic recommendations are unavailable because the Milvus collection schema does not match the configured embedding dimensions."
		health.Retryable = false
		return health
	}
	if strings.Contains(msg, "disabled") {
		health.ReasonCode = "recommendation_service_disabled"
		health.Message = "Semantic recommendations are disabled."
		health.Retryable = false
	}
	return health
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
