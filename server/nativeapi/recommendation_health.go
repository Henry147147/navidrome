package nativeapi

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/navidrome/navidrome/server/subsonic"
)

type recommendationDependencyStatus struct {
	Ready      bool   `json:"ready"`
	ReasonCode string `json:"reasonCode,omitempty"`
	Message    string `json:"message,omitempty"`
}

type recommendationHealthPayload struct {
	Status         string                         `json:"status"`
	Engine         recommendationDependencyStatus `json:"engine"`
	Text           recommendationDependencyStatus `json:"text"`
	Batch          recommendationDependencyStatus `json:"batch"`
	AvailableModes []string                       `json:"availableModes,omitempty"`
	DegradedModes  []string                       `json:"degradedModes,omitempty"`
}

type recommendationErrorBody struct {
	Code       string `json:"code"`
	Message    string `json:"message"`
	Retryable  bool   `json:"retryable"`
	Dependency string `json:"dependency,omitempty"`
}

type recommendationAPIError struct {
	status int
	body   recommendationErrorBody
}

func (e *recommendationAPIError) Error() string {
	if e == nil {
		return ""
	}
	return e.body.Message
}

func newRecommendationAPIError(status int, code string, message string, retryable bool, dependency string) *recommendationAPIError {
	return &recommendationAPIError{
		status: status,
		body: recommendationErrorBody{
			Code:       code,
			Message:    message,
			Retryable:  retryable,
			Dependency: dependency,
		},
	}
}

func writeRecommendationAPIError(w http.ResponseWriter, err error) {
	var apiErr *recommendationAPIError
	if err != nil && AsRecommendationAPIError(err, &apiErr) {
		writeJSON(w, apiErr.status, apiErr.body)
		return
	}
	writeJSON(w, http.StatusBadGateway, recommendationErrorBody{
		Code:       "recommendation_backend_error",
		Message:    "Unable to generate recommendations right now.",
		Retryable:  true,
		Dependency: "engine",
	})
}

func AsRecommendationAPIError(err error, target **recommendationAPIError) bool {
	if err == nil || target == nil {
		return false
	}
	var apiErr *recommendationAPIError
	if !errors.As(err, &apiErr) {
		return false
	}
	*target = apiErr
	return true
}

func (n *Router) handleRecommendationHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, n.getRecommendationHealth(r.Context()))
}

func (n *Router) getRecommendationHealth(ctx context.Context) recommendationHealthPayload {
	engine := n.recommendationEngineHealth(ctx)
	text := n.textEmbeddingHealth(ctx)
	batch := n.batchEmbeddingHealth(ctx)

	availableModes := make([]string, 0, 6)
	degradedModes := make([]string, 0, 6)

	baseModes := []string{
		modeRecentRecommendations,
		modeFavoritesRecommendations,
		modeAllRecommendations,
		modeDiscoveryRecommendations,
		modeCustomRecommendations,
	}
	for _, mode := range baseModes {
		if engine.Ready {
			availableModes = append(availableModes, mode)
		} else {
			degradedModes = append(degradedModes, mode)
		}
	}
	if engine.Ready && text.Ready {
		availableModes = append(availableModes, modeTextRecommendations)
	} else {
		degradedModes = append(degradedModes, modeTextRecommendations)
	}

	status := "ready"
	switch {
	case len(availableModes) == 0:
		status = "unavailable"
	case !engine.Ready || !text.Ready || !batch.Ready:
		status = "degraded"
	}

	return recommendationHealthPayload{
		Status:         status,
		Engine:         engine,
		Text:           text,
		Batch:          batch,
		AvailableModes: availableModes,
		DegradedModes:  degradedModes,
	}
}

func (n *Router) recommendationEngineHealth(ctx context.Context) recommendationDependencyStatus {
	if n.recommender == nil {
		return recommendationDependencyStatus{
			Ready:      false,
			ReasonCode: "recommendation_service_disabled",
			Message:    "Semantic recommendations are disabled.",
		}
	}
	if provider, ok := n.recommender.(subsonic.RecommendationHealthProvider); ok {
		return recommendationDependencyStatusFromClient(provider.RecommendationHealth(ctx))
	}
	return recommendationDependencyStatus{Ready: true}
}

func recommendationDependencyStatusFromClient(health subsonic.RecommendationHealth) recommendationDependencyStatus {
	if health.Ready {
		return recommendationDependencyStatus{Ready: true}
	}
	return recommendationDependencyStatus{
		Ready:      false,
		ReasonCode: strings.TrimSpace(health.ReasonCode),
		Message:    strings.TrimSpace(health.Message),
	}
}

func (n *Router) textEmbeddingHealth(ctx context.Context) recommendationDependencyStatus {
	if !n.recommendationEngineHealth(ctx).Ready {
		return recommendationDependencyStatus{
			Ready:      false,
			ReasonCode: "text_embedding_unreachable",
			Message:    "Text recommendations are unavailable until the semantic recommendation engine is ready.",
		}
	}
	baseURL := strings.TrimSpace(textEmbedBaseURL())
	if baseURL == "" {
		return recommendationDependencyStatus{
			Ready:      false,
			ReasonCode: "text_embedding_unreachable",
			Message:    "Text recommendation service is not configured.",
		}
	}
	if err := probeRecommendationHTTP(ctx, baseURL+"/v1/models"); err != nil {
		return recommendationDependencyStatus{
			Ready:      false,
			ReasonCode: "text_embedding_unreachable",
			Message:    "Text recommendations are currently offline.",
		}
	}
	return recommendationDependencyStatus{Ready: true}
}

func (n *Router) batchEmbeddingHealth(ctx context.Context) recommendationDependencyStatus {
	baseURL := strings.TrimSpace(batchBaseURL())
	if baseURL == "" {
		return recommendationDependencyStatus{
			Ready:      false,
			ReasonCode: "batch_service_unreachable",
			Message:    "Batch embedding service is not configured.",
		}
	}
	if err := probeRecommendationHTTP(ctx, baseURL+"/batch/progress"); err != nil {
		return recommendationDependencyStatus{
			Ready:      false,
			ReasonCode: "batch_service_unreachable",
			Message:    "Batch embedding service is currently offline.",
		}
	}
	return recommendationDependencyStatus{Ready: true}
}

func probeRecommendationHTTP(ctx context.Context, url string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	client := &http.Client{Timeout: recommendationHealthTimeout()}
	resp, err := client.Do(req) // #nosec G704 -- destination URL comes from trusted server config
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 500 {
		return fmt.Errorf("received status %d", resp.StatusCode)
	}
	return nil
}

func recommendationHealthTimeout() time.Duration {
	timeout := textEmbeddingHTTPTimeout() / 3
	if timeout < 5*time.Second {
		timeout = 5 * time.Second
	}
	return timeout
}

func (n *Router) requireRecommendationEngine(ctx context.Context) error {
	health := n.recommendationEngineHealth(ctx)
	if health.Ready {
		return nil
	}
	return newRecommendationAPIError(
		http.StatusServiceUnavailable,
		health.ReasonCode,
		health.Message,
		health.ReasonCode != "milvus_schema_mismatch",
		"engine",
	)
}

func (n *Router) requireTextRecommendationReady(ctx context.Context) error {
	if err := n.requireRecommendationEngine(ctx); err != nil {
		return err
	}
	health := n.textEmbeddingHealth(ctx)
	if health.Ready {
		return nil
	}
	return newRecommendationAPIError(
		http.StatusServiceUnavailable,
		health.ReasonCode,
		health.Message,
		true,
		"text",
	)
}
