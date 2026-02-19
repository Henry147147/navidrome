package nativeapi

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/navidrome/navidrome/conf"
)

func TestGetTextEmbeddingOpenAIResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/embeddings" {
			http.NotFound(w, r)
			return
		}
		var reqBody map[string]any
		if err := json.NewDecoder(r.Body).Decode(&reqBody); err != nil {
			t.Fatalf("failed to decode request: %v", err)
		}
		if got := reqBody["input"]; got != "hello world" {
			t.Fatalf("expected input hello world, got %#v", got)
		}
		if got := reqBody["model"]; got != "qwen8b" {
			t.Fatalf("expected model qwen8b, got %#v", got)
		}
		if got := reqBody["dimensions"]; got != float64(3) {
			t.Fatalf("expected requested dimensions=3, got %#v", got)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{
				{
					"embedding": []float64{0.1, 0.2, 0.3},
					"index":     0,
					"object":    "embedding",
				},
			},
			"model": "qwen8b",
		})
	}))
	defer srv.Close()

	prev := conf.Server.Recommendations.TextBaseURL
	conf.Server.Recommendations.TextBaseURL = srv.URL
	t.Cleanup(func() { conf.Server.Recommendations.TextBaseURL = prev })

	var router Router
	vec, err := router.getTextEmbedding(context.Background(), "hello world", "qwen8b", 3)
	if err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	if len(vec) != 3 {
		t.Fatalf("expected 3 values, got %d", len(vec))
	}
}

func TestGetTextEmbeddingLegacyFallbackResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/embeddings" {
			http.NotFound(w, r)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"embedding": []float64{1, 2, 3, 4},
		})
	}))
	defer srv.Close()

	prev := conf.Server.Recommendations.TextBaseURL
	conf.Server.Recommendations.TextBaseURL = srv.URL
	t.Cleanup(func() { conf.Server.Recommendations.TextBaseURL = prev })

	var router Router
	vec, err := router.getTextEmbedding(context.Background(), "hello world", "qwen8b", 0)
	if err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	if len(vec) != 4 {
		t.Fatalf("expected 4 values, got %d", len(vec))
	}
}

func TestGetTextEmbeddingErrorResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"error": map[string]any{
				"message": "invalid input",
				"type":    "invalid_request_error",
			},
		})
	}))
	defer srv.Close()

	prev := conf.Server.Recommendations.TextBaseURL
	conf.Server.Recommendations.TextBaseURL = srv.URL
	t.Cleanup(func() { conf.Server.Recommendations.TextBaseURL = prev })

	var router Router
	_, err := router.getTextEmbedding(context.Background(), "hello world", "qwen8b", 0)
	if err == nil {
		t.Fatalf("expected error")
	}
	if !strings.Contains(err.Error(), "invalid input") {
		t.Fatalf("expected propagated service error, got %v", err)
	}
}

func TestGetTextEmbeddingTruncatesToDesiredDimension(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{
				{
					"embedding": []float64{0.1, 0.2, 0.3, 0.4},
					"index":     0,
					"object":    "embedding",
				},
			},
		})
	}))
	defer srv.Close()

	prev := conf.Server.Recommendations.TextBaseURL
	conf.Server.Recommendations.TextBaseURL = srv.URL
	t.Cleanup(func() { conf.Server.Recommendations.TextBaseURL = prev })

	var router Router
	vec, err := router.getTextEmbedding(context.Background(), "hello world", "qwen8b", 2)
	if err != nil {
		t.Fatalf("expected embedding call to succeed, got %v", err)
	}
	if len(vec) != 2 {
		t.Fatalf("expected mock embedding length 2, got %d", len(vec))
	}
	if vec[0] != 0.1 || vec[1] != 0.2 {
		t.Fatalf("unexpected truncated vector: %#v", vec)
	}
}

func TestGetTextEmbeddingErrorsWhenReturnedDimensionTooSmall(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{
				{
					"embedding": []float64{0.1},
					"index":     0,
					"object":    "embedding",
				},
			},
		})
	}))
	defer srv.Close()

	prev := conf.Server.Recommendations.TextBaseURL
	conf.Server.Recommendations.TextBaseURL = srv.URL
	t.Cleanup(func() { conf.Server.Recommendations.TextBaseURL = prev })

	var router Router
	_, err := router.getTextEmbedding(context.Background(), "hello world", "qwen8b", 2)
	if err == nil {
		t.Fatalf("expected dimension error")
	}
	if !strings.Contains(err.Error(), "too small") {
		t.Fatalf("expected dimension size error, got %v", err)
	}
}

func TestTextEmbeddingHTTPTimeoutUsesRecommendationTimeoutWithMinimumFloor(t *testing.T) {
	prev := conf.Server.Recommendations.Timeout
	conf.Server.Recommendations.Timeout = 3 * time.Second
	t.Cleanup(func() { conf.Server.Recommendations.Timeout = prev })
	if got := textEmbeddingHTTPTimeout(); got != 30*time.Second {
		t.Fatalf("expected minimum timeout floor of 30s, got %s", got)
	}
}

func TestTextEmbeddingHTTPTimeoutUsesConfiguredTimeoutWhenHigher(t *testing.T) {
	prev := conf.Server.Recommendations.Timeout
	conf.Server.Recommendations.Timeout = 2 * time.Minute
	t.Cleanup(func() { conf.Server.Recommendations.Timeout = prev })
	if got := textEmbeddingHTTPTimeout(); got != 2*time.Minute {
		t.Fatalf("expected configured timeout to be used, got %s", got)
	}
}

func TestGetTextEmbeddingWithoutConfiguredURL(t *testing.T) {
	prevText := conf.Server.Recommendations.TextBaseURL
	prevBase := conf.Server.Recommendations.BaseURL
	conf.Server.Recommendations.TextBaseURL = ""
	conf.Server.Recommendations.BaseURL = ""
	t.Cleanup(func() {
		conf.Server.Recommendations.TextBaseURL = prevText
		conf.Server.Recommendations.BaseURL = prevBase
	})

	var router Router
	_, err := router.getTextEmbedding(context.Background(), "hello world", "qwen8b", 0)
	if err == nil {
		t.Fatalf("expected error when embedding URL is not configured")
	}
	if !strings.Contains(err.Error(), "base URL") {
		t.Fatalf("expected base URL error, got %v", err)
	}
}

func TestGetTextEmbeddingPropagatesNonJSONErrorBody(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
		_, _ = fmt.Fprintln(w, "upstream unavailable")
	}))
	defer srv.Close()

	prev := conf.Server.Recommendations.TextBaseURL
	conf.Server.Recommendations.TextBaseURL = srv.URL
	t.Cleanup(func() { conf.Server.Recommendations.TextBaseURL = prev })

	var router Router
	_, err := router.getTextEmbedding(context.Background(), "hello world", "qwen8b", 0)
	if err == nil {
		t.Fatalf("expected error")
	}
	if !strings.Contains(err.Error(), "status 502") {
		t.Fatalf("expected status code in error, got %v", err)
	}
}
