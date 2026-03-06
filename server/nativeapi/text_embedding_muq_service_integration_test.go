//go:build integration

package nativeapi

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/navidrome/navidrome/conf"
)

const runMuQServiceEnv = "RUN_MUQ_SERVICE_TEST"

func TestMuQMulanTextEmbeddingService(t *testing.T) {
	if os.Getenv(runMuQServiceEnv) != "1" {
		t.Skipf("set %s=1 to run MuQ service integration test", runMuQServiceEnv)
	}

	baseURL := strings.TrimSpace(os.Getenv("MUQ_SERVICE_URL"))
	if baseURL == "" {
		baseURL = "http://127.0.0.1:9002"
	}

	prevBase := conf.Server.Recommendations.BaseURL
	prevTextBase := conf.Server.Recommendations.TextBaseURL
	prevTimeout := conf.Server.Recommendations.Timeout
	conf.Server.Recommendations.BaseURL = baseURL
	conf.Server.Recommendations.TextBaseURL = baseURL
	conf.Server.Recommendations.Timeout = 5 * time.Minute
	t.Cleanup(func() {
		conf.Server.Recommendations.BaseURL = prevBase
		conf.Server.Recommendations.TextBaseURL = prevTextBase
		conf.Server.Recommendations.Timeout = prevTimeout
	})

	var router Router
	vec, err := router.getTextEmbedding(context.Background(), "melancholic synthwave with dreamy vocals", "muq_mulan", 512)
	if err != nil {
		t.Fatalf("failed to embed text via MuQ service: %v", err)
	}
	if len(vec) == 0 {
		t.Fatalf("expected non-empty embedding")
	}
	if len(vec) != 512 {
		t.Fatalf("expected embedding dimension 512, got %d", len(vec))
	}
}
