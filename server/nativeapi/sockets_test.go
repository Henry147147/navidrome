package nativeapi

import (
	"sync"
	"testing"
	"time"

	"github.com/navidrome/navidrome/conf"
)

func TestGetEmbedClientUsesEmbedTimeout(t *testing.T) {
	embedClient = nil
	embedClientOnce = sync.Once{}

	conf.Server.Recommendations.BaseURL = "http://localhost:9999"
	conf.Server.Recommendations.Timeout = 5 * time.Second
	conf.Server.Recommendations.EmbedTimeout = 2 * time.Minute

	client := getEmbedClient()

	impl, ok := client.(*embedHTTPClient)
	if !ok {
		t.Fatalf("expected embedHTTPClient, got %T", client)
	}
	if impl.httpClient.Timeout != conf.Server.Recommendations.EmbedTimeout {
		t.Fatalf("expected timeout %s, got %s", conf.Server.Recommendations.EmbedTimeout, impl.httpClient.Timeout)
	}
}
