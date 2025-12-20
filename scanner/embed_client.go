package scanner

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/navidrome/navidrome/conf"
	"github.com/navidrome/navidrome/log"
)

// statusCheckTimeout is the maximum time to wait for the /embed/status check.
// This should be short since it's just a database lookup.
const statusCheckTimeout = 30 * time.Second

type embeddingStatus struct {
	Embedded       bool   `json:"embedded"`
	HasDescription bool   `json:"hasDescription"`
	Name           string `json:"name"`
}

type embeddingClient interface {
	CheckEmbedding(ctx context.Context, candidate embeddingCandidate) (embeddingStatus, error)
	EmbedSong(ctx context.Context, candidate embeddingCandidate) error
}

type pythonEmbeddingClient struct {
	baseURL    string
	httpClient *http.Client
}

func newPythonEmbeddingClient() embeddingClient {
	resolveBase := func() string {
		candidates := []string{
			conf.Server.Recommendations.BaseURL,
			conf.Server.Recommendations.TextBaseURL,
			conf.Server.Recommendations.BatchBaseURL,
			"http://127.0.0.1:9002",
		}
		for _, val := range candidates {
			if base := normalizeBaseURL(val); base != "" {
				return base
			}
		}
		return ""
	}

	base := resolveBase()
	if base == "" {
		return nil
	}
	timeout := conf.Server.Recommendations.EmbedTimeout
	if timeout <= 0 {
		timeout = conf.Server.Recommendations.Timeout
	}
	if timeout <= 0 {
		timeout = 30 * time.Second
	}
	return &pythonEmbeddingClient{
		baseURL:    base,
		httpClient: &http.Client{Timeout: timeout},
	}
}

// normalizeBaseURL trims whitespace, removes a trailing slash, ensures a scheme is present,
// and validates the URL. Returns an empty string when the value is unusable so callers can
// fall back to the next candidate.
func normalizeBaseURL(raw string) string {
	raw = strings.TrimSpace(raw)
	raw = strings.TrimSuffix(raw, "/")
	if raw == "" {
		return ""
	}
	// Assume http if the caller omitted a scheme (common in config/env)
	if !strings.Contains(raw, "://") {
		raw = "http://" + raw
	}
	u, err := url.Parse(raw)
	if err != nil || u.Scheme == "" || u.Host == "" {
		return ""
	}
	return u.String()
}

func (c *pythonEmbeddingClient) CheckEmbedding(ctx context.Context, candidate embeddingCandidate) (embeddingStatus, error) {
	// Use a shorter timeout for status checks - they should be fast database lookups
	ctx, cancel := context.WithTimeout(ctx, statusCheckTimeout)
	defer cancel()

	payload := map[string]any{
		"track_id":        candidate.key(),
		"artist":          candidate.Artist,
		"title":           candidate.Title,
		"album":           candidate.Album,
		"alternate_names": []string{filepath.Base(candidate.TrackPath)},
	}
	var resp embeddingStatus
	if err := c.postJSON(ctx, "/embed/status", payload, &resp); err != nil {
		return embeddingStatus{}, err
	}
	return resp, nil
}

func (c *pythonEmbeddingClient) EmbedSong(ctx context.Context, candidate embeddingCandidate) error {
	payload := map[string]any{
		"music_file": candidate.absolutePath(),
		"name":       filepath.Base(candidate.TrackPath),
		"artist":     candidate.Artist,
		"title":      candidate.Title,
		"album":      candidate.Album,
		"track_id":   candidate.key(),
	}
	return c.postJSON(ctx, "/embed/audio", payload, nil)
}

func (c *pythonEmbeddingClient) postJSON(ctx context.Context, path string, payload any, decodeInto any) error {
	body, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("encode payload: %w", err)
	}

	url := c.baseURL + path
	log.Debug(ctx, "Calling embedding service", "url", url)
	fmt.Fprintf(os.Stderr, "[EMBED-DEBUG] POST %s payloadBytes=%d\n", url, len(body))

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		fmt.Fprintf(os.Stderr, "[EMBED-DEBUG] POST %s failed: %v\n", url, err)
		log.Debug(ctx, "Embedding service call failed", "url", url, "error", err)
		return fmt.Errorf("call %s: %w", path, err)
	}
	defer resp.Body.Close()

	fmt.Fprintf(os.Stderr, "[EMBED-DEBUG] POST %s status=%s\n", url, resp.Status)

	if resp.StatusCode >= http.StatusMultipleChoices {
		log.Debug(ctx, "Embedding service returned error status", "url", url, "status", resp.Status)
		return fmt.Errorf("%s returned %s", path, resp.Status)
	}

	log.Debug(ctx, "Embedding service call succeeded", "url", url, "status", resp.Status)

	if decodeInto == nil {
		return nil
	}
	if err := json.NewDecoder(resp.Body).Decode(decodeInto); err != nil {
		return fmt.Errorf("decode response: %w", err)
	}
	return nil
}
