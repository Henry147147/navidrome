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
	baseURL      string
	statusClient *http.Client
	embedClient  *http.Client
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
	statusTimeout := statusCheckTimeout
	if conf.Server.Recommendations.Timeout > 0 && conf.Server.Recommendations.Timeout < statusTimeout {
		statusTimeout = conf.Server.Recommendations.Timeout
	}
	// Use a simpler Transport based on DefaultTransport to avoid connection issues
	statusTransport := http.DefaultTransport.(*http.Transport).Clone()
	statusTransport.DisableKeepAlives = true
	statusTransport.ResponseHeaderTimeout = statusTimeout
	statusTransport.IdleConnTimeout = statusTimeout
	return &pythonEmbeddingClient{
		baseURL:      base,
		statusClient: &http.Client{Timeout: statusTimeout, Transport: statusTransport},
		embedClient:  &http.Client{Timeout: timeout},
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
	start := time.Now()

	payload := map[string]any{
		"track_id":        candidate.key(),
		"artist":          candidate.Artist,
		"title":           candidate.Title,
		"album":           candidate.Album,
		"alternate_names": []string{filepath.Base(candidate.TrackPath)},
	}
	var resp embeddingStatus
	if err := c.postJSON(ctx, c.statusClient, "/embed/status", payload, &resp, true); err != nil {
		fmt.Fprintf(os.Stderr, "[EMBED-DEBUG] CheckEmbedding error after %s: %v\n", time.Since(start), err)
		return embeddingStatus{}, err
	}
	fmt.Fprintf(os.Stderr, "[EMBED-DEBUG] CheckEmbedding finished in %s\n", time.Since(start))
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
	return c.postJSON(ctx, c.embedClient, "/embed/audio", payload, nil, false)
}

func (c *pythonEmbeddingClient) postJSON(ctx context.Context, client *http.Client, path string, payload any, decodeInto any, closeConn bool) error {
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
	if closeConn {
		req.Close = true
	}

	if client == nil {
		client = http.DefaultClient
	}
	fmt.Fprintf(os.Stderr, "[EMBED-DEBUG] About to call client.Do for %s\n", url)
	os.Stderr.Sync()
	resp, err := client.Do(req)
	fmt.Fprintf(os.Stderr, "[EMBED-DEBUG] client.Do returned for %s, err=%v\n", url, err)
	os.Stderr.Sync()
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
	fmt.Fprintf(os.Stderr, "[EMBED-DEBUG] Decoding response from %s\n", url)
	if err := json.NewDecoder(resp.Body).Decode(decodeInto); err != nil {
		fmt.Fprintf(os.Stderr, "[EMBED-DEBUG] Decode failed for %s: %v\n", url, err)
		return fmt.Errorf("decode response: %w", err)
	}
	fmt.Fprintf(os.Stderr, "[EMBED-DEBUG] Decoded response from %s\n", url)
	return nil
}
