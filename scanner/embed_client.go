package scanner

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"path/filepath"
	"strings"
	"time"

	"github.com/navidrome/navidrome/conf"
)

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
			base := strings.TrimSuffix(strings.TrimSpace(val), "/")
			if base != "" {
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

func (c *pythonEmbeddingClient) CheckEmbedding(ctx context.Context, candidate embeddingCandidate) (embeddingStatus, error) {
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
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+path, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("call %s: %w", path, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= http.StatusMultipleChoices {
		return fmt.Errorf("%s returned %s", path, resp.Status)
	}

	if decodeInto == nil {
		return nil
	}
	if err := json.NewDecoder(resp.Body).Decode(decodeInto); err != nil {
		return fmt.Errorf("decode response: %w", err)
	}
	return nil
}
