// Package llamacpp provides an HTTP client for llama.cpp inference servers.
// This is a stub implementation - actual llama.cpp integration will be added later.
package llamacpp

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/navidrome/navidrome/log"
)

// Config holds llama.cpp server configuration.
type Config struct {
	AudioEmbedURL    string
	AudioDescribeURL string
	TextEmbedURL     string
	Timeout          time.Duration
	MaxRetries       int
	RetryBackoff     time.Duration
}

// AudioEmbedRequest for audio embedding endpoint.
type AudioEmbedRequest struct {
	AudioPath  string `json:"audio_path"`
	SampleRate int    `json:"sample_rate,omitempty"`
	BatchID    string `json:"batch_id,omitempty"`
}

// AudioEmbedResponse from audio embedding endpoint.
type AudioEmbedResponse struct {
	Embedding []float64 `json:"embedding"`
	ModelID   string    `json:"model_id"`
	Duration  float64   `json:"duration_seconds"`
	Error     string    `json:"error,omitempty"`
}

// AudioDescribeRequest for audio description endpoint.
type AudioDescribeRequest struct {
	AudioPath string `json:"audio_path"`
	Prompt    string `json:"prompt,omitempty"`
}

// AudioDescribeResponse from audio description endpoint.
type AudioDescribeResponse struct {
	Description    string    `json:"description"`
	AudioEmbedding []float64 `json:"audio_embedding"`
	ModelID        string    `json:"model_id"`
	Error          string    `json:"error,omitempty"`
}

// TextEmbedRequest for text embedding endpoint.
type TextEmbedRequest struct {
	Text    string `json:"text"`
	ModelID string `json:"model_id,omitempty"`
}

// TextEmbedResponse from text embedding endpoint.
type TextEmbedResponse struct {
	Embedding []float64 `json:"embedding"`
	ModelID   string    `json:"model_id"`
	Dimension int       `json:"dimension"`
	Error     string    `json:"error,omitempty"`
}

// Client communicates with llama.cpp servers for model inference.
type Client struct {
	config     Config
	httpClient *http.Client
}

// NewClient creates a new llama.cpp client.
func NewClient(cfg Config) *Client {
	if cfg.AudioEmbedURL == "" {
		cfg.AudioEmbedURL = "http://localhost:8080/embed/audio"
	}
	if cfg.AudioDescribeURL == "" {
		cfg.AudioDescribeURL = "http://localhost:8081/describe"
	}
	if cfg.TextEmbedURL == "" {
		cfg.TextEmbedURL = "http://localhost:8082/embed/text"
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 10 * time.Minute
	}
	if cfg.MaxRetries <= 0 {
		cfg.MaxRetries = 3
	}
	if cfg.RetryBackoff <= 0 {
		cfg.RetryBackoff = 2 * time.Second
	}

	return &Client{
		config: cfg,
		httpClient: &http.Client{
			Timeout: cfg.Timeout,
		},
	}
}

// HealthCheck verifies the llama.cpp server is reachable.
func (c *Client) HealthCheck(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.config.AudioEmbedURL+"/health", nil)
	if err != nil {
		return fmt.Errorf("create health check request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("llama.cpp health check failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		return fmt.Errorf("llama.cpp server unhealthy: %s", resp.Status)
	}

	return nil
}

// EmbedAudioBatch generates audio embeddings for multiple files.
// STUB: Returns error until llama.cpp backend is implemented.
func (c *Client) EmbedAudioBatch(ctx context.Context, reqs []AudioEmbedRequest) ([]AudioEmbedResponse, error) {
	if len(reqs) == 0 {
		return nil, nil
	}

	log.Debug(ctx, "EmbedAudioBatch called (stub)", "count", len(reqs))

	// STUB: In the future, this will send batch requests to llama.cpp
	return nil, fmt.Errorf("llama.cpp audio embedding not yet implemented: batch of %d files", len(reqs))
}

// DescribeAudioBatch generates text descriptions for multiple audio files.
// STUB: Returns error until llama.cpp backend is implemented.
func (c *Client) DescribeAudioBatch(ctx context.Context, reqs []AudioDescribeRequest) ([]AudioDescribeResponse, error) {
	if len(reqs) == 0 {
		return nil, nil
	}

	log.Debug(ctx, "DescribeAudioBatch called (stub)", "count", len(reqs))

	return nil, fmt.Errorf("llama.cpp audio description not yet implemented: batch of %d files", len(reqs))
}

// EmbedTextBatch generates text embeddings for multiple strings.
// STUB: Returns error until llama.cpp backend is implemented.
func (c *Client) EmbedTextBatch(ctx context.Context, reqs []TextEmbedRequest) ([]TextEmbedResponse, error) {
	if len(reqs) == 0 {
		return nil, nil
	}

	log.Debug(ctx, "EmbedTextBatch called (stub)", "count", len(reqs))

	return nil, fmt.Errorf("llama.cpp text embedding not yet implemented: batch of %d texts", len(reqs))
}

// EmbedAudio generates an audio embedding for a single file.
func (c *Client) EmbedAudio(ctx context.Context, req AudioEmbedRequest) (*AudioEmbedResponse, error) {
	results, err := c.EmbedAudioBatch(ctx, []AudioEmbedRequest{req})
	if err != nil {
		return nil, err
	}
	if len(results) == 0 {
		return nil, fmt.Errorf("no result from audio embedding")
	}
	return &results[0], nil
}

// DescribeAudio generates a text description for a single audio file.
func (c *Client) DescribeAudio(ctx context.Context, req AudioDescribeRequest) (*AudioDescribeResponse, error) {
	results, err := c.DescribeAudioBatch(ctx, []AudioDescribeRequest{req})
	if err != nil {
		return nil, err
	}
	if len(results) == 0 {
		return nil, fmt.Errorf("no result from audio description")
	}
	return &results[0], nil
}

// EmbedText generates a text embedding for a single string.
func (c *Client) EmbedText(ctx context.Context, req TextEmbedRequest) (*TextEmbedResponse, error) {
	results, err := c.EmbedTextBatch(ctx, []TextEmbedRequest{req})
	if err != nil {
		return nil, err
	}
	if len(results) == 0 {
		return nil, fmt.Errorf("no result from text embedding")
	}
	return &results[0], nil
}

// sendRequest sends a JSON request and decodes the response.
func (c *Client) sendRequest(ctx context.Context, url string, payload any, response any) error {
	data, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("encode request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(data))
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	var lastErr error
	for attempt := 0; attempt <= c.config.MaxRetries; attempt++ {
		if attempt > 0 {
			backoff := c.config.RetryBackoff * time.Duration(1<<(attempt-1))
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(backoff):
			}
		}

		resp, err := c.httpClient.Do(req)
		if err != nil {
			lastErr = fmt.Errorf("request failed: %w", err)
			continue
		}

		body, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			lastErr = fmt.Errorf("read response: %w", err)
			continue
		}

		if resp.StatusCode >= 500 {
			lastErr = fmt.Errorf("server error: %s", resp.Status)
			continue
		}

		if resp.StatusCode >= 400 {
			return fmt.Errorf("client error: %s - %s", resp.Status, string(body))
		}

		if err := json.Unmarshal(body, response); err != nil {
			return fmt.Errorf("decode response: %w", err)
		}

		return nil
	}

	return fmt.Errorf("all retries failed: %w", lastErr)
}
