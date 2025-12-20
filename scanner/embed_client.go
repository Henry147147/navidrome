package scanner

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"net"
	"path/filepath"
	"time"

	"github.com/navidrome/navidrome/conf"
	"github.com/navidrome/navidrome/log"
)

// defaultSocketPath is the Unix socket path for the embedding service.
const defaultSocketPath = "/tmp/navidrome_embed.sock"

// statusCheckTimeout is the maximum time to wait for a status check.
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

type socketEmbeddingClient struct {
	socketPath    string
	statusTimeout time.Duration
	embedTimeout  time.Duration
}

func newPythonEmbeddingClient() embeddingClient {
	return newSocketEmbeddingClient()
}

func newSocketEmbeddingClient() embeddingClient {
	embedTimeout := conf.Server.Recommendations.EmbedTimeout
	if embedTimeout <= 0 {
		embedTimeout = conf.Server.Recommendations.Timeout
	}
	if embedTimeout <= 0 {
		embedTimeout = 10 * time.Minute
	}
	return &socketEmbeddingClient{
		socketPath:    defaultSocketPath,
		statusTimeout: statusCheckTimeout,
		embedTimeout:  embedTimeout,
	}
}

func (c *socketEmbeddingClient) CheckEmbedding(ctx context.Context, candidate embeddingCandidate) (embeddingStatus, error) {
	payload := map[string]any{
		"action":          "status",
		"track_id":        candidate.key(),
		"artist":          candidate.Artist,
		"title":           candidate.Title,
		"album":           candidate.Album,
		"alternate_names": []string{filepath.Base(candidate.TrackPath)},
	}

	var resp embeddingStatus
	if err := c.sendRequest(ctx, c.statusTimeout, payload, &resp); err != nil {
		return embeddingStatus{}, err
	}

	return resp, nil
}

func (c *socketEmbeddingClient) EmbedSong(ctx context.Context, candidate embeddingCandidate) error {
	payload := map[string]any{
		"action":     "embed",
		"music_file": candidate.absolutePath(),
		"name":       filepath.Base(candidate.TrackPath),
		"artist":     candidate.Artist,
		"title":      candidate.Title,
		"album":      candidate.Album,
		"track_id":   candidate.key(),
	}

	var resp struct {
		Status  string `json:"status"`
		Message string `json:"message,omitempty"`
	}
	if err := c.sendRequest(ctx, c.embedTimeout, payload, &resp); err != nil {
		return err
	}

	if resp.Status == "error" {
		return fmt.Errorf("embedding failed: %s", resp.Message)
	}

	return nil
}

func (c *socketEmbeddingClient) sendRequest(ctx context.Context, timeout time.Duration, payload any, response any) error {
	log.Debug(ctx, "Connecting to embedding socket", "path", c.socketPath)

	// Connect with timeout
	conn, err := net.DialTimeout("unix", c.socketPath, timeout)
	if err != nil {
		return fmt.Errorf("connect to socket %s: %w", c.socketPath, err)
	}
	defer conn.Close()

	// Set deadline for the entire operation
	if err := conn.SetDeadline(time.Now().Add(timeout)); err != nil {
		return fmt.Errorf("set deadline: %w", err)
	}

	// Encode and send JSON payload with newline
	encoder := json.NewEncoder(conn)
	if err := encoder.Encode(payload); err != nil {
		return fmt.Errorf("send request: %w", err)
	}

	log.Debug(ctx, "Sent request to embedding socket", "action", payload.(map[string]any)["action"])

	// Read response line
	reader := bufio.NewReader(conn)
	line, err := reader.ReadBytes('\n')
	if err != nil {
		return fmt.Errorf("read response: %w", err)
	}

	// Decode JSON response
	if err := json.Unmarshal(line, response); err != nil {
		return fmt.Errorf("decode response: %w", err)
	}

	log.Debug(ctx, "Received response from embedding socket")
	return nil
}
