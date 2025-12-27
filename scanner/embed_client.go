package scanner

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net"
	"path/filepath"
	"sync/atomic"
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
	HealthCheck(ctx context.Context) error
	CheckEmbedding(ctx context.Context, candidate embeddingCandidate) (embeddingStatus, error)
	EmbedSong(ctx context.Context, candidate embeddingCandidate) error
	FlushBatch(ctx context.Context) error
}

type socketEmbeddingClient struct {
	socketPath    string
	statusTimeout time.Duration
	embedTimeout  time.Duration
}

var embedRequestSeq uint64

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

func (c *socketEmbeddingClient) HealthCheck(ctx context.Context) error {
	payload := map[string]any{
		"action": "health",
	}

	var resp struct {
		Status          string `json:"status"`
		MilvusConnected bool   `json:"milvus_connected"`
	}

	if err := c.sendRequest(ctx, 10*time.Second, payload, &resp); err != nil {
		return fmt.Errorf("health check request failed: %w", err)
	}

	if resp.Status != "ok" {
		return fmt.Errorf("service unhealthy: status=%s milvus_connected=%v", resp.Status, resp.MilvusConnected)
	}

	return nil
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

// FlushBatch signals the embedding server to immediately process any pending
// batch requests. This should be called at the end of a scan to ensure all
// queued embeddings are processed without waiting for the batch timeout.
func (c *socketEmbeddingClient) FlushBatch(ctx context.Context) error {
	payload := map[string]any{
		"action": "flush",
	}

	var resp struct {
		Status  string `json:"status"`
		Message string `json:"message,omitempty"`
	}

	if err := c.sendRequest(ctx, 30*time.Second, payload, &resp); err != nil {
		return fmt.Errorf("flush batch request failed: %w", err)
	}

	if resp.Status == "error" {
		return fmt.Errorf("flush batch failed: %s", resp.Message)
	}

	log.Debug(ctx, "Flushed embedding batch", "message", resp.Message)
	return nil
}

func (c *socketEmbeddingClient) sendRequest(ctx context.Context, timeout time.Duration, payload any, response any) error {
	requestID := fmt.Sprintf("embed-%d", atomic.AddUint64(&embedRequestSeq, 1))
	action := "unknown"
	trackID := ""
	if payloadMap, ok := payload.(map[string]any); ok {
		payloadMap["request_id"] = requestID
		if value, ok := payloadMap["action"].(string); ok && value != "" {
			action = value
		}
		if value, ok := payloadMap["track_id"].(string); ok {
			trackID = value
		}
	}

	log.Debug(ctx, "Connecting to embedding socket", "path", c.socketPath, "requestId", requestID, "action", action, "trackId", trackID)

	// Connect with timeout
	connectStart := time.Now()
	conn, err := net.DialTimeout("unix", c.socketPath, timeout)
	if err != nil {
		return fmt.Errorf("connect to socket %s: %w", c.socketPath, err)
	}
	defer conn.Close()
	log.Debug(ctx, "Connected to embedding socket", "requestId", requestID, "elapsed", time.Since(connectStart))

	// Set deadline for the entire operation
	if err := conn.SetDeadline(time.Now().Add(timeout)); err != nil {
		return fmt.Errorf("set deadline: %w", err)
	}

	// Encode and send JSON payload with newline
	var buf bytes.Buffer
	encoder := json.NewEncoder(&buf)
	if err := encoder.Encode(payload); err != nil {
		return fmt.Errorf("encode request: %w", err)
	}
	writeStart := time.Now()
	if _, err := conn.Write(buf.Bytes()); err != nil {
		return fmt.Errorf("send request: %w", err)
	}
	if unixConn, ok := conn.(*net.UnixConn); ok {
		// Signal that we're done writing to avoid half-open ambiguity on the server.
		_ = unixConn.CloseWrite()
	}

	log.Debug(ctx, "Sent request to embedding socket", "requestId", requestID, "action", action, "trackId", trackID, "payloadBytes", buf.Len(), "elapsed", time.Since(writeStart))

	// Read response line
	reader := bufio.NewReader(conn)
	readStart := time.Now()
	line, err := reader.ReadBytes('\n')
	if err != nil {
		return fmt.Errorf("read response: %w", err)
	}

	// Decode JSON response
	if err := json.Unmarshal(line, response); err != nil {
		return fmt.Errorf("decode response: %w", err)
	}

	log.Debug(ctx, "Received response from embedding socket", "requestId", requestID, "action", action, "trackId", trackID, "payloadBytes", len(line), "elapsed", time.Since(readStart))
	return nil
}
