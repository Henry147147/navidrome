// Package milvus provides a client for the Milvus vector database.
package milvus

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/navidrome/navidrome/log"
)

// Collection names for Milvus.
const (
	CollectionEmbedding            = "embedding"                // MuQ audio (1536D)
	CollectionDescriptionEmbedding = "description_embedding"    // Qwen3 text (2560D)
	CollectionFlamingoAudio        = "flamingo_audio_embedding" // Flamingo (1024D)
)

// Embedding dimensions.
const (
	DimMuQ           = 1536 // 512 * 3 (enriched: mean + IQR sigma + dmean)
	DimQwen3         = 2560
	DimFlamingoAudio = 1024
)

// Config holds Milvus connection settings.
type Config struct {
	URI        string        // Server URI or file path for Milvus Lite
	Timeout    time.Duration // Connection/operation timeout
	MaxRetries int           // Max retry attempts
}

// DefaultConfig returns sensible defaults.
func DefaultConfig() Config {
	return Config{
		URI:        "http://localhost:19530",
		Timeout:    30 * time.Second,
		MaxRetries: 3,
	}
}

// Client wraps the Milvus SDK client with collection management.
type Client struct {
	config       Config
	milvusClient client.Client
	mu           sync.RWMutex
	loaded       map[string]bool // Track loaded collections
}

// NewClient creates a new Milvus client.
func NewClient(ctx context.Context, cfg interface{}) (*Client, error) {
	// Accept either Config or any struct with URI field
	var milvusCfg Config
	switch c := cfg.(type) {
	case Config:
		milvusCfg = c
	default:
		milvusCfg = DefaultConfig()
	}

	if milvusCfg.URI == "" {
		milvusCfg = DefaultConfig()
	}

	log.Info(ctx, "Connecting to Milvus", "uri", milvusCfg.URI)

	c, err := client.NewClient(ctx, client.Config{
		Address: milvusCfg.URI,
	})
	if err != nil {
		return nil, fmt.Errorf("connect to milvus at %s: %w", milvusCfg.URI, err)
	}

	return &Client{
		config:       milvusCfg,
		milvusClient: c,
		loaded:       make(map[string]bool),
	}, nil
}

// Close releases the Milvus connection.
func (c *Client) Close() error {
	if c.milvusClient != nil {
		return c.milvusClient.Close()
	}
	return nil
}

// EnsureCollections creates all required collections if they don't exist.
func (c *Client) EnsureCollections(ctx context.Context) error {
	collections := []struct {
		name string
		dim  int
	}{
		{CollectionEmbedding, DimMuQ},
		{CollectionDescriptionEmbedding, DimQwen3},
		{CollectionFlamingoAudio, DimFlamingoAudio},
	}

	for _, col := range collections {
		if err := c.ensureCollection(ctx, col.name, col.dim); err != nil {
			return fmt.Errorf("ensure collection %s: %w", col.name, err)
		}
	}

	return nil
}

// loadCollection loads a collection into memory if not already loaded.
func (c *Client) loadCollection(ctx context.Context, name string) error {
	c.mu.RLock()
	loaded := c.loaded[name]
	c.mu.RUnlock()

	if loaded {
		return nil
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	// Double-check after acquiring write lock
	if c.loaded[name] {
		return nil
	}

	if err := c.milvusClient.LoadCollection(ctx, name, false); err != nil {
		return fmt.Errorf("load collection %s: %w", name, err)
	}

	c.loaded[name] = true
	return nil
}

// flush ensures data is persisted to disk.
func (c *Client) flush(ctx context.Context, name string) error {
	if err := c.milvusClient.Flush(ctx, name, false); err != nil {
		return fmt.Errorf("flush collection %s: %w", name, err)
	}
	return nil
}

// Raw returns the underlying Milvus client for advanced operations.
func (c *Client) Raw() client.Client {
	return c.milvusClient
}
