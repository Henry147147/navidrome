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
	CollectionMuQAudio = "muq_audio_embedding" // MuQ audio-only embedding
	CollectionMuQMulan = "muq_mulan_embedding" // MuQ-MuLan shared audio/text embedding

	// Deprecated aliases kept temporarily while callers migrate.
	CollectionLyrics      = CollectionMuQMulan
	CollectionDescription = CollectionMuQMulan
	CollectionFlamingo    = CollectionMuQAudio
)

// Embedding dimensions.
const (
	DimMuQAudio = 1024 // MuQ audio embedding dimension
	DimMuQMulan = 512  // MuQ-MuLan shared embedding dimension

	// Deprecated aliases kept temporarily while callers migrate.
	DimLyrics      = DimMuQMulan
	DimDescription = DimMuQMulan
	DimFlamingo    = DimMuQAudio
)

// Dimensions holds embedding dimensions for Milvus collections.
type Dimensions struct {
	MuQAudio int
	MuQMulan int
}

// DefaultDimensions returns the default embedding dimensions.
func DefaultDimensions() Dimensions {
	return Dimensions{
		MuQAudio: DimMuQAudio,
		MuQMulan: DimMuQMulan,
	}
}

// Config holds Milvus connection settings.
type Config struct {
	URI        string        // Server URI or file path for Milvus Lite
	Timeout    time.Duration // Connection/operation timeout
	MaxRetries int           // Max retry attempts
	Dimensions Dimensions    // Embedding dimensions by collection
}

// DefaultConfig returns sensible defaults.
func DefaultConfig() Config {
	return Config{
		URI:        "http://localhost:19530",
		Timeout:    30 * time.Second,
		MaxRetries: 3,
		Dimensions: DefaultDimensions(),
	}
}

func normalizeDimensions(dims Dimensions) Dimensions {
	defaults := DefaultDimensions()
	if dims.MuQAudio <= 0 {
		dims.MuQAudio = defaults.MuQAudio
	}
	if dims.MuQMulan <= 0 {
		dims.MuQMulan = defaults.MuQMulan
	}
	return dims
}

func normalizeConfig(cfg Config) Config {
	defaults := DefaultConfig()
	if cfg.URI == "" {
		cfg.URI = defaults.URI
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = defaults.Timeout
	}
	if cfg.MaxRetries <= 0 {
		cfg.MaxRetries = defaults.MaxRetries
	}
	cfg.Dimensions = normalizeDimensions(cfg.Dimensions)
	return cfg
}

// Client wraps the Milvus SDK client with collection management.
type Client struct {
	config       Config
	milvusClient client.Client
	mu           sync.RWMutex
	loaded       map[string]bool // Track loaded collections
	dimensions   Dimensions
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

	milvusCfg = normalizeConfig(milvusCfg)

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
		dimensions:   milvusCfg.Dimensions,
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
		{CollectionMuQAudio, c.dimensions.MuQAudio},
		{CollectionMuQMulan, c.dimensions.MuQMulan},
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
