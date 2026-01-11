package main

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/BurntSushi/toml"
)

// Config holds all configuration for the embedder CLI.
type Config struct {
	Database DatabaseConfig `toml:"database"`
	Milvus   MilvusConfig   `toml:"milvus"`
	Models   ModelsConfig   `toml:"models"`
	Embedder EmbedderConfig `toml:"embedder"`
	Logging  LoggingConfig  `toml:"logging"`
	CLI      CLIConfig      `toml:"-"` // CLI-only options, not in config file
}

// DatabaseConfig holds database connection settings.
type DatabaseConfig struct {
	Path string `toml:"path"` // Path to Navidrome database (or postgres:// URI)
	Type string `toml:"type"` // "sqlite3" or "postgres" (auto-detected from path)
}

// MilvusConfig holds Milvus connection settings.
type MilvusConfig struct {
	URI        string           `toml:"uri"`
	Timeout    time.Duration    `toml:"timeout"`
	Dimensions MilvusDimensions `toml:"dimensions"`
}

// MilvusDimensions holds embedding dimensions for each model type.
type MilvusDimensions struct {
	Lyrics      int `toml:"lyrics"`      // Lyrics text embedding dimension
	Description int `toml:"description"` // Description text embedding dimension
	Flamingo    int `toml:"flamingo"`    // Flamingo audio embedding dimension
}

// ModelsConfig holds paths to model files and libraries.
type ModelsConfig struct {
	LibraryPath string `toml:"library_path"` // Path to llama.cpp libraries
	TextModel   string `toml:"text_model"`   // Path to text embedding model
	AudioModel  string `toml:"audio_model"`  // Path to audio model
	Projector   string `toml:"projector"`    // Path to audio projector (mmproj)
}

// EmbedderConfig holds embedding generation settings.
type EmbedderConfig struct {
	BatchSize         int  `toml:"batch_size"`         // Batch size for processing
	GPULayers         int  `toml:"gpu_layers"`         // GPU layers to offload
	Threads           int  `toml:"threads"`            // CPU threads for inference (0 = auto)
	EnableLyrics      bool `toml:"enable_lyrics"`      // Enable lyrics generation
	EnableDescription bool `toml:"enable_description"` // Enable audio description
	EnableFlamingo    bool `toml:"enable_flamingo"`    // Enable audio embedding
}

// LoggingConfig holds logging settings.
type LoggingConfig struct {
	Level string `toml:"level"` // Log level: debug, info, warn, error
	File  string `toml:"file"`  // Log file path (empty = stderr)
}

// CLIConfig holds command-line only options.
type CLIConfig struct {
	ConfigFile string // Path to TOML config file
	DryRun     bool   // Show what would be embedded without actually doing it
	Force      bool   // Re-embed tracks even if embeddings exist
	Limit      int    // Max tracks to process (0 = all)
	Filter     string // SQL WHERE clause to filter tracks
}

// DefaultConfig returns configuration with sensible defaults.
func DefaultConfig() Config {
	return Config{
		Database: DatabaseConfig{
			Path: getEnvOrDefault("ND_DBPATH", "navidrome.db"),
			Type: "sqlite3",
		},
		Milvus: MilvusConfig{
			URI:     getEnvOrDefault("MILVUS_URI", "http://localhost:19530"),
			Timeout: 30 * time.Second,
			Dimensions: MilvusDimensions{
				Lyrics:      2560,
				Description: 2560,
				Flamingo:    3584,
			},
		},
		Models: ModelsConfig{
			LibraryPath: resolveModelPath("musicembed/llama-lib"),
			TextModel:   resolveModelPath("musicembed/models/qwen-embedder-4b.gguf"),
			AudioModel:  resolveModelPath("musicembed/models/music-flamingo.gguf"),
			Projector:   resolveModelPath("musicembed/models/mmproj-music-flamingo.gguf"),
		},
		Embedder: EmbedderConfig{
			BatchSize:         50,
			GPULayers:         99,
			Threads:           0, // 0 = auto-detect
			EnableLyrics:      true,
			EnableDescription: true,
			EnableFlamingo:    true,
		},
		Logging: LoggingConfig{
			Level: getEnvOrDefault("LOG_LEVEL", "info"),
			File:  "", // Empty = stderr
		},
		CLI: CLIConfig{
			DryRun: false,
			Force:  false,
			Limit:  0, // 0 = no limit
			Filter: "",
		},
	}
}

// LoadConfigFile loads configuration from a TOML file and merges with existing config.
func LoadConfigFile(path string, cfg *Config) error {
	if path == "" {
		return nil // No config file specified
	}

	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil // Config file doesn't exist, use defaults
		}
		return fmt.Errorf("failed to read config file: %w", err)
	}

	if err := toml.Unmarshal(data, cfg); err != nil {
		return fmt.Errorf("failed to parse config file: %w", err)
	}

	return nil
}

// Validate checks if the configuration is valid.
func (c *Config) Validate() error {
	// Validate database path
	if c.Database.Path == "" {
		return fmt.Errorf("database path is required")
	}

	// Auto-detect database type from path
	if c.Database.Type == "" {
		if isPostgresURI(c.Database.Path) {
			c.Database.Type = "postgres"
		} else {
			c.Database.Type = "sqlite3"
		}
	}

	// Validate Milvus URI
	if c.Milvus.URI == "" {
		return fmt.Errorf("Milvus URI is required")
	}

	// Validate model paths exist
	if !fileExists(c.Models.TextModel) {
		return fmt.Errorf("text model not found: %s", c.Models.TextModel)
	}
	if c.Embedder.EnableFlamingo && !fileExists(c.Models.AudioModel) {
		return fmt.Errorf("audio model not found: %s", c.Models.AudioModel)
	}
	if c.Embedder.EnableFlamingo && !fileExists(c.Models.Projector) {
		return fmt.Errorf("projector not found: %s", c.Models.Projector)
	}

	// Validate library path
	if !dirExists(c.Models.LibraryPath) {
		return fmt.Errorf("library path not found: %s", c.Models.LibraryPath)
	}

	// Validate batch size
	if c.Embedder.BatchSize < 1 {
		return fmt.Errorf("batch size must be at least 1, got: %d", c.Embedder.BatchSize)
	}

	// Validate log level
	validLevels := map[string]bool{"debug": true, "info": true, "warn": true, "error": true}
	if !validLevels[c.Logging.Level] {
		return fmt.Errorf("invalid log level: %s (must be debug, info, warn, or error)", c.Logging.Level)
	}

	return nil
}

// Helper functions

func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func resolveModelPath(relativePath string) string {
	// If absolute path, return as-is
	if filepath.IsAbs(relativePath) {
		return relativePath
	}

	// Check if file exists relative to current directory
	if fileExists(relativePath) {
		return relativePath
	}

	// Check if file exists relative to parent directory (when running from embedmusic/)
	parentPath := filepath.Join("..", relativePath)
	if fileExists(parentPath) {
		return parentPath
	}

	// Return original relative path
	return relativePath
}

func fileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

func dirExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && info.IsDir()
}

func isPostgresURI(path string) bool {
	return len(path) > 11 && path[:11] == "postgres://"
}
