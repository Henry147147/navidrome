package musicembed

import (
	"os"
	"path/filepath"
)

// Config holds all configuration for the music embedding library.
type Config struct {
	// LibraryPath is the path to the llama-lib directory containing shared libraries.
	LibraryPath string

	// EmbeddingModelFile is the path to the Qwen embedder model file.
	EmbeddingModelFile string

	// MmprojFile is the path to the mmproj model for audio processing.
	MmprojFile string

	// ModelFile is the path to the Music Flamingo model.
	ModelFile string

	// ContextSize is the context size for model inference.
	ContextSize int

	// BatchSize is the batch size for model inference.
	BatchSize int

	// GenerationMargin is extra context reserved for generation.
	GenerationMargin int

	// MaxOutputTokens is the maximum number of tokens to generate.
	MaxOutputTokens int

	// GPULayers is the number of layers to offload to GPU.
	GPULayers int

	// MainGPU is the GPU index to use when offloading.
	MainGPU int

	// UseGPU enables GPU acceleration for mmproj processing.
	UseGPU bool

	// Threads is the number of threads for CPU processing.
	Threads int

	// DescriptionPrompt is the prompt template for generating song descriptions.
	DescriptionPrompt string

	// LyricsPrompt is the prompt template for generating song lyrics.
	LyricsPrompt string
}

// DefaultConfig returns a Config with sensible default values.
func DefaultConfig() Config {
	cfg := Config{
		LibraryPath:        "./llama-lib",
		EmbeddingModelFile: "models/qwen-embedder-4b.gguf",
		MmprojFile:         "models/mmproj-music-flamingo.gguf",
		ModelFile:          "models/music-flamingo.gguf",
		ContextSize:        8192,
		BatchSize:          2048,
		GenerationMargin:   512,
		MaxOutputTokens:    8192,
		GPULayers:          99,
		MainGPU:            0,
		UseGPU:             true,
		Threads:            8,
		DescriptionPrompt:  "Describe this track in full detail - tell me the genre, tempo, and key, then dive into the instruments, production style, and overall mood it creates.",
		LyricsPrompt:       "Please provide the structured lyric sheet for this song.",
	}

	// When running from the repo root, assets live under ./musicembed.
	if needsRepoAssetFallback(cfg) {
		assetBase := "musicembed"
		cfg.EmbeddingModelFile = filepath.Join(assetBase, cfg.EmbeddingModelFile)
		cfg.MmprojFile = filepath.Join(assetBase, cfg.MmprojFile)
		cfg.ModelFile = filepath.Join(assetBase, cfg.ModelFile)

		libPath := filepath.Join(assetBase, "llama-lib")
		if dirExists(libPath) {
			cfg.LibraryPath = libPath
		}
	}

	return cfg
}

func needsRepoAssetFallback(cfg Config) bool {
	if filepath.IsAbs(cfg.EmbeddingModelFile) {
		return false
	}
	if fileExists(cfg.EmbeddingModelFile) {
		return false
	}
	return fileExists(filepath.Join("musicembed", cfg.EmbeddingModelFile))
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func dirExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && info.IsDir()
}
