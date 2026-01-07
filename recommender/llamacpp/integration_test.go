//go:build integration
// +build integration

// Integration tests for llama.cpp backend functionality.
//
// NOTE: These tests require the llama.cpp models and libraries to be present.
// Run individual tests rather than the entire suite due to llama.cpp library
// lifecycle constraints - the library can only be initialized/closed once per
// process. Running all tests together will fail because Close() shuts down
// the library globally.
//
// Run individual tests:
//
//	go test -tags=integration -v -run TestIntegration_EmbedTextBatch ./recommender/llamacpp/...
//	go test -tags=integration -v -run TestIntegration_EmbedAudioBatch ./recommender/llamacpp/...
//	go test -tags=integration -v -run TestIntegration_DescribeAudioBatch ./recommender/llamacpp/...
//	go test -tags=integration -v -run TestIntegration_FullPipeline ./recommender/llamacpp/...

package llamacpp

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const (
	testAudioPath = "../../test/music/Beach Bunny - Deadweight.flac"
	testTimeout   = 10 * time.Minute
)

func getProjectRoot() string {
	// Get the directory of the current test file and go up to project root
	wd, _ := os.Getwd()
	return filepath.Join(wd, "../..")
}

func skipIfNoModels(t *testing.T) {
	root := getProjectRoot()
	paths := []string{
		filepath.Join(root, "musicembed/models/qwen-embedder-4b.gguf"),
		filepath.Join(root, "musicembed/models/music-flamingo.gguf"),
		filepath.Join(root, "musicembed/models/mmproj-music-flamingo.gguf"),
	}
	for _, p := range paths {
		if _, err := os.Stat(p); os.IsNotExist(err) {
			t.Skipf("Model not found: %s", p)
		}
	}
}

func skipIfNoLlamaLib(t *testing.T) {
	root := getProjectRoot()
	libPath := filepath.Join(root, "musicembed/llama-lib")
	if _, err := os.Stat(libPath); os.IsNotExist(err) {
		t.Skipf("Llama library not found: %s", libPath)
	}
}

func skipIfNoTestAudio(t *testing.T) {
	root := getProjectRoot()
	audioPath := filepath.Join(root, "test/music/Beach Bunny - Deadweight.flac")
	if _, err := os.Stat(audioPath); os.IsNotExist(err) {
		t.Skipf("Test audio not found: %s", audioPath)
	}
}

func TestIntegration_EmbedTextBatch(t *testing.T) {
	skipIfNoModels(t)
	skipIfNoLlamaLib(t)

	root := getProjectRoot()
	cfg := Config{
		TextModelPath: filepath.Join(root, "musicembed/models/qwen-embedder-4b.gguf"),
		LibraryPath:   filepath.Join(root, "musicembed/llama-lib"),
		Timeout:       testTimeout,
		GPULayers:     99, // Offload all layers to GPU if available
	}

	client, err := NewClient(cfg)
	require.NoError(t, err)
	defer client.Close()

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	reqs := []TextEmbedRequest{
		{Text: "An upbeat pop song with catchy hooks and electronic beats"},
		{Text: "A melancholic acoustic ballad with soft vocals"},
	}

	responses, err := client.EmbedTextBatch(ctx, reqs)
	require.NoError(t, err)
	require.Len(t, responses, 2)

	for i, resp := range responses {
		assert.Empty(t, resp.Error, "Response %d should not have error: %s", i, resp.Error)
		assert.NotEmpty(t, resp.Embedding, "Response %d should have embedding", i)
		assert.Equal(t, "qwen", resp.ModelID)
		assert.Greater(t, resp.Dimension, 0, "Dimension should be positive")

		// Verify embeddings are not all zeros
		var sum float64
		for _, v := range resp.Embedding {
			sum += v * v
		}
		assert.Greater(t, sum, 0.0, "Embedding should not be zero vector")
	}

	t.Logf("Text embedding dimension: %d", responses[0].Dimension)
}

func TestIntegration_EmbedAudioBatch(t *testing.T) {
	skipIfNoModels(t)
	skipIfNoLlamaLib(t)
	skipIfNoTestAudio(t)

	root := getProjectRoot()
	cfg := Config{
		AudioModelPath:     filepath.Join(root, "musicembed/models/music-flamingo.gguf"),
		AudioProjectorPath: filepath.Join(root, "musicembed/models/mmproj-music-flamingo.gguf"),
		LibraryPath:        filepath.Join(root, "musicembed/llama-lib"),
		Timeout:            testTimeout,
		GPULayers:          99,
		ContextSize:        8192, // Audio processing needs larger context
		BatchSize:          2048,
	}

	client, err := NewClient(cfg)
	require.NoError(t, err)
	defer client.Close()

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	absPath := filepath.Join(root, "test/music/Beach Bunny - Deadweight.flac")

	reqs := []AudioEmbedRequest{
		{AudioPath: absPath},
	}

	responses, err := client.EmbedAudioBatch(ctx, reqs)
	require.NoError(t, err)
	require.Len(t, responses, 1)

	resp := responses[0]
	assert.Empty(t, resp.Error, "Response should not have error: %s", resp.Error)
	assert.NotEmpty(t, resp.Embedding, "Should have embedding")
	assert.Equal(t, "flamingo", resp.ModelID)
	assert.Greater(t, resp.Duration, 0.0, "Duration should be positive")

	// Verify embeddings are not all zeros
	var sum float64
	for _, v := range resp.Embedding {
		sum += v * v
	}
	assert.Greater(t, sum, 0.0, "Embedding should not be zero vector")

	t.Logf("Audio embedding dimension: %d, duration: %.2fs", len(resp.Embedding), resp.Duration)
}

func TestIntegration_DescribeAudioBatch(t *testing.T) {
	skipIfNoModels(t)
	skipIfNoLlamaLib(t)
	skipIfNoTestAudio(t)

	root := getProjectRoot()
	cfg := Config{
		AudioModelPath:     filepath.Join(root, "musicembed/models/music-flamingo.gguf"),
		AudioProjectorPath: filepath.Join(root, "musicembed/models/mmproj-music-flamingo.gguf"),
		LibraryPath:        filepath.Join(root, "musicembed/llama-lib"),
		Timeout:            testTimeout,
		GPULayers:          99,
		ContextSize:        8192, // Audio processing needs larger context
		BatchSize:          2048,
	}

	client, err := NewClient(cfg)
	require.NoError(t, err)
	defer client.Close()

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	absPath := filepath.Join(root, "test/music/Beach Bunny - Deadweight.flac")

	reqs := []AudioDescribeRequest{
		{AudioPath: absPath},
	}

	responses, err := client.DescribeAudioBatch(ctx, reqs)
	require.NoError(t, err)
	require.Len(t, responses, 1)

	resp := responses[0]
	assert.Empty(t, resp.Error, "Response should not have error: %s", resp.Error)
	assert.NotEmpty(t, resp.Description, "Description should not be empty")
	assert.Equal(t, "flamingo", resp.ModelID)

	// Description should contain some relevant content
	desc := strings.ToLower(resp.Description)
	hasRelevantContent := strings.Contains(desc, "music") ||
		strings.Contains(desc, "song") ||
		strings.Contains(desc, "audio") ||
		strings.Contains(desc, "genre") ||
		strings.Contains(desc, "tempo") ||
		strings.Contains(desc, "vocal") ||
		strings.Contains(desc, "guitar") ||
		strings.Contains(desc, "drum") ||
		strings.Contains(desc, "lyrics") ||
		strings.Contains(desc, "description")
	assert.True(t, hasRelevantContent, "Description should contain relevant content: %s", resp.Description)

	t.Logf("Generated description (%d chars):\n%s", len(resp.Description), resp.Description)
}

func TestIntegration_FullPipeline(t *testing.T) {
	skipIfNoModels(t)
	skipIfNoLlamaLib(t)
	skipIfNoTestAudio(t)

	root := getProjectRoot()
	cfg := Config{
		TextModelPath:      filepath.Join(root, "musicembed/models/qwen-embedder-4b.gguf"),
		AudioModelPath:     filepath.Join(root, "musicembed/models/music-flamingo.gguf"),
		AudioProjectorPath: filepath.Join(root, "musicembed/models/mmproj-music-flamingo.gguf"),
		LibraryPath:        filepath.Join(root, "musicembed/llama-lib"),
		Timeout:            testTimeout,
		GPULayers:          99,
		ContextSize:        8192, // Audio processing needs larger context
		BatchSize:          2048,
	}

	client, err := NewClient(cfg)
	require.NoError(t, err)
	defer client.Close()

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	absPath := filepath.Join(root, "test/music/Beach Bunny - Deadweight.flac")

	// Step 1: Get audio description
	t.Log("Step 1: Generating audio description...")
	descResp, err := client.DescribeAudio(ctx, AudioDescribeRequest{AudioPath: absPath})
	require.NoError(t, err)
	require.Empty(t, descResp.Error, "Description error: %s", descResp.Error)
	t.Logf("Description: %s", descResp.Description[:min(200, len(descResp.Description))])

	// Step 2: Embed the description
	t.Log("Step 2: Embedding description text...")
	textResp, err := client.EmbedText(ctx, TextEmbedRequest{Text: descResp.Description})
	require.NoError(t, err)
	require.Empty(t, textResp.Error, "Text embed error: %s", textResp.Error)
	t.Logf("Description embedding dimension: %d", len(textResp.Embedding))

	// Step 3: Get audio embedding
	t.Log("Step 3: Generating audio embedding...")
	audioResp, err := client.EmbedAudio(ctx, AudioEmbedRequest{AudioPath: absPath})
	require.NoError(t, err)
	require.Empty(t, audioResp.Error, "Audio embed error: %s", audioResp.Error)
	t.Logf("Audio embedding dimension: %d", len(audioResp.Embedding))

	// Verify all outputs have reasonable dimensions
	assert.NotEmpty(t, textResp.Embedding, "Description embedding should not be empty")
	assert.NotEmpty(t, audioResp.Embedding, "Audio embedding should not be empty")

	t.Log("Full pipeline completed successfully!")
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
