//go:build integration
// +build integration

package embedder

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"test/llama/musicembed"

	"github.com/navidrome/navidrome/recommender/milvus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const integrationTimeout = 10 * time.Minute

func TestIntegration_MusicEmbedPipeline(t *testing.T) {
	root := projectRoot(t)

	llamaLib := filepath.Join(root, "musicembed/llama-lib")
	embedModel := filepath.Join(root, "musicembed/models/qwen-embedder-4b.gguf")
	mmproj := filepath.Join(root, "musicembed/models/mmproj-music-flamingo.gguf")
	musicModel := filepath.Join(root, "musicembed/models/music-flamingo.gguf")
	audio := filepath.Join(root, "test/music/Beach Bunny - Deadweight.flac")

	skipIfMissing(t, llamaLib)
	skipIfMissing(t, embedModel)
	skipIfMissing(t, mmproj)
	skipIfMissing(t, musicModel)
	skipIfMissing(t, audio)

	cfg := musicembed.DefaultConfig()
	cfg.LibraryPath = llamaLib
	cfg.EmbeddingModelFile = embedModel
	cfg.MmprojFile = mmproj
	cfg.ModelFile = musicModel
	cfg.GPULayers = 0
	cfg.UseGPU = false
	cfg.Threads = runtime.NumCPU()
	cfg.MaxOutputTokens = 512

	client, err := musicembed.New(cfg)
	require.NoError(t, err)
	defer client.Close()

	store := &fakeVectorStore{}
	embedder := New(Config{EnableDescription: true, EnableFlamingo: true}, client, store)

	ctx, cancel := context.WithTimeout(context.Background(), integrationTimeout)
	defer cancel()

	result, err := embedder.EmbedAudio(ctx, EmbedRequest{FilePath: audio, TrackName: "Integration Track"})
	require.NoError(t, err)

	assert.NotEmpty(t, result.Description)
	assert.NotEmpty(t, result.DescriptionEmbedding)
	assert.NotEmpty(t, result.FlamingoEmbedding)

	require.Len(t, store.upserts, 2)
	assert.Equal(t, milvus.CollectionDescription, store.upserts[0].collection)
	assert.Equal(t, milvus.CollectionFlamingo, store.upserts[1].collection)
}

func projectRoot(t *testing.T) string {
	wd, err := os.Getwd()
	require.NoError(t, err)

	root := filepath.Clean(filepath.Join(wd, "..", ".."))
	return root
}

func skipIfMissing(t *testing.T, path string) {
	if _, err := os.Stat(path); err != nil {
		t.Skipf("missing integration dependency: %s", path)
	}
}
