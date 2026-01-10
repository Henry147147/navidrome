package llamacpp

import (
	"context"
	"errors"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type fakeBackend struct {
	initErr   error
	healthErr error
	closed    bool

	audioReqs []AudioEmbedRequest
	descReqs  []AudioDescribeRequest
	textReqs  []TextEmbedRequest

	audioResp []AudioEmbedResponse
	descResp  []AudioDescribeResponse
	textResp  []TextEmbedResponse
}

func (f *fakeBackend) Init(ctx context.Context) error {
	return f.initErr
}

func (f *fakeBackend) Close() error {
	f.closed = true
	return nil
}

func (f *fakeBackend) EmbedAudioBatch(ctx context.Context, reqs []AudioEmbedRequest) ([]AudioEmbedResponse, error) {
	f.audioReqs = append([]AudioEmbedRequest(nil), reqs...)
	return f.audioResp, nil
}

func (f *fakeBackend) DescribeAudioBatch(ctx context.Context, reqs []AudioDescribeRequest) ([]AudioDescribeResponse, error) {
	f.descReqs = append([]AudioDescribeRequest(nil), reqs...)
	return f.descResp, nil
}

func (f *fakeBackend) EmbedTextBatch(ctx context.Context, reqs []TextEmbedRequest) ([]TextEmbedResponse, error) {
	f.textReqs = append([]TextEmbedRequest(nil), reqs...)
	return f.textResp, nil
}

func (f *fakeBackend) HealthCheck(ctx context.Context) error {
	return f.healthErr
}

func TestConfig(t *testing.T) {
	cfg := Config{
		LibraryPath:        "/llama",
		TextModelPath:      "/models/text.gguf",
		AudioModelPath:     "/models/audio.gguf",
		AudioProjectorPath: "/models/audio.mmproj",
		ContextSize:        4096,
		BatchSize:          512,
		UBatchSize:         128,
		Threads:            8,
		ThreadsBatch:       4,
		GPULayers:          40,
		Timeout:            10 * time.Minute,
		MaxRetries:         3,
		RetryBackoff:       2 * time.Second,
	}

	assert.Equal(t, "/llama", cfg.LibraryPath)
	assert.Equal(t, "/models/text.gguf", cfg.TextModelPath)
	assert.Equal(t, "/models/audio.gguf", cfg.AudioModelPath)
	assert.Equal(t, "/models/audio.mmproj", cfg.AudioProjectorPath)
	assert.Equal(t, uint32(4096), cfg.ContextSize)
	assert.Equal(t, uint32(512), cfg.BatchSize)
	assert.Equal(t, uint32(128), cfg.UBatchSize)
	assert.Equal(t, 8, cfg.Threads)
	assert.Equal(t, 4, cfg.ThreadsBatch)
	assert.Equal(t, 40, cfg.GPULayers)
	assert.Equal(t, 10*time.Minute, cfg.Timeout)
	assert.Equal(t, 3, cfg.MaxRetries)
	assert.Equal(t, 2*time.Second, cfg.RetryBackoff)
}

func TestNewClientDefaults(t *testing.T) {
	backend := &fakeBackend{}
	c, err := NewClient(Config{}, WithBackend(backend))
	require.NoError(t, err)
	defer func() { _ = c.Close() }()

	defaults := DefaultConfig()
	assert.Equal(t, defaults.LibraryPath, c.config.LibraryPath)
	assert.Equal(t, defaults.TextModelPath, c.config.TextModelPath)
	assert.Equal(t, defaults.AudioModelPath, c.config.AudioModelPath)
	assert.Equal(t, defaults.AudioProjectorPath, c.config.AudioProjectorPath)
	assert.Equal(t, defaults.Timeout, c.config.Timeout)
	assert.Equal(t, defaults.MaxRetries, c.config.MaxRetries)
	assert.Equal(t, defaults.RetryBackoff, c.config.RetryBackoff)
}

func TestDefaultLibraryPathUsesEnv(t *testing.T) {
	t.Setenv("YZMA_LIB", "relative/lib")
	expected, err := filepath.Abs("relative/lib")
	require.NoError(t, err)

	got := defaultLibraryPath()
	assert.Equal(t, expected, got)
}

func TestApplyDefaultsResolvesLibraryPath(t *testing.T) {
	cfg := Config{LibraryPath: "relative/lib"}
	expected, err := filepath.Abs("relative/lib")
	require.NoError(t, err)

	updated := applyDefaults(cfg)
	assert.Equal(t, expected, updated.LibraryPath)
}

func TestNextBatchSizeRoundsUp(t *testing.T) {
	t.Run("uses min batch", func(t *testing.T) {
		assert.Equal(t, uint32(2048), nextBatchSize(10, 0, 0))
	})

	t.Run("respects configured", func(t *testing.T) {
		assert.Equal(t, uint32(4096), nextBatchSize(3000, 0, 4096))
	})

	t.Run("rounds to step", func(t *testing.T) {
		assert.Equal(t, uint32(2304), nextBatchSize(2201, 0, 0))
	})

	t.Run("keeps current when sufficient", func(t *testing.T) {
		assert.Equal(t, uint32(4096), nextBatchSize(1024, 4096, 0))
	})
}

func TestNextContextSizeRoundsUp(t *testing.T) {
	t.Run("uses min ctx", func(t *testing.T) {
		assert.Equal(t, uint32(2048), nextContextSize(10, 0, 0))
	})

	t.Run("respects configured", func(t *testing.T) {
		assert.Equal(t, uint32(4096), nextContextSize(3000, 0, 4096))
	})

	t.Run("rounds to step", func(t *testing.T) {
		assert.Equal(t, uint32(2304), nextContextSize(2201, 0, 0))
	})

	t.Run("keeps current when sufficient", func(t *testing.T) {
		assert.Equal(t, uint32(4096), nextContextSize(1024, 4096, 0))
	})
}

func TestCountChunkPositionsFallbacksToTokens(t *testing.T) {
	// This test ensures the helper behaves predictably when there are no chunks.
	// It should return 0 without panicking, which is the same behavior as tokens.
	assert.Equal(t, uint32(0), countChunkPositions(0))
}

func TestNewClientWithConfig(t *testing.T) {
	backend := &fakeBackend{}
	cfg := Config{
		LibraryPath:    "/custom/lib",
		TextModelPath:  "/custom/text.gguf",
		AudioModelPath: "/custom/audio.gguf",
		Timeout:        5 * time.Minute,
		MaxRetries:     5,
		RetryBackoff:   3 * time.Second,
	}

	c, err := NewClient(cfg, WithBackend(backend))
	require.NoError(t, err)
	defer func() { _ = c.Close() }()

	assert.Equal(t, "/custom/lib", c.config.LibraryPath)
	assert.Equal(t, "/custom/text.gguf", c.config.TextModelPath)
	assert.Equal(t, "/custom/audio.gguf", c.config.AudioModelPath)
	assert.Equal(t, DefaultAudioProjectorPath, c.config.AudioProjectorPath)
	assert.Equal(t, 5*time.Minute, c.config.Timeout)
	assert.Equal(t, 5, c.config.MaxRetries)
	assert.Equal(t, 3*time.Second, c.config.RetryBackoff)
}

func TestNewClientInitError(t *testing.T) {
	backend := &fakeBackend{initErr: errors.New("boom")}
	_, err := NewClient(Config{}, WithBackend(backend))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "boom")
}

func TestEmbedAudioBatchDelegates(t *testing.T) {
	backend := &fakeBackend{
		audioResp: []AudioEmbedResponse{{Embedding: []float64{0.1, 0.2}, ModelID: "audio"}},
	}
	c, err := NewClient(Config{}, WithBackend(backend))
	require.NoError(t, err)
	defer func() { _ = c.Close() }()

	ctx := context.Background()
	reqs := []AudioEmbedRequest{{AudioPath: "/test.mp3"}}
	resp, err := c.EmbedAudioBatch(ctx, reqs)
	require.NoError(t, err)

	assert.Equal(t, reqs, backend.audioReqs)
	assert.Equal(t, backend.audioResp, resp)
}

func TestDescribeAudioBatchDelegates(t *testing.T) {
	backend := &fakeBackend{
		descResp: []AudioDescribeResponse{{Description: "desc", ModelID: "audio"}},
	}
	c, err := NewClient(Config{}, WithBackend(backend))
	require.NoError(t, err)
	defer func() { _ = c.Close() }()

	ctx := context.Background()
	reqs := []AudioDescribeRequest{{AudioPath: "/test.mp3"}}
	resp, err := c.DescribeAudioBatch(ctx, reqs)
	require.NoError(t, err)

	assert.Equal(t, reqs, backend.descReqs)
	assert.Equal(t, backend.descResp, resp)
}

func TestEmbedTextBatchDelegates(t *testing.T) {
	backend := &fakeBackend{
		textResp: []TextEmbedResponse{{Embedding: []float64{0.1, 0.2}, ModelID: "text"}},
	}
	c, err := NewClient(Config{}, WithBackend(backend))
	require.NoError(t, err)
	defer func() { _ = c.Close() }()

	ctx := context.Background()
	reqs := []TextEmbedRequest{{Text: "hello"}}
	resp, err := c.EmbedTextBatch(ctx, reqs)
	require.NoError(t, err)

	assert.Equal(t, reqs, backend.textReqs)
	assert.Equal(t, backend.textResp, resp)
}

func TestEmbedAudioSingle(t *testing.T) {
	backend := &fakeBackend{
		audioResp: []AudioEmbedResponse{{Embedding: []float64{0.1, 0.2}, ModelID: "audio"}},
	}
	c, err := NewClient(Config{}, WithBackend(backend))
	require.NoError(t, err)
	defer func() { _ = c.Close() }()

	ctx := context.Background()
	resp, err := c.EmbedAudio(ctx, AudioEmbedRequest{AudioPath: "/test.mp3"})
	require.NoError(t, err)
	assert.Equal(t, backend.audioResp[0], *resp)
}

func TestDescribeAudioSingle(t *testing.T) {
	backend := &fakeBackend{
		descResp: []AudioDescribeResponse{{Description: "desc", ModelID: "audio"}},
	}
	c, err := NewClient(Config{}, WithBackend(backend))
	require.NoError(t, err)
	defer func() { _ = c.Close() }()

	ctx := context.Background()
	resp, err := c.DescribeAudio(ctx, AudioDescribeRequest{AudioPath: "/test.mp3"})
	require.NoError(t, err)
	assert.Equal(t, backend.descResp[0], *resp)
}

func TestEmbedTextSingle(t *testing.T) {
	backend := &fakeBackend{
		textResp: []TextEmbedResponse{{Embedding: []float64{0.1, 0.2}, ModelID: "text"}},
	}
	c, err := NewClient(Config{}, WithBackend(backend))
	require.NoError(t, err)
	defer func() { _ = c.Close() }()

	ctx := context.Background()
	resp, err := c.EmbedText(ctx, TextEmbedRequest{Text: "hello"})
	require.NoError(t, err)
	assert.Equal(t, backend.textResp[0], *resp)
}

func TestEmptyBatchesReturnNil(t *testing.T) {
	backend := &fakeBackend{}
	c, err := NewClient(Config{}, WithBackend(backend))
	require.NoError(t, err)
	defer func() { _ = c.Close() }()

	ctx := context.Background()
	audioResp, err := c.EmbedAudioBatch(ctx, nil)
	assert.NoError(t, err)
	assert.Nil(t, audioResp)

	descResp, err := c.DescribeAudioBatch(ctx, []AudioDescribeRequest{})
	assert.NoError(t, err)
	assert.Nil(t, descResp)

	textResp, err := c.EmbedTextBatch(ctx, []TextEmbedRequest{})
	assert.NoError(t, err)
	assert.Nil(t, textResp)
}

func TestHealthCheckDelegates(t *testing.T) {
	backend := &fakeBackend{healthErr: errors.New("unhealthy")}
	c, err := NewClient(Config{}, WithBackend(backend))
	require.NoError(t, err)
	defer func() { _ = c.Close() }()

	err = c.HealthCheck(context.Background())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "unhealthy")
}
