package llamacpp

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestConfig(t *testing.T) {
	cfg := Config{
		AudioEmbedURL:    "http://localhost:8080/embed/audio",
		AudioDescribeURL: "http://localhost:8081/describe",
		TextEmbedURL:     "http://localhost:8082/embed/text",
		Timeout:          10 * time.Minute,
		MaxRetries:       3,
		RetryBackoff:     2 * time.Second,
	}

	assert.Equal(t, "http://localhost:8080/embed/audio", cfg.AudioEmbedURL)
	assert.Equal(t, "http://localhost:8081/describe", cfg.AudioDescribeURL)
	assert.Equal(t, "http://localhost:8082/embed/text", cfg.TextEmbedURL)
	assert.Equal(t, 10*time.Minute, cfg.Timeout)
	assert.Equal(t, 3, cfg.MaxRetries)
	assert.Equal(t, 2*time.Second, cfg.RetryBackoff)
}

func TestNewClientDefaults(t *testing.T) {
	cfg := Config{} // Zero values
	c := NewClient(cfg)

	assert.NotNil(t, c)
	assert.Equal(t, "http://localhost:8080/embed/audio", c.config.AudioEmbedURL)
	assert.Equal(t, "http://localhost:8081/describe", c.config.AudioDescribeURL)
	assert.Equal(t, "http://localhost:8082/embed/text", c.config.TextEmbedURL)
	assert.Equal(t, 10*time.Minute, c.config.Timeout)
	assert.Equal(t, 3, c.config.MaxRetries)
	assert.Equal(t, 2*time.Second, c.config.RetryBackoff)
}

func TestNewClientWithConfig(t *testing.T) {
	cfg := Config{
		AudioEmbedURL:    "http://custom:8080/embed/audio",
		AudioDescribeURL: "http://custom:8081/describe",
		TextEmbedURL:     "http://custom:8082/embed/text",
		Timeout:          5 * time.Minute,
		MaxRetries:       5,
		RetryBackoff:     3 * time.Second,
	}
	c := NewClient(cfg)

	assert.NotNil(t, c)
	assert.Equal(t, "http://custom:8080/embed/audio", c.config.AudioEmbedURL)
	assert.Equal(t, "http://custom:8081/describe", c.config.AudioDescribeURL)
	assert.Equal(t, "http://custom:8082/embed/text", c.config.TextEmbedURL)
	assert.Equal(t, 5*time.Minute, c.config.Timeout)
	assert.Equal(t, 5, c.config.MaxRetries)
	assert.Equal(t, 3*time.Second, c.config.RetryBackoff)
}

func TestAudioEmbedRequest(t *testing.T) {
	req := AudioEmbedRequest{
		AudioPath:  "/path/to/audio.mp3",
		SampleRate: 48000,
		BatchID:    "batch123",
	}

	assert.Equal(t, "/path/to/audio.mp3", req.AudioPath)
	assert.Equal(t, 48000, req.SampleRate)
	assert.Equal(t, "batch123", req.BatchID)
}

func TestAudioEmbedResponse(t *testing.T) {
	resp := AudioEmbedResponse{
		Embedding: []float64{0.1, 0.2, 0.3},
		ModelID:   "flamingo",
		Duration:  180.5,
		Error:     "",
	}

	assert.Equal(t, []float64{0.1, 0.2, 0.3}, resp.Embedding)
	assert.Equal(t, "flamingo", resp.ModelID)
	assert.Equal(t, 180.5, resp.Duration)
	assert.Empty(t, resp.Error)
}

func TestAudioDescribeRequest(t *testing.T) {
	req := AudioDescribeRequest{
		AudioPath: "/path/to/audio.mp3",
		Prompt:    "Describe this music",
	}

	assert.Equal(t, "/path/to/audio.mp3", req.AudioPath)
	assert.Equal(t, "Describe this music", req.Prompt)
}

func TestAudioDescribeResponse(t *testing.T) {
	resp := AudioDescribeResponse{
		Description:    "An upbeat electronic track",
		AudioEmbedding: []float64{0.1, 0.2, 0.3},
		ModelID:        "qwen",
		Error:          "",
	}

	assert.Equal(t, "An upbeat electronic track", resp.Description)
	assert.Equal(t, []float64{0.1, 0.2, 0.3}, resp.AudioEmbedding)
	assert.Equal(t, "qwen", resp.ModelID)
	assert.Empty(t, resp.Error)
}

func TestTextEmbedRequest(t *testing.T) {
	req := TextEmbedRequest{
		Text:    "Some text to embed",
		ModelID: "lyrics",
	}

	assert.Equal(t, "Some text to embed", req.Text)
	assert.Equal(t, "lyrics", req.ModelID)
}

func TestTextEmbedResponse(t *testing.T) {
	resp := TextEmbedResponse{
		Embedding: []float64{0.1, 0.2, 0.3},
		ModelID:   "lyrics",
		Dimension: 4096,
		Error:     "",
	}

	assert.Equal(t, []float64{0.1, 0.2, 0.3}, resp.Embedding)
	assert.Equal(t, "lyrics", resp.ModelID)
	assert.Equal(t, 4096, resp.Dimension)
	assert.Empty(t, resp.Error)
}

func TestEmbedAudioBatchStub(t *testing.T) {
	c := NewClient(Config{})
	ctx := context.Background()

	// Empty batch should return nil
	result, err := c.EmbedAudioBatch(ctx, nil)
	assert.NoError(t, err)
	assert.Nil(t, result)

	result, err = c.EmbedAudioBatch(ctx, []AudioEmbedRequest{})
	assert.NoError(t, err)
	assert.Nil(t, result)

	// Non-empty batch should return stub error
	_, err = c.EmbedAudioBatch(ctx, []AudioEmbedRequest{{AudioPath: "/test.mp3"}})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not yet implemented")
}

func TestDescribeAudioBatchStub(t *testing.T) {
	c := NewClient(Config{})
	ctx := context.Background()

	// Empty batch should return nil
	result, err := c.DescribeAudioBatch(ctx, nil)
	assert.NoError(t, err)
	assert.Nil(t, result)

	result, err = c.DescribeAudioBatch(ctx, []AudioDescribeRequest{})
	assert.NoError(t, err)
	assert.Nil(t, result)

	// Non-empty batch should return stub error
	_, err = c.DescribeAudioBatch(ctx, []AudioDescribeRequest{{AudioPath: "/test.mp3"}})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not yet implemented")
}

func TestEmbedTextBatchStub(t *testing.T) {
	c := NewClient(Config{})
	ctx := context.Background()

	// Empty batch should return nil
	result, err := c.EmbedTextBatch(ctx, nil)
	assert.NoError(t, err)
	assert.Nil(t, result)

	result, err = c.EmbedTextBatch(ctx, []TextEmbedRequest{})
	assert.NoError(t, err)
	assert.Nil(t, result)

	// Non-empty batch should return stub error
	_, err = c.EmbedTextBatch(ctx, []TextEmbedRequest{{Text: "test"}})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not yet implemented")
}

func TestEmbedAudioStub(t *testing.T) {
	c := NewClient(Config{})
	ctx := context.Background()

	// Should return stub error
	_, err := c.EmbedAudio(ctx, AudioEmbedRequest{AudioPath: "/test.mp3"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not yet implemented")
}

func TestDescribeAudioStub(t *testing.T) {
	c := NewClient(Config{})
	ctx := context.Background()

	// Should return stub error
	_, err := c.DescribeAudio(ctx, AudioDescribeRequest{AudioPath: "/test.mp3"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not yet implemented")
}

func TestEmbedTextStub(t *testing.T) {
	c := NewClient(Config{})
	ctx := context.Background()

	// Should return stub error
	_, err := c.EmbedText(ctx, TextEmbedRequest{Text: "test"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not yet implemented")
}
