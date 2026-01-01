package embedder

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestPipelineConfig(t *testing.T) {
	cfg := PipelineConfig{
		BatchTimeout:      10 * time.Second,
		BatchSize:         100,
		EnableLyrics:      true,
		EnableDescription: true,
		EnableFlamingo:    true,
	}

	assert.Equal(t, 10*time.Second, cfg.BatchTimeout)
	assert.Equal(t, 100, cfg.BatchSize)
	assert.True(t, cfg.EnableLyrics)
	assert.True(t, cfg.EnableDescription)
	assert.True(t, cfg.EnableFlamingo)
}

func TestTrackContext(t *testing.T) {
	tc := &TrackContext{
		FilePath:  "/path/to/audio.mp3",
		TrackName: "Test Track",
		TrackID:   "track123",
		Artist:    "Test Artist",
		Title:     "Test Title",
		Album:     "Test Album",
		Lyrics:    "Some lyrics here",
		Done:      make(chan struct{}),
	}

	assert.Equal(t, "/path/to/audio.mp3", tc.FilePath)
	assert.Equal(t, "Test Track", tc.TrackName)
	assert.Equal(t, "track123", tc.TrackID)
	assert.Equal(t, "Test Artist", tc.Artist)
	assert.Equal(t, "Test Title", tc.Title)
	assert.Equal(t, "Test Album", tc.Album)
	assert.Equal(t, "Some lyrics here", tc.Lyrics)
	assert.NotNil(t, tc.Done)
}

func TestNewPipelineDefaults(t *testing.T) {
	cfg := PipelineConfig{} // Zero values
	p := NewPipeline(cfg, nil, nil)

	assert.NotNil(t, p)
	assert.Equal(t, 5*time.Second, p.config.BatchTimeout)
	assert.Equal(t, 50, p.config.BatchSize)
}

func TestNewPipelineWithConfig(t *testing.T) {
	cfg := PipelineConfig{
		BatchTimeout:      10 * time.Second,
		BatchSize:         100,
		EnableLyrics:      true,
		EnableDescription: true,
		EnableFlamingo:    false,
	}
	p := NewPipeline(cfg, nil, nil)

	assert.NotNil(t, p)
	assert.Equal(t, 10*time.Second, p.config.BatchTimeout)
	assert.Equal(t, 100, p.config.BatchSize)
	assert.True(t, p.config.EnableLyrics)
	assert.True(t, p.config.EnableDescription)
	assert.False(t, p.config.EnableFlamingo)
}

func TestPipelineEnqueueWhenClosed(t *testing.T) {
	p := NewPipeline(PipelineConfig{}, nil, nil)
	p.Close()

	ctx := context.Background()
	err := p.Enqueue(ctx, &TrackContext{Done: make(chan struct{})})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "closed")
}

func TestPipelineClose(t *testing.T) {
	p := NewPipeline(PipelineConfig{}, nil, nil)

	// Add a pending track
	track := &TrackContext{
		FilePath: "/test.mp3",
		Done:     make(chan struct{}),
	}
	ctx := context.Background()
	err := p.Enqueue(ctx, track)
	require.NoError(t, err)

	// Close the pipeline
	p.Close()

	// Verify the track was marked as cancelled
	select {
	case <-track.Done:
		// Good - track was cancelled
		assert.Error(t, track.Error)
		assert.Contains(t, track.Error.Error(), "closed")
	case <-time.After(100 * time.Millisecond):
		t.Fatal("Track was not cancelled")
	}

	// Verify pipeline is closed
	assert.True(t, p.closed)
	assert.Nil(t, p.pending)
}

func TestPipelineFlush(t *testing.T) {
	p := NewPipeline(PipelineConfig{
		BatchTimeout: 1 * time.Hour, // Long timeout so we can test flush
		BatchSize:    100,
	}, nil, nil)

	ctx := context.Background()

	// Flush should complete immediately when no pending items
	err := p.Flush(ctx)
	assert.NoError(t, err)

	// Close for cleanup
	p.Close()
}

func TestModelConstants(t *testing.T) {
	assert.Equal(t, "lyrics", ModelLyrics)
	assert.Equal(t, "description", ModelDescription)
	assert.Equal(t, "flamingo", ModelFlamingo)
}

func TestEmbeddingData(t *testing.T) {
	data := EmbeddingData{
		Name:        "Test Track",
		Embedding:   []float64{0.1, 0.2, 0.3},
		Offset:      0.5,
		ModelID:     ModelLyrics,
		Description: "A test description",
	}

	assert.Equal(t, "Test Track", data.Name)
	assert.Equal(t, []float64{0.1, 0.2, 0.3}, data.Embedding)
	assert.Equal(t, 0.5, data.Offset)
	assert.Equal(t, ModelLyrics, data.ModelID)
	assert.Equal(t, "A test description", data.Description)
}
