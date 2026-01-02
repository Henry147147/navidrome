package embedder

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/navidrome/navidrome/recommender/milvus"
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

func TestPipelineConfigDefaults(t *testing.T) {
	cfg := PipelineConfig{}

	assert.Equal(t, time.Duration(0), cfg.BatchTimeout)
	assert.Equal(t, 0, cfg.BatchSize)
	assert.False(t, cfg.EnableLyrics)
	assert.False(t, cfg.EnableDescription)
	assert.False(t, cfg.EnableFlamingo)
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

func TestTrackContextWithEmbeddings(t *testing.T) {
	tc := &TrackContext{
		FilePath:             "/path/to/audio.mp3",
		TrackName:            "Test Track",
		LyricsEmbedding:      []float64{0.1, 0.2, 0.3},
		Description:          "An upbeat pop song",
		DescriptionEmbedding: []float64{0.4, 0.5, 0.6},
		FlamingoEmbedding:    []float64{0.7, 0.8, 0.9},
		Done:                 make(chan struct{}),
	}

	assert.Equal(t, []float64{0.1, 0.2, 0.3}, tc.LyricsEmbedding)
	assert.Equal(t, "An upbeat pop song", tc.Description)
	assert.Equal(t, []float64{0.4, 0.5, 0.6}, tc.DescriptionEmbedding)
	assert.Equal(t, []float64{0.7, 0.8, 0.9}, tc.FlamingoEmbedding)
}

func TestTrackContextWithError(t *testing.T) {
	tc := &TrackContext{
		FilePath: "/path/to/audio.mp3",
		Error:    assert.AnError,
		Done:     make(chan struct{}),
	}

	assert.Error(t, tc.Error)
}

func TestNewPipelineDefaults(t *testing.T) {
	cfg := PipelineConfig{} // Zero values
	p := NewPipeline(cfg, nil, nil)

	assert.NotNil(t, p)
	assert.Equal(t, 5*time.Second, p.config.BatchTimeout)
	assert.Equal(t, 50, p.config.BatchSize)
	assert.False(t, p.closed)
	assert.Empty(t, p.pending)
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

func TestPipelineEnqueueAddsToQueue(t *testing.T) {
	p := NewPipeline(PipelineConfig{
		BatchTimeout: 1 * time.Hour,
		BatchSize:    100,
	}, nil, nil)
	defer p.Close()

	ctx := context.Background()

	track1 := &TrackContext{FilePath: "/test1.mp3", Done: make(chan struct{})}
	track2 := &TrackContext{FilePath: "/test2.mp3", Done: make(chan struct{})}

	err := p.Enqueue(ctx, track1)
	require.NoError(t, err)

	err = p.Enqueue(ctx, track2)
	require.NoError(t, err)

	p.mu.Lock()
	pendingCount := len(p.pending)
	p.mu.Unlock()

	assert.Equal(t, 2, pendingCount)
}

func TestPipelineEnqueueStartsTimer(t *testing.T) {
	p := NewPipeline(PipelineConfig{
		BatchTimeout: 1 * time.Hour,
		BatchSize:    100,
	}, nil, nil)
	defer p.Close()

	ctx := context.Background()

	// No timer initially
	p.mu.Lock()
	assert.Nil(t, p.timer)
	p.mu.Unlock()

	track := &TrackContext{FilePath: "/test.mp3", Done: make(chan struct{})}
	err := p.Enqueue(ctx, track)
	require.NoError(t, err)

	// Timer should be set after first enqueue
	p.mu.Lock()
	assert.NotNil(t, p.timer)
	p.mu.Unlock()
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

func TestPipelineCloseMultipleTracks(t *testing.T) {
	p := NewPipeline(PipelineConfig{
		BatchTimeout: 1 * time.Hour,
		BatchSize:    100,
	}, nil, nil)

	ctx := context.Background()
	tracks := make([]*TrackContext, 5)
	for i := range tracks {
		tracks[i] = &TrackContext{
			FilePath: "/test.mp3",
			Done:     make(chan struct{}),
		}
		err := p.Enqueue(ctx, tracks[i])
		require.NoError(t, err)
	}

	p.Close()

	// All tracks should be marked as cancelled
	for i, track := range tracks {
		select {
		case <-track.Done:
			assert.Error(t, track.Error, "Track %d should have error", i)
		case <-time.After(100 * time.Millisecond):
			t.Fatalf("Track %d was not cancelled", i)
		}
	}
}

func TestPipelineCloseIdempotent(t *testing.T) {
	p := NewPipeline(PipelineConfig{}, nil, nil)

	// Close multiple times should not panic
	p.Close()
	p.Close()
	p.Close()

	assert.True(t, p.closed)
}

func TestPipelineFlush(t *testing.T) {
	p := NewPipeline(PipelineConfig{
		BatchTimeout: 1 * time.Hour, // Long timeout so we can test flush
		BatchSize:    100,
	}, nil, nil)
	defer p.Close()

	ctx := context.Background()

	// Flush should complete immediately when no pending items
	err := p.Flush(ctx)
	assert.NoError(t, err)
}

func TestPipelineFlushWaitsForProcessing(t *testing.T) {
	// This test verifies that Flush waits for the waitgroup
	p := NewPipeline(PipelineConfig{
		BatchTimeout: 1 * time.Hour,
		BatchSize:    100,
	}, nil, nil)
	defer p.Close()

	ctx := context.Background()

	// Add some processing work manually
	p.wg.Add(1)
	go func() {
		time.Sleep(50 * time.Millisecond)
		p.wg.Done()
	}()

	start := time.Now()
	err := p.Flush(ctx)
	elapsed := time.Since(start)

	assert.NoError(t, err)
	// Should have waited at least 50ms for the work to complete
	assert.GreaterOrEqual(t, elapsed.Milliseconds(), int64(40))
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

func TestEmbeddingDataOptionalFields(t *testing.T) {
	// Offset and Description are optional
	data := EmbeddingData{
		Name:      "Track",
		Embedding: []float64{0.1},
		ModelID:   ModelFlamingo,
	}

	assert.Equal(t, 0.0, data.Offset)
	assert.Empty(t, data.Description)
}

func TestToMilvusData(t *testing.T) {
	input := []EmbeddingData{
		{
			Name:        "Track1",
			Embedding:   []float64{0.1, 0.2},
			Offset:      0.5,
			ModelID:     ModelLyrics,
			Description: "Description 1",
		},
		{
			Name:      "Track2",
			Embedding: []float64{0.3, 0.4},
			ModelID:   ModelFlamingo,
		},
	}

	result := toMilvusData(input)

	require.Len(t, result, 2)

	assert.Equal(t, "Track1", result[0].Name)
	assert.Equal(t, []float64{0.1, 0.2}, result[0].Embedding)
	assert.Equal(t, 0.5, result[0].Offset)
	assert.Equal(t, ModelLyrics, result[0].ModelID)
	assert.Equal(t, "Description 1", result[0].Description)

	assert.Equal(t, "Track2", result[1].Name)
	assert.Equal(t, []float64{0.3, 0.4}, result[1].Embedding)
	assert.Equal(t, 0.0, result[1].Offset)
	assert.Equal(t, ModelFlamingo, result[1].ModelID)
	assert.Empty(t, result[1].Description)
}

func TestToMilvusDataEmpty(t *testing.T) {
	result := toMilvusData([]EmbeddingData{})
	assert.Len(t, result, 0)
}

func TestToMilvusDataPreservesOrder(t *testing.T) {
	input := make([]EmbeddingData, 100)
	for i := range input {
		input[i] = EmbeddingData{
			Name:      string(rune('A' + i%26)),
			Embedding: []float64{float64(i)},
			ModelID:   ModelLyrics,
		}
	}

	result := toMilvusData(input)

	for i := range input {
		assert.Equal(t, input[i].Name, result[i].Name)
		assert.Equal(t, input[i].Embedding, result[i].Embedding)
	}
}

func TestToMilvusDataTypeConversion(t *testing.T) {
	input := []EmbeddingData{
		{
			Name:        "Track",
			Embedding:   []float64{0.123456789, 0.987654321},
			Offset:      0.555,
			ModelID:     "test",
			Description: "Test description",
		},
	}

	result := toMilvusData(input)

	// Verify all fields are correctly converted to milvus.EmbeddingData
	var _ milvus.EmbeddingData = result[0] // Compile-time type check
	assert.IsType(t, milvus.EmbeddingData{}, result[0])
}

func TestPipelineConcurrentEnqueue(t *testing.T) {
	p := NewPipeline(PipelineConfig{
		BatchTimeout: 1 * time.Hour,
		BatchSize:    1000,
	}, nil, nil)
	defer p.Close()

	ctx := context.Background()
	numGoroutines := 10
	tracksPerGoroutine := 10

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		go func() {
			defer wg.Done()
			for j := 0; j < tracksPerGoroutine; j++ {
				track := &TrackContext{
					FilePath: "/test.mp3",
					Done:     make(chan struct{}),
				}
				_ = p.Enqueue(ctx, track)
			}
		}()
	}

	wg.Wait()

	p.mu.Lock()
	pendingCount := len(p.pending)
	p.mu.Unlock()

	assert.Equal(t, numGoroutines*tracksPerGoroutine, pendingCount)
}

func TestPipelineBatchSizeTrigger(t *testing.T) {
	batchSize := 5
	p := NewPipeline(PipelineConfig{
		BatchTimeout: 1 * time.Hour, // Long timeout
		BatchSize:    batchSize,
	}, nil, nil)
	defer p.Close()

	ctx := context.Background()
	tracks := make([]*TrackContext, batchSize)

	// Enqueue exactly batchSize-1 items (shouldn't trigger)
	for i := 0; i < batchSize-1; i++ {
		tracks[i] = &TrackContext{
			FilePath: "/test.mp3",
			Done:     make(chan struct{}),
		}
		err := p.Enqueue(ctx, tracks[i])
		require.NoError(t, err)
	}

	p.mu.Lock()
	notProcessingYet := !p.processing
	pendingCount := len(p.pending)
	p.mu.Unlock()

	assert.True(t, notProcessingYet, "Should not start processing before batch size reached")
	assert.Equal(t, batchSize-1, pendingCount)
}

func TestTrackContextDoneChannelClosed(t *testing.T) {
	tc := &TrackContext{
		FilePath: "/test.mp3",
		Done:     make(chan struct{}),
	}

	// Simulate completion
	close(tc.Done)

	select {
	case <-tc.Done:
		// Expected - channel is closed
	default:
		t.Fatal("Done channel should be closed")
	}
}

func TestTrackContextWithAllFields(t *testing.T) {
	tc := &TrackContext{
		FilePath:             "/music/artist/album/track.flac",
		TrackName:            "Artist - Track",
		TrackID:              "uuid-1234-5678",
		Artist:               "Test Artist",
		Title:                "Test Track",
		Album:                "Test Album",
		Lyrics:               "These are the lyrics\nWith multiple lines",
		LyricsEmbedding:      make([]float64, 4096),
		Description:          "An energetic rock song with heavy guitar riffs",
		DescriptionEmbedding: make([]float64, 4096),
		FlamingoEmbedding:    make([]float64, 1024),
		Error:                nil,
		Done:                 make(chan struct{}),
	}

	assert.NotEmpty(t, tc.FilePath)
	assert.NotEmpty(t, tc.TrackName)
	assert.NotEmpty(t, tc.TrackID)
	assert.NotEmpty(t, tc.Lyrics)
	assert.Len(t, tc.LyricsEmbedding, 4096)
	assert.Len(t, tc.DescriptionEmbedding, 4096)
	assert.Len(t, tc.FlamingoEmbedding, 1024)
	assert.NoError(t, tc.Error)
}

func TestPipelineTimerStoppedOnBatchSizeTrigger(t *testing.T) {
	p := NewPipeline(PipelineConfig{
		BatchTimeout: 1 * time.Hour,
		BatchSize:    2,
	}, nil, nil)
	defer p.Close()

	ctx := context.Background()

	// First enqueue starts timer
	track1 := &TrackContext{FilePath: "/test1.mp3", Done: make(chan struct{})}
	err := p.Enqueue(ctx, track1)
	require.NoError(t, err)

	p.mu.Lock()
	hasTimer := p.timer != nil
	p.mu.Unlock()
	assert.True(t, hasTimer, "Timer should be set after first enqueue")

	// Second enqueue triggers batch (since batchSize=2)
	track2 := &TrackContext{FilePath: "/test2.mp3", Done: make(chan struct{})}
	err = p.Enqueue(ctx, track2)
	require.NoError(t, err)

	// Wait a bit for processing to start
	time.Sleep(10 * time.Millisecond)

	p.mu.Lock()
	timerAfter := p.timer
	p.mu.Unlock()
	// Timer should be nil after batch triggered
	assert.Nil(t, timerAfter, "Timer should be nil after batch triggered")
}
