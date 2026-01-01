package embedder

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCanonicalName(t *testing.T) {
	tests := []struct {
		name     string
		artist   string
		title    string
		expected string
	}{
		{"both artist and title", "The Beatles", "Hey Jude", "The Beatles - Hey Jude"},
		{"only artist", "The Beatles", "", "The Beatles"},
		{"only title", "", "Hey Jude", "Hey Jude"},
		{"neither", "", "", ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := canonicalName(tt.artist, tt.title)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestBuildPossibleNames(t *testing.T) {
	tests := []struct {
		name     string
		req      StatusRequest
		expected []string
	}{
		{
			"all fields",
			StatusRequest{
				TrackID:        "track123",
				Artist:         "Artist",
				Title:          "Title",
				AlternateNames: []string{"alt1", "alt2"},
			},
			[]string{"Artist - Title", "track123", "alt1", "alt2"},
		},
		{
			"only trackID",
			StatusRequest{
				TrackID: "track123",
			},
			[]string{"track123"},
		},
		{
			"no fields",
			StatusRequest{},
			[]string{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := buildPossibleNames(tt.req)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestConfig(t *testing.T) {
	cfg := Config{
		BatchTimeout:      5 * time.Second,
		BatchSize:         50,
		EnableLyrics:      true,
		EnableDescription: true,
		EnableFlamingo:    true,
	}

	assert.Equal(t, 5*time.Second, cfg.BatchTimeout)
	assert.Equal(t, 50, cfg.BatchSize)
	assert.True(t, cfg.EnableLyrics)
	assert.True(t, cfg.EnableDescription)
	assert.True(t, cfg.EnableFlamingo)
}

func TestEmbedRequest(t *testing.T) {
	req := EmbedRequest{
		FilePath:  "/path/to/audio.mp3",
		TrackName: "Test Track",
		TrackID:   "track123",
		Artist:    "Test Artist",
		Title:     "Test Title",
		Album:     "Test Album",
	}

	assert.Equal(t, "/path/to/audio.mp3", req.FilePath)
	assert.Equal(t, "Test Track", req.TrackName)
	assert.Equal(t, "track123", req.TrackID)
	assert.Equal(t, "Test Artist", req.Artist)
	assert.Equal(t, "Test Title", req.Title)
	assert.Equal(t, "Test Album", req.Album)
}

func TestEmbedResult(t *testing.T) {
	result := EmbedResult{
		TrackName:            "Test Track",
		LyricsEmbedding:      []float64{0.1, 0.2, 0.3},
		DescriptionEmbedding: []float64{0.4, 0.5, 0.6},
		FlamingoEmbedding:    []float64{0.7, 0.8, 0.9},
		Description:          "A beautiful song",
	}

	assert.Equal(t, "Test Track", result.TrackName)
	assert.Equal(t, []float64{0.1, 0.2, 0.3}, result.LyricsEmbedding)
	assert.Equal(t, []float64{0.4, 0.5, 0.6}, result.DescriptionEmbedding)
	assert.Equal(t, []float64{0.7, 0.8, 0.9}, result.FlamingoEmbedding)
	assert.Equal(t, "A beautiful song", result.Description)
}

func TestStatusResult(t *testing.T) {
	result := StatusResult{
		Embedded:          true,
		HasDescription:    true,
		HasAudioEmbedding: true,
		CanonicalName:     "Artist - Title",
	}

	assert.True(t, result.Embedded)
	assert.True(t, result.HasDescription)
	assert.True(t, result.HasAudioEmbedding)
	assert.Equal(t, "Artist - Title", result.CanonicalName)
}

func TestEmbedderClosed(t *testing.T) {
	// Create embedder with nil clients (will fail on actual calls but can test closed state)
	e := &Embedder{
		closed: true,
	}

	ctx := context.Background()

	// Test that operations fail when embedder is closed
	_, err := e.EmbedAudio(ctx, EmbedRequest{FilePath: "/test.mp3"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "closed")

	_, err = e.EmbedText(ctx, "test text")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "closed")

	_, err = e.CheckStatus(ctx, StatusRequest{TrackID: "test"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "closed")

	err = e.FlushBatch(ctx)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "closed")
}

func TestEmbedAudioRequiresFilePath(t *testing.T) {
	e := &Embedder{
		closed: false,
	}

	ctx := context.Background()

	_, err := e.EmbedAudio(ctx, EmbedRequest{})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "file path is required")
}

func TestEmbedTextRequiresText(t *testing.T) {
	e := &Embedder{
		closed: false,
	}

	ctx := context.Background()

	_, err := e.EmbedText(ctx, "")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "text is required")
}
