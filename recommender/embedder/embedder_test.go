package embedder

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/navidrome/navidrome/recommender/milvus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type fakeMusicClient struct {
	textEmbeddings  map[string][]float32
	audioEmbeddings map[string][]float32
	descriptions    map[string]string
	lyrics          map[string]string

	textErr   error
	audioErr  error
	descErr   error
	lyricsErr error

	textCalls   []string
	audioCalls  []string
	descCalls   []string
	lyricsCalls []string
	callOrder   []string
	closed      bool
}

func (f *fakeMusicClient) EmbedText(text string) ([]float32, error) {
	f.callOrder = append(f.callOrder, "text:"+text)
	f.textCalls = append(f.textCalls, text)
	if f.textErr != nil {
		return nil, f.textErr
	}
	if embedding, ok := f.textEmbeddings[text]; ok {
		out := make([]float32, len(embedding))
		copy(out, embedding)
		return out, nil
	}
	return nil, nil
}

func (f *fakeMusicClient) EmbedAudio(path string) ([]float32, error) {
	f.callOrder = append(f.callOrder, "audio:"+path)
	f.audioCalls = append(f.audioCalls, path)
	if f.audioErr != nil {
		return nil, f.audioErr
	}
	if embedding, ok := f.audioEmbeddings[path]; ok {
		out := make([]float32, len(embedding))
		copy(out, embedding)
		return out, nil
	}
	return nil, nil
}

func (f *fakeMusicClient) GenerateDescription(path string) (string, error) {
	f.callOrder = append(f.callOrder, "desc:"+path)
	f.descCalls = append(f.descCalls, path)
	if f.descErr != nil {
		return "", f.descErr
	}
	if desc, ok := f.descriptions[path]; ok {
		return desc, nil
	}
	return "", nil
}

func (f *fakeMusicClient) GenerateLyrics(path string) (string, error) {
	f.callOrder = append(f.callOrder, "lyrics:"+path)
	f.lyricsCalls = append(f.lyricsCalls, path)
	if f.lyricsErr != nil {
		return "", f.lyricsErr
	}
	if lyrics, ok := f.lyrics[path]; ok {
		return lyrics, nil
	}
	return "", nil
}

func (f *fakeMusicClient) Close() {
	f.closed = true
}

type upsertCall struct {
	collection string
	data       []milvus.EmbeddingData
}

type fakeVectorStore struct {
	upserts            []upsertCall
	existsByCollection map[string]map[string]bool
	upsertErr          error
	existsErr          error
}

func (f *fakeVectorStore) Upsert(ctx context.Context, collection string, data []milvus.EmbeddingData) error {
	f.upserts = append(f.upserts, upsertCall{collection: collection, data: data})
	return f.upsertErr
}

func (f *fakeVectorStore) Exists(ctx context.Context, collection string, names []string) (map[string]bool, error) {
	if f.existsErr != nil {
		return nil, f.existsErr
	}
	if f.existsByCollection == nil {
		return map[string]bool{}, nil
	}
	if entries, ok := f.existsByCollection[collection]; ok {
		return entries, nil
	}
	return map[string]bool{}, nil
}

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
		Lyrics:    "Some lyrics",
	}

	assert.Equal(t, "/path/to/audio.mp3", req.FilePath)
	assert.Equal(t, "Test Track", req.TrackName)
	assert.Equal(t, "track123", req.TrackID)
	assert.Equal(t, "Test Artist", req.Artist)
	assert.Equal(t, "Test Title", req.Title)
	assert.Equal(t, "Test Album", req.Album)
	assert.Equal(t, "Some lyrics", req.Lyrics)
}

func TestEmbedResult(t *testing.T) {
	result := EmbedResult{
		TrackName:            "Test Track",
		LyricsEmbedding:      []float64{0.1, 0.2, 0.3},
		DescriptionEmbedding: []float64{0.4, 0.5, 0.6},
		FlamingoEmbedding:    []float64{0.7, 0.8, 0.9},
		Description:          "A beautiful song",
		GeneratedLyrics:      "Some generated lyrics",
	}

	assert.Equal(t, "Test Track", result.TrackName)
	assert.Equal(t, []float64{0.1, 0.2, 0.3}, result.LyricsEmbedding)
	assert.Equal(t, []float64{0.4, 0.5, 0.6}, result.DescriptionEmbedding)
	assert.Equal(t, []float64{0.7, 0.8, 0.9}, result.FlamingoEmbedding)
	assert.Equal(t, "A beautiful song", result.Description)
	assert.Equal(t, "Some generated lyrics", result.GeneratedLyrics)
}

func TestStatusResult(t *testing.T) {
	result := StatusResult{
		Embedded:          true,
		HasDescription:    true,
		HasAudioEmbedding: true,
		HasLyrics:         true,
		CanonicalName:     "Artist - Title",
	}

	assert.True(t, result.Embedded)
	assert.True(t, result.HasDescription)
	assert.True(t, result.HasAudioEmbedding)
	assert.True(t, result.HasLyrics)
	assert.Equal(t, "Artist - Title", result.CanonicalName)
}

func TestEmbedderClosed(t *testing.T) {
	e := &Embedder{closed: true}

	ctx := context.Background()

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
	music := &fakeMusicClient{}
	store := &fakeVectorStore{}
	e := New(Config{}, music, store)

	ctx := context.Background()

	_, err := e.EmbedAudio(ctx, EmbedRequest{})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "file path is required")
}

func TestEmbedTextRequiresText(t *testing.T) {
	music := &fakeMusicClient{}
	e := New(Config{}, music, &fakeVectorStore{})

	ctx := context.Background()

	_, err := e.EmbedText(ctx, "")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "text is required")
}

func TestEmbedAudioStoresEmbeddings(t *testing.T) {
	music := &fakeMusicClient{
		textEmbeddings: map[string][]float32{
			"generated lyrics": {0.1, 0.2},
			"desc":             {0.3, 0.4},
		},
		audioEmbeddings: map[string][]float32{
			"/test.mp3": {0.5, 0.6},
		},
		descriptions: map[string]string{
			"/test.mp3": "desc",
		},
		lyrics: map[string]string{
			"/test.mp3": "generated lyrics",
		},
	}
	store := &fakeVectorStore{}
	e := New(Config{EnableLyrics: true, EnableDescription: true, EnableFlamingo: true}, music, store)

	ctx := context.Background()
	result, err := e.EmbedAudio(ctx, EmbedRequest{FilePath: "/test.mp3", TrackName: "Track"})
	require.NoError(t, err)

	assert.Equal(t, "Track", result.TrackName)
	assert.InDeltaSlice(t, []float64{0.1, 0.2}, result.LyricsEmbedding, 1e-6)
	assert.InDeltaSlice(t, []float64{0.3, 0.4}, result.DescriptionEmbedding, 1e-6)
	assert.InDeltaSlice(t, []float64{0.5, 0.6}, result.FlamingoEmbedding, 1e-6)
	assert.Equal(t, "desc", result.Description)
	assert.Equal(t, "generated lyrics", result.GeneratedLyrics)

	require.Len(t, store.upserts, 3)
	assert.Equal(t, milvus.CollectionLyrics, store.upserts[0].collection)
	assert.Equal(t, ModelLyrics, store.upserts[0].data[0].ModelID)
	assert.Equal(t, milvus.CollectionDescription, store.upserts[1].collection)
	assert.Equal(t, ModelDescription, store.upserts[1].data[0].ModelID)
	assert.Equal(t, milvus.CollectionFlamingo, store.upserts[2].collection)
	assert.Equal(t, ModelFlamingo, store.upserts[2].data[0].ModelID)
}

func TestEmbedAudioGroupsModelStages(t *testing.T) {
	ctx := context.Background()
	music := &fakeMusicClient{
		textEmbeddings: map[string][]float32{
			"generated lyrics": {0.1, 0.2},
			"description":      {0.3, 0.4},
		},
		audioEmbeddings: map[string][]float32{
			"/test.mp3": {0.5, 0.6},
		},
		descriptions: map[string]string{
			"/test.mp3": "description",
		},
		lyrics: map[string]string{
			"/test.mp3": "generated lyrics",
		},
	}
	store := &fakeVectorStore{}
	e := New(Config{EnableLyrics: true, EnableDescription: true, EnableFlamingo: true}, music, store)

	_, err := e.EmbedAudio(ctx, EmbedRequest{FilePath: "/test.mp3", TrackName: "Track"})
	require.NoError(t, err)

	// Order: audio embedding, description generation, lyrics generation, then text embeddings for lyrics and description
	assert.Equal(t, []string{
		"audio:/test.mp3",
		"desc:/test.mp3",
		"lyrics:/test.mp3",
		"text:generated lyrics",
		"text:description",
	}, music.callOrder)
}

func TestEmbedAudioSkipsFailedStages(t *testing.T) {
	music := &fakeMusicClient{
		descErr:  errors.New("desc boom"),
		audioErr: errors.New("audio boom"),
	}
	store := &fakeVectorStore{}
	e := New(Config{EnableDescription: true, EnableFlamingo: true}, music, store)

	ctx := context.Background()
	result, err := e.EmbedAudio(ctx, EmbedRequest{FilePath: "/test.mp3", TrackName: "Track"})
	require.NoError(t, err)

	assert.Empty(t, result.Description)
	assert.Nil(t, result.DescriptionEmbedding)
	assert.Nil(t, result.FlamingoEmbedding)
	assert.Empty(t, store.upserts)
}

func TestEmbedTextDelegates(t *testing.T) {
	music := &fakeMusicClient{
		textEmbeddings: map[string][]float32{
			"hello": {0.9, 0.8},
		},
	}
	e := New(Config{}, music, &fakeVectorStore{})

	ctx := context.Background()
	embedding, err := e.EmbedText(ctx, "hello")
	require.NoError(t, err)
	assert.InDeltaSlice(t, []float64{0.9, 0.8}, embedding, 1e-6)
	assert.Equal(t, []string{"hello"}, music.textCalls)
}

func TestCheckStatusUsesStore(t *testing.T) {
	store := &fakeVectorStore{
		existsByCollection: map[string]map[string]bool{
			milvus.CollectionLyrics:      {"Artist - Title": true},
			milvus.CollectionDescription: {"Artist - Title": false},
			milvus.CollectionFlamingo:    {"Artist - Title": true},
		},
	}
	e := New(Config{}, &fakeMusicClient{}, store)

	ctx := context.Background()
	result, err := e.CheckStatus(ctx, StatusRequest{Artist: "Artist", Title: "Title"})
	require.NoError(t, err)

	assert.True(t, result.Embedded)
	assert.True(t, result.HasAudioEmbedding)
	assert.True(t, result.HasLyrics)
	assert.False(t, result.HasDescription)
	assert.Equal(t, "Artist - Title", result.CanonicalName)
}

func TestEmbedAudioGeneratesLyrics(t *testing.T) {
	music := &fakeMusicClient{
		lyrics: map[string]string{
			"/test.mp3": "generated lyrics content",
		},
		textEmbeddings: map[string][]float32{
			"generated lyrics content": {0.1, 0.2},
		},
	}
	store := &fakeVectorStore{}
	e := New(Config{EnableLyrics: true}, music, store)

	ctx := context.Background()
	result, err := e.EmbedAudio(ctx, EmbedRequest{FilePath: "/test.mp3", TrackName: "Track"})
	require.NoError(t, err)

	assert.Equal(t, "generated lyrics content", result.GeneratedLyrics)
	assert.InDeltaSlice(t, []float64{0.1, 0.2}, result.LyricsEmbedding, 1e-6)
	assert.Contains(t, music.lyricsCalls, "/test.mp3")
}

func TestCloseClosesMusicClient(t *testing.T) {
	music := &fakeMusicClient{}
	e := New(Config{}, music, &fakeVectorStore{})

	require.NoError(t, e.Close())
	assert.True(t, music.closed)
}
