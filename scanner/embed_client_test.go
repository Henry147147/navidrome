package scanner

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"
	"time"

	"github.com/navidrome/navidrome/conf"
)

func withRecommendConfig(base string, timeout time.Duration, fn func()) {
	prevBase := conf.Server.Recommendations.BaseURL
	prevTimeout := conf.Server.Recommendations.Timeout
	prevEmbedTimeout := conf.Server.Recommendations.EmbedTimeout
	conf.Server.Recommendations.BaseURL = base
	conf.Server.Recommendations.Timeout = timeout
	conf.Server.Recommendations.EmbedTimeout = 0
	fn()
	conf.Server.Recommendations.BaseURL = prevBase
	conf.Server.Recommendations.Timeout = prevTimeout
	conf.Server.Recommendations.EmbedTimeout = prevEmbedTimeout
}

func TestPythonEmbeddingClientCheckEmbeddingPayload(t *testing.T) {
	var received map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embed/status" {
			t.Fatalf("unexpected path %s", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&received); err != nil {
			t.Fatalf("decode: %v", err)
		}
		_, _ = w.Write([]byte(`{"embedded":true,"hasDescription":false,"name":"Artist - Title"}`))
	}))
	defer server.Close()

	withRecommendConfig(server.URL, time.Second, func() {
		client := newPythonEmbeddingClient()
		if client == nil {
			t.Fatalf("expected client when base URL set")
		}
		status, err := client.CheckEmbedding(context.Background(), embeddingCandidate{
			LibraryID:   1,
			LibraryPath: "/music",
			TrackPath:   "folder/song.flac",
			Artist:      "Artist",
			Title:       "Title",
			Album:       "Album",
		})
		if err != nil {
			t.Fatalf("check embedding: %v", err)
		}
		if !status.Embedded || status.HasDescription {
			t.Fatalf("unexpected status: %+v", status)
		}
	})

	if received["artist"] != "Artist" || received["title"] != "Title" {
		t.Fatalf("payload missing fields: %v", received)
	}
	alt := received["alternate_names"].([]any)
	if len(alt) != 1 || alt[0] != "song.flac" {
		t.Fatalf("unexpected alternate names: %v", alt)
	}
	if received["track_id"] == "" {
		t.Fatalf("track_id should be populated")
	}
}

func TestPythonEmbeddingClientEmbedSongPayload(t *testing.T) {
	var received map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embed/audio" {
			t.Fatalf("unexpected path %s", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&received); err != nil {
			t.Fatalf("decode: %v", err)
		}
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	}))
	defer server.Close()

	withRecommendConfig(server.URL, time.Second, func() {
		client := newPythonEmbeddingClient()
		if client == nil {
			t.Fatalf("expected client when base URL set")
		}
		err := client.EmbedSong(context.Background(), embeddingCandidate{
			LibraryID:   2,
			LibraryPath: "/music",
			TrackPath:   "artist/song.flac",
			Artist:      "Artist",
			Title:       "Song",
		})
		if err != nil {
			t.Fatalf("embed song: %v", err)
		}
	})

	if received["artist"] != "Artist" || received["title"] != "Song" {
		t.Fatalf("payload missing metadata: %v", received)
	}
	if received["music_file"] != filepath.Join("/music", "artist", "song.flac") {
		t.Fatalf("unexpected music_file: %v", received["music_file"])
	}
	if received["track_id"] == "" {
		t.Fatalf("track_id should be set")
	}
}
