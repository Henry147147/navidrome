package scanner

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"sync"
	"testing"
	"testing/fstest"
	"time"

	"github.com/navidrome/navidrome/conf"
	"github.com/navidrome/navidrome/conf/configtest"
	"github.com/navidrome/navidrome/core"
	"github.com/navidrome/navidrome/core/artwork"
	"github.com/navidrome/navidrome/core/metrics"
	"github.com/navidrome/navidrome/core/storage/storagetest"
	"github.com/navidrome/navidrome/db"
	"github.com/navidrome/navidrome/model"
	"github.com/navidrome/navidrome/persistence"
	"github.com/navidrome/navidrome/server/events"
	"github.com/navidrome/navidrome/tests"
)

type embedRecorder struct {
	mu          sync.Mutex
	statusCalls []string
	embedCalls  []string
}

func (r *embedRecorder) recordStatus(id string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.statusCalls = append(r.statusCalls, id)
}

func (r *embedRecorder) recordEmbed(id string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.embedCalls = append(r.embedCalls, id)
}

func (r *embedRecorder) counts() (int, int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.statusCalls), len(r.embedCalls)
}

// This test exercises the full scan path and asserts that a newly discovered
// song triggers calls into the Python embedding service (status + embed).
func TestScanTriggersEmbeddingServiceForMissingTrack(t *testing.T) {
	tests.Init(t, true)
	restore := configtest.SetupConfig()
	defer restore()

	ctx := context.Background()
	conf.Server.DbPath = filepath.Join(t.TempDir(), "scan-embed.db?_journal_mode=WAL")
	conf.Server.MusicFolder = "fake:///music"
	conf.Server.DevExternalScanner = false

	db.Init(ctx)
	t.Cleanup(func() { _ = tests.ClearDB() })

	// Minimal library with a single track using the fake storage backend
	fs := storagetest.FakeFS{}
	fs.SetFiles(fstest.MapFS{
		"Artist/Album/01 - Song.mp3": storagetest.Template()(storagetest.Track(1, "Song")),
	})
	storagetest.Register("fake", &fs)

	ds := persistence.New(db.Db())
	if err := ds.User(ctx).Put(&model.User{ID: "admin", UserName: "admin", IsAdmin: true, NewPassword: "pw"}); err != nil {
		t.Fatalf("create user: %v", err)
	}
	if err := ds.Library(ctx).Put(&model.Library{ID: 1, Name: "Fake Library", Path: "fake:///music"}); err != nil {
		t.Fatalf("create library: %v", err)
	}

	recorder := &embedRecorder{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload map[string]any
		_ = json.NewDecoder(r.Body).Decode(&payload)
		trackID, _ := payload["track_id"].(string)

		switch r.URL.Path {
		case "/embed/status":
			recorder.recordStatus(trackID)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"embedded": false, "hasDescription": false, "name": "stub"}`))
		case "/embed/audio":
			recorder.recordEmbed(trackID)
			w.WriteHeader(http.StatusOK)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	conf.Server.Recommendations.BaseURL = srv.URL
	conf.Server.Recommendations.TextBaseURL = srv.URL
	conf.Server.Recommendations.BatchBaseURL = srv.URL

	s := New(ctx, ds, artwork.NoopCacheWarmer(), events.NoopBroker(),
		core.NewPlaylists(ds), metrics.NewNoopInstance())

	if _, err := s.ScanAll(ctx, true); err != nil {
		t.Fatalf("scan failed: %v", err)
	}

	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		status, embed := recorder.counts()
		if status > 0 && embed > 0 {
			return
		}
		time.Sleep(25 * time.Millisecond)
	}
	status, embed := recorder.counts()
	t.Fatalf("expected embedding service to be called, got status=%d embed=%d", status, embed)
}
