package scanner

import (
	"bufio"
	"context"
	"encoding/json"
	"net"
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

// createMockEmbedSocketServer creates a Unix socket server that records embedding calls.
func createMockEmbedSocketServer(t *testing.T, recorder *embedRecorder) (string, func()) {
	socketPath := filepath.Join(t.TempDir(), "embed.sock")
	listener, err := net.Listen("unix", socketPath)
	if err != nil {
		t.Fatalf("failed to create socket: %v", err)
	}

	done := make(chan struct{})
	go func() {
		for {
			conn, err := listener.Accept()
			if err != nil {
				select {
				case <-done:
					return
				default:
					continue
				}
			}
			go handleMockConnection(conn, recorder)
		}
	}()

	cleanup := func() {
		close(done)
		listener.Close()
	}
	return socketPath, cleanup
}

func handleMockConnection(conn net.Conn, recorder *embedRecorder) {
	defer conn.Close()

	reader := bufio.NewReader(conn)
	line, err := reader.ReadBytes('\n')
	if err != nil {
		return
	}

	var payload map[string]any
	if err := json.Unmarshal(line, &payload); err != nil {
		return
	}

	trackID, _ := payload["track_id"].(string)
	action, _ := payload["action"].(string)

	var response []byte
	switch action {
	case "status":
		recorder.recordStatus(trackID)
		response = []byte(`{"embedded": false, "hasDescription": false, "name": "stub"}` + "\n")
	case "embed":
		recorder.recordEmbed(trackID)
		response = []byte(`{"status": "ok"}` + "\n")
	default:
		response = []byte(`{"status": "error", "message": "unknown action"}` + "\n")
	}

	if _, err := conn.Write(response); err != nil {
		return
	}
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
	socketPath, cleanup := createMockEmbedSocketServer(t, recorder)
	defer cleanup()

	// Override the default socket path for testing
	origSocketPath := defaultSocketPath
	// We need to patch the socket path - create a custom client for this test
	testClient := &socketEmbeddingClient{
		socketPath:    socketPath,
		statusTimeout: statusCheckTimeout,
		embedTimeout:  10 * time.Minute,
	}

	// Create scanner with custom embed worker using our test client
	s := &controller{
		rootCtx:     ctx,
		ds:          ds,
		cw:          artwork.NoopCacheWarmer(),
		broker:      events.NoopBroker(),
		pls:         core.NewPlaylists(ds),
		metrics:     metrics.NewNoopInstance(),
		embedWorker: newEmbeddingWorker(testClient),
	}
	_ = origSocketPath // suppress unused warning

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
