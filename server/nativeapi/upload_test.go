package nativeapi

import (
	"context"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/navidrome/navidrome/conf"
)

type stubEmbedClient struct {
	resp map[string]any
	err  error
}

func (s *stubEmbedClient) Embed(musicPath, musicName, cuePath string, settings map[string]any) (map[string]any, error) {
	return s.resp, s.err
}

func TestEmbedQueueProcessesJobAndCopiesFile(t *testing.T) {
	// Inject stub embed client
	embedClient = &stubEmbedClient{
		resp: map[string]any{
			"status":        "ok",
			"duplicates":    []string{},
			"allDuplicates": false,
		},
	}
	embedClientOnce = sync.Once{}

	musicFolder := t.TempDir()
	conf.Server.MusicFolder = musicFolder

	tempDir := t.TempDir()
	musicFile := filepath.Join(tempDir, "song.flac")
	if err := os.WriteFile(musicFile, []byte("audio-bytes"), 0o644); err != nil {
		t.Fatalf("write temp music: %v", err)
	}

	router := &Router{embedQueue: newEmbedQueue()}
	defer router.embedQueue.Stop()

	input := embedJobInput{
		MusicPath: musicFile,
		MusicName: "song.flac",
		TempDir:   tempDir,
	}

	work := &embedWork{}
	work.run = func() error {
		resp, err := router.processEmbedJob(context.Background(), input)
		if err == nil {
			work.result = resp
		}
		return err
	}

	jobID, err := router.embedQueue.Enqueue(work)
	if err != nil {
		t.Fatalf("enqueue: %v", err)
	}

	var job *embedJob
	for i := 0; i < 100; i++ {
		var ok bool
		job, ok = router.embedQueue.Get(jobID)
		if ok && job.Status == StatusSucceeded {
			break
		}
		if ok && job.Status == StatusFailed {
			t.Fatalf("job failed: %s", job.Error)
		}
		time.Sleep(10 * time.Millisecond)
	}
	if job == nil || job.Status != StatusSucceeded {
		t.Fatalf("job did not succeed, status=%v", job)
	}

	destPath := filepath.Join(musicFolder, "song.flac")
	if _, err := os.Stat(destPath); err != nil {
		t.Fatalf("expected copied file at %s: %v", destPath, err)
	}
}
