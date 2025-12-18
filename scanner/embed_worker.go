package scanner

import (
	"context"
	"fmt"
	"os"
	"sync"

	"github.com/navidrome/navidrome/log"
)

type embeddingWorker struct {
	client  embeddingClient
	active  map[string]struct{}
	queue   []embeddingCandidate
	mu      sync.Mutex
	running bool
}

func newEmbeddingWorker(client embeddingClient) *embeddingWorker {
	return &embeddingWorker{
		client: client,
		active: make(map[string]struct{}),
		queue:  make([]embeddingCandidate, 0),
	}
}

func (w *embeddingWorker) Enqueue(candidates []embeddingCandidate) {
	w.mu.Lock()
	for _, c := range candidates {
		key := c.key()
		if _, exists := w.active[key]; exists {
			continue
		}
		w.active[key] = struct{}{}
		w.queue = append(w.queue, c)
	}
	if w.running || len(w.queue) == 0 {
		w.mu.Unlock()
		return
	}
	w.running = true
	w.mu.Unlock()

	go w.loop()
}

func (w *embeddingWorker) loop() {
	ctx := context.Background()

	// Ensure we always reset running state when the loop exits (including panics)
	defer func() {
		if r := recover(); r != nil {
			log.Error(ctx, "Embedding worker panicked", "error", r)
		}
		w.mu.Lock()
		w.running = false
		w.mu.Unlock()
		log.Info(ctx, "Embedding worker loop finished")
	}()

	w.mu.Lock()
	queueSize := len(w.queue)
	w.mu.Unlock()
	fmt.Fprintf(os.Stderr, "[EMBED-DEBUG] Worker loop started, queue size=%d\n", queueSize)
	log.Info(ctx, "Embedding worker loop started", "queueSize", queueSize)

	iteration := 0
	for {
		iteration++
		// Write directly to stderr in case logging is misconfigured
		fmt.Fprintf(os.Stderr, "[EMBED-DEBUG] Loop iteration %d starting\n", iteration)
		log.Info(ctx, "Embedding worker loop iteration", "iteration", iteration)

		w.mu.Lock()
		currentLen := len(w.queue)
		fmt.Fprintf(os.Stderr, "[EMBED-DEBUG] Acquired lock, queue len=%d\n", currentLen)
		log.Info(ctx, "Embedding worker acquired lock", "queueLen", currentLen)
		if currentLen == 0 {
			w.mu.Unlock()
			log.Info(ctx, "Embedding worker queue empty, exiting")
			return
		}
		candidate := w.queue[0]
		w.queue = w.queue[1:]
		remaining := len(w.queue)
		w.mu.Unlock()

		log.Info(ctx, "Processing embedding candidate", "track", candidate.TrackPath, "artist", candidate.Artist, "title", candidate.Title, "remaining", remaining)
		w.process(ctx, candidate)

		w.mu.Lock()
		delete(w.active, candidate.key())
		w.mu.Unlock()
	}
}

func (w *embeddingWorker) process(ctx context.Context, candidate embeddingCandidate) {
	fmt.Fprintf(os.Stderr, "[EMBED-DEBUG] Calling CheckEmbedding for %s\n", candidate.TrackPath)
	status, err := w.client.CheckEmbedding(ctx, candidate)
	fmt.Fprintf(os.Stderr, "[EMBED-DEBUG] CheckEmbedding returned, err=%v\n", err)
	if err == nil {
		if status.Embedded && status.HasDescription {
			log.Info(ctx, "Embedding already present, skipping", "track", candidate.TrackPath, "name", status.Name)
			return
		}
		log.Debug(ctx, "Embedding status check returned", "embedded", status.Embedded, "hasDescription", status.HasDescription, "name", status.Name)
	} else {
		log.Warn(ctx, "Embedding status check failed; proceeding to embed", "track", candidate.TrackPath, "error", err)
	}

	fmt.Fprintf(os.Stderr, "[EMBED-DEBUG] Calling EmbedSong for %s\n", candidate.TrackPath)
	log.Info(ctx, "Sending track to embedding service", "track", candidate.TrackPath)
	if err := w.client.EmbedSong(ctx, candidate); err != nil {
		fmt.Fprintf(os.Stderr, "[EMBED-DEBUG] EmbedSong failed: %v\n", err)
		log.Error(ctx, "Embedding failed", err, "track", candidate.TrackPath)
		return
	}
	fmt.Fprintf(os.Stderr, "[EMBED-DEBUG] EmbedSong succeeded for %s\n", candidate.TrackPath)
	log.Info(ctx, "Embedded track in background", "track", candidate.TrackPath, "name", status.Name)
}
