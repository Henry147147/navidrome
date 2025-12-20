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

	iteration := 0
	for {
		w.mu.Lock()
		if len(w.queue) == 0 {
			w.running = false
			w.mu.Unlock()
			log.Info(ctx, "Embedding worker queue empty, exiting")
			return
		}
		candidate := w.queue[0]
		w.queue = w.queue[1:]
		remaining := len(w.queue)
		w.mu.Unlock()

		if iteration == 0 {
			fmt.Fprintf(os.Stderr, "[EMBED-DEBUG] Worker loop started, queue size=%d\n", remaining+1)
			log.Info(ctx, "Embedding worker loop started", "queueSize", remaining+1)
		}
		iteration++
		// Write directly to stderr in case logging is misconfigured
		fmt.Fprintf(os.Stderr, "[EMBED-DEBUG] Loop iteration %d starting\n", iteration)
		log.Info(ctx, "Embedding worker loop iteration", "iteration", iteration)
		fmt.Fprintf(os.Stderr, "[EMBED-DEBUG] Dequeued candidate, remaining=%d\n", remaining)
		log.Info(ctx, "Embedding worker dequeued candidate", "remaining", remaining)

		log.Info(ctx, "Processing embedding candidate", "track", candidate.TrackPath, "artist", candidate.Artist, "title", candidate.Title, "remaining", remaining)
		w.processWithRecovery(ctx, candidate)

		w.mu.Lock()
		delete(w.active, candidate.key())
		w.mu.Unlock()
	}
}

func (w *embeddingWorker) processWithRecovery(ctx context.Context, candidate embeddingCandidate) {
	defer func() {
		if r := recover(); r != nil {
			log.Error(ctx, "Embedding worker panicked while processing candidate", "track", candidate.TrackPath, "error", r)
		}
	}()
	w.process(ctx, candidate)
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
