package scanner

import (
	"context"
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
	for {
		w.mu.Lock()
		if len(w.queue) == 0 {
			w.running = false
			w.mu.Unlock()
			return
		}
		candidate := w.queue[0]
		w.queue = w.queue[1:]
		w.mu.Unlock()

		w.process(ctx, candidate)

		w.mu.Lock()
		delete(w.active, candidate.key())
		w.mu.Unlock()
	}
}

func (w *embeddingWorker) process(ctx context.Context, candidate embeddingCandidate) {
	status, err := w.client.CheckEmbedding(ctx, candidate)
	if err != nil {
		log.Error(ctx, "Embedding status check failed", err, "track", candidate.TrackPath)
		return
	}
	if status.Embedded && status.HasDescription {
		log.Debug(ctx, "Embedding already present, skipping", "track", candidate.TrackPath, "name", status.Name)
		return
	}

	if err := w.client.EmbedSong(ctx, candidate); err != nil {
		log.Error(ctx, "Embedding failed", err, "track", candidate.TrackPath)
		return
	}
	log.Info(ctx, "Embedded track in background", "track", candidate.TrackPath, "name", status.Name)
}
