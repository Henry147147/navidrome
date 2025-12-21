package scanner

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/navidrome/navidrome/log"
)

type retryConfig struct {
	maxRetries     int
	initialBackoff time.Duration
	maxBackoff     time.Duration
}

var defaultRetryConfig = retryConfig{
	maxRetries:     3,
	initialBackoff: 2 * time.Second,
	maxBackoff:     30 * time.Second,
}

type embeddingError struct {
	Track string
	Error string
	Time  time.Time
}

type embeddingWorker struct {
	client         embeddingClient
	active         map[string]struct{}
	queue          []embeddingCandidate
	mu             sync.Mutex
	running        bool
	closed         bool
	wg             sync.WaitGroup
	totalQueued    atomic.Int64
	totalProcessed atomic.Int64
	totalSkipped   atomic.Int64
	totalFailed    atomic.Int64
	progressTicker *time.Ticker
	progressDone   chan struct{}
	recentErrors   []embeddingError
	errorsMu       sync.Mutex
	maxErrors      int
}

func newEmbeddingWorker(client embeddingClient) *embeddingWorker {
	return &embeddingWorker{
		client:       client,
		active:       make(map[string]struct{}),
		queue:        make([]embeddingCandidate, 0),
		maxErrors:    100,
		recentErrors: make([]embeddingError, 0, 100),
	}
}

// Wait blocks until the current queue finishes processing.
func (w *embeddingWorker) Wait() {
	w.wg.Wait()
}

// Close stops the worker and waits for any in-flight processing to finish.
func (w *embeddingWorker) Close() {
	w.mu.Lock()
	w.closed = true
	w.queue = nil
	w.active = make(map[string]struct{})
	w.mu.Unlock()
	w.wg.Wait()
}

func (w *embeddingWorker) Enqueue(ctx context.Context, candidates []embeddingCandidate) {
	w.mu.Lock()
	if w.closed {
		w.mu.Unlock()
		return
	}
	newCount := 0
	for _, c := range candidates {
		key := c.key()
		if _, exists := w.active[key]; exists {
			continue
		}
		w.active[key] = struct{}{}
		w.queue = append(w.queue, c)
		newCount++
	}
	w.totalQueued.Add(int64(newCount))

	if w.running || len(w.queue) == 0 {
		w.mu.Unlock()
		return
	}
	w.running = true
	w.mu.Unlock()

	// Start progress reporter
	w.startProgressReporter(ctx)

	w.wg.Add(1)
	go func() {
		defer w.wg.Done()
		w.loop(ctx)
	}()
}

func (w *embeddingWorker) loop(ctx context.Context) {
	iteration := 0
	for {
		// Check for context cancellation
		select {
		case <-ctx.Done():
			w.mu.Lock()
			w.running = false
			w.mu.Unlock()
			log.Info(ctx, "Embedding worker context cancelled, exiting gracefully")
			return
		default:
		}

		w.mu.Lock()
		if w.closed || len(w.queue) == 0 {
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
			log.Info(ctx, "Embedding worker loop started", "queueSize", remaining+1)
		}
		iteration++

		log.Debug(ctx, "Processing embedding candidate", "track", candidate.TrackPath, "artist", candidate.Artist, "title", candidate.Title, "remaining", remaining)
		w.processWithRecovery(ctx, candidate)

		w.mu.Lock()
		delete(w.active, candidate.key())
		w.mu.Unlock()
	}
}

func (w *embeddingWorker) startProgressReporter(ctx context.Context) {
	w.mu.Lock()
	if w.progressTicker != nil {
		w.mu.Unlock()
		return
	}
	w.progressTicker = time.NewTicker(30 * time.Second)
	w.progressDone = make(chan struct{})
	w.mu.Unlock()

	go func() {
		defer func() {
			w.mu.Lock()
			if w.progressTicker != nil {
				w.progressTicker.Stop()
				w.progressTicker = nil
			}
			if w.progressDone != nil {
				close(w.progressDone)
				w.progressDone = nil
			}
			w.mu.Unlock()
		}()

		for {
			select {
			case <-w.progressTicker.C:
				w.logProgress(ctx)
			case <-ctx.Done():
				return
			case <-w.progressDone:
				return
			}
		}
	}()
}

func (w *embeddingWorker) logProgress(ctx context.Context) {
	queued := w.totalQueued.Load()
	processed := w.totalProcessed.Load()
	skipped := w.totalSkipped.Load()
	failed := w.totalFailed.Load()

	w.mu.Lock()
	remaining := len(w.queue)
	w.mu.Unlock()

	if queued > 0 {
		pct := float64(processed) / float64(queued) * 100
		log.Info(ctx, "Embedding progress",
			"queued", queued,
			"processed", processed,
			"skipped", skipped,
			"failed", failed,
			"remaining", remaining,
			"completion", fmt.Sprintf("%.1f%%", pct))
	}
}

func (w *embeddingWorker) recordError(candidate embeddingCandidate, err error) {
	w.errorsMu.Lock()
	defer w.errorsMu.Unlock()

	e := embeddingError{
		Track: candidate.TrackPath,
		Error: err.Error(),
		Time:  time.Now(),
	}

	w.recentErrors = append(w.recentErrors, e)
	if len(w.recentErrors) > w.maxErrors {
		w.recentErrors = w.recentErrors[len(w.recentErrors)-w.maxErrors:]
	}
}

func (w *embeddingWorker) GetRecentErrors() []embeddingError {
	w.errorsMu.Lock()
	defer w.errorsMu.Unlock()
	result := make([]embeddingError, len(w.recentErrors))
	copy(result, w.recentErrors)
	return result
}

func (w *embeddingWorker) processWithRecovery(ctx context.Context, candidate embeddingCandidate) {
	defer func() {
		if r := recover(); r != nil {
			log.Error(ctx, "Embedding worker panicked while processing candidate", "track", candidate.TrackPath, "error", r)
		}
	}()
	w.processWithRetry(ctx, candidate)
}

func (w *embeddingWorker) processWithRetry(ctx context.Context, candidate embeddingCandidate) {
	cfg := defaultRetryConfig
	backoff := cfg.initialBackoff

	for attempt := 0; attempt <= cfg.maxRetries; attempt++ {
		if attempt > 0 {
			log.Debug(ctx, "Retrying embedding", "track", candidate.TrackPath, "attempt", attempt, "backoff", backoff)
			select {
			case <-time.After(backoff):
				// Continue with retry
			case <-ctx.Done():
				log.Debug(ctx, "Context cancelled during retry backoff", "track", candidate.TrackPath)
				w.totalProcessed.Add(1)
				w.totalFailed.Add(1)
				return
			}
			backoff *= 2
			if backoff > cfg.maxBackoff {
				backoff = cfg.maxBackoff
			}
		}

		skipped, err := w.processSingle(ctx, candidate)
		if err == nil {
			w.totalProcessed.Add(1)
			if skipped {
				w.totalSkipped.Add(1)
			}
			return // Success
		}

		if attempt == cfg.maxRetries {
			log.Error(ctx, "Embedding failed after retries", err, "track", candidate.TrackPath, "attempts", attempt+1)
			w.recordError(candidate, err)
			w.totalProcessed.Add(1)
			w.totalFailed.Add(1)
			return
		}
	}
}

func (w *embeddingWorker) processSingle(ctx context.Context, candidate embeddingCandidate) (skipped bool, err error) {
	status, err := w.client.CheckEmbedding(ctx, candidate)
	if err == nil {
		if status.Embedded && status.HasDescription {
			log.Debug(ctx, "Embedding already present, skipping", "track", candidate.TrackPath, "name", status.Name)
			return true, nil
		}
		log.Debug(ctx, "Embedding status check returned", "embedded", status.Embedded, "hasDescription", status.HasDescription, "name", status.Name)
	} else {
		log.Warn(ctx, "Embedding status check failed; proceeding to embed", "track", candidate.TrackPath, "error", err)
	}

	log.Debug(ctx, "Sending track to embedding service", "track", candidate.TrackPath)
	if err := w.client.EmbedSong(ctx, candidate); err != nil {
		return false, err
	}
	log.Info(ctx, "Embedded track in background", "track", candidate.TrackPath, "name", status.Name)
	return false, nil
}
