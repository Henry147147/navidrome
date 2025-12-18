package scanner

import (
	"context"
	"errors"
	"reflect"
	"sync"
	"testing"
	"time"
)

type stubEmbeddingClient struct {
	mu         sync.Mutex
	statuses   map[string]embeddingStatus
	checkErr   map[string]error
	embedErr   map[string]error
	checkCalls []string
	embedCalls []string
	embedBlock chan struct{}
}

func newStubEmbeddingClient() *stubEmbeddingClient {
	return &stubEmbeddingClient{
		statuses: make(map[string]embeddingStatus),
		checkErr: make(map[string]error),
		embedErr: make(map[string]error),
	}
}

func (c *stubEmbeddingClient) CheckEmbedding(_ context.Context, candidate embeddingCandidate) (embeddingStatus, error) {
	key := candidate.key()
	c.mu.Lock()
	c.checkCalls = append(c.checkCalls, key)
	err := c.checkErr[key]
	status, ok := c.statuses[key]
	c.mu.Unlock()
	if err != nil {
		return embeddingStatus{}, err
	}
	if !ok {
		return embeddingStatus{}, nil
	}
	return status, nil
}

func (c *stubEmbeddingClient) EmbedSong(_ context.Context, candidate embeddingCandidate) error {
	if c.embedBlock != nil {
		<-c.embedBlock
	}
	key := candidate.key()
	c.mu.Lock()
	c.embedCalls = append(c.embedCalls, key)
	err := c.embedErr[key]
	c.mu.Unlock()
	return err
}

func waitForCondition(t *testing.T, timeout time.Duration, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("condition not met within %s", timeout)
}

func TestEmbeddingWorkerSkipsAlreadyEmbedded(t *testing.T) {
	client := newStubEmbeddingClient()
	candidate := embeddingCandidate{LibraryID: 1, LibraryPath: "/music", TrackPath: "song.flac"}
	client.statuses[candidate.key()] = embeddingStatus{Embedded: true, HasDescription: true}

	worker := newEmbeddingWorker(client)
	worker.Enqueue([]embeddingCandidate{candidate})

	waitForCondition(t, time.Second, func() bool {
		client.mu.Lock()
		defer client.mu.Unlock()
		return len(client.checkCalls) == 1
	})

	client.mu.Lock()
	defer client.mu.Unlock()
	if len(client.embedCalls) != 0 {
		t.Fatalf("expected no embed calls, got %v", client.embedCalls)
	}
}

func TestEmbeddingWorkerDedupeAndSequentialOrder(t *testing.T) {
	client := newStubEmbeddingClient()
	first := embeddingCandidate{LibraryID: 1, LibraryPath: "/lib", TrackPath: "a.flac"}
	second := embeddingCandidate{LibraryID: 1, LibraryPath: "/lib", TrackPath: "b.flac"}

	worker := newEmbeddingWorker(client)
	worker.Enqueue([]embeddingCandidate{first, second, first})

	waitForCondition(t, time.Second, func() bool {
		client.mu.Lock()
		defer client.mu.Unlock()
		return len(client.embedCalls) == 2
	})

	client.mu.Lock()
	defer client.mu.Unlock()
	if !reflect.DeepEqual(client.embedCalls, []string{first.key(), second.key()}) {
		t.Fatalf("unexpected embed order: %v", client.embedCalls)
	}
}

func TestEmbeddingWorkerContinuesAfterErrors(t *testing.T) {
	client := newStubEmbeddingClient()
	first := embeddingCandidate{LibraryID: 1, LibraryPath: "/lib", TrackPath: "err.flac"}
	second := embeddingCandidate{LibraryID: 1, LibraryPath: "/lib", TrackPath: "ok.flac"}
	client.checkErr[first.key()] = errors.New("check failed")

	worker := newEmbeddingWorker(client)
	worker.Enqueue([]embeddingCandidate{first, second})

	waitForCondition(t, time.Second, func() bool {
		client.mu.Lock()
		defer client.mu.Unlock()
		return len(client.embedCalls) == 2
	})

	client.mu.Lock()
	defer client.mu.Unlock()
	if !reflect.DeepEqual(client.embedCalls, []string{first.key(), second.key()}) {
		t.Fatalf("expected both candidates embedded despite check error, got %v", client.embedCalls)
	}
}

func TestEmbeddingWorkerEnqueueNonBlocking(t *testing.T) {
	client := newStubEmbeddingClient()
	client.embedBlock = make(chan struct{})
	candidate := embeddingCandidate{LibraryID: 2, LibraryPath: "/music", TrackPath: "slow.flac"}
	worker := newEmbeddingWorker(client)

	start := time.Now()
	worker.Enqueue([]embeddingCandidate{candidate})
	if time.Since(start) > 50*time.Millisecond {
		t.Fatalf("enqueue should not block")
	}

	close(client.embedBlock)
	waitForCondition(t, time.Second, func() bool {
		client.mu.Lock()
		defer client.mu.Unlock()
		return len(client.embedCalls) == 1
	})
}

// panicEmbeddingClient panics on the first embed call, then works normally
type panicEmbeddingClient struct {
	mu           sync.Mutex
	panicOnFirst bool
	checkCalls   []string
	embedCalls   []string
}

func (c *panicEmbeddingClient) CheckEmbedding(_ context.Context, candidate embeddingCandidate) (embeddingStatus, error) {
	c.mu.Lock()
	c.checkCalls = append(c.checkCalls, candidate.key())
	c.mu.Unlock()
	return embeddingStatus{}, nil
}

func (c *panicEmbeddingClient) EmbedSong(_ context.Context, candidate embeddingCandidate) error {
	c.mu.Lock()
	shouldPanic := c.panicOnFirst
	c.panicOnFirst = false
	c.embedCalls = append(c.embedCalls, candidate.key())
	c.mu.Unlock()
	if shouldPanic {
		panic("simulated panic in embedding")
	}
	return nil
}

func TestEmbeddingWorkerRecoverFromPanic(t *testing.T) {
	client := &panicEmbeddingClient{panicOnFirst: true}
	first := embeddingCandidate{LibraryID: 1, LibraryPath: "/lib", TrackPath: "panic.flac"}
	second := embeddingCandidate{LibraryID: 1, LibraryPath: "/lib", TrackPath: "ok.flac"}

	worker := newEmbeddingWorker(client)

	// First enqueue will trigger a panic on the first candidate
	worker.Enqueue([]embeddingCandidate{first})

	// Wait for the panic to be caught and worker to reset
	waitForCondition(t, time.Second, func() bool {
		worker.mu.Lock()
		defer worker.mu.Unlock()
		return !worker.running
	})

	// Verify the worker can process new items after the panic
	worker.Enqueue([]embeddingCandidate{second})

	waitForCondition(t, time.Second, func() bool {
		client.mu.Lock()
		defer client.mu.Unlock()
		return len(client.embedCalls) == 2
	})

	client.mu.Lock()
	defer client.mu.Unlock()
	// Both should have been attempted (first one panicked but was still called)
	if len(client.embedCalls) != 2 {
		t.Fatalf("expected 2 embed calls after panic recovery, got %d: %v", len(client.embedCalls), client.embedCalls)
	}
}

func TestEmbeddingWorkerProcessesItemsEnqueuedWhileRunning(t *testing.T) {
	client := newStubEmbeddingClient()
	client.embedBlock = make(chan struct{})

	first := embeddingCandidate{LibraryID: 1, LibraryPath: "/lib", TrackPath: "first.flac"}
	second := embeddingCandidate{LibraryID: 1, LibraryPath: "/lib", TrackPath: "second.flac"}
	third := embeddingCandidate{LibraryID: 1, LibraryPath: "/lib", TrackPath: "third.flac"}

	worker := newEmbeddingWorker(client)

	// Enqueue first item - this will start the loop but block on embedBlock
	worker.Enqueue([]embeddingCandidate{first})

	// Wait for the first item to be picked up (loop is running)
	waitForCondition(t, time.Second, func() bool {
		client.mu.Lock()
		defer client.mu.Unlock()
		return len(client.checkCalls) == 1
	})

	// Enqueue more items while the loop is running
	worker.Enqueue([]embeddingCandidate{second, third})

	// Unblock the embed call
	close(client.embedBlock)

	// Wait for all items to be processed
	waitForCondition(t, time.Second, func() bool {
		client.mu.Lock()
		defer client.mu.Unlock()
		return len(client.embedCalls) == 3
	})

	client.mu.Lock()
	defer client.mu.Unlock()
	if !reflect.DeepEqual(client.embedCalls, []string{first.key(), second.key(), third.key()}) {
		t.Fatalf("expected all 3 candidates to be embedded in order, got %v", client.embedCalls)
	}
}
