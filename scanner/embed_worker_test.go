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
