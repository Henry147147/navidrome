package scanner

import (
	"context"
	"testing"
	"time"
)

func TestScheduleEmbeddingsInvokesPythonClient(t *testing.T) {
	client := newStubEmbeddingClient()
	worker := newTestEmbeddingWorker(t, client)
	s := &scannerImpl{embedWorker: worker}

	state := &scanState{}
	state.addEmbedCandidates(
		embeddingCandidate{LibraryID: 1, LibraryPath: "/music", TrackPath: "a.flac", Artist: "A", Title: "T1"},
		embeddingCandidate{LibraryID: 1, LibraryPath: "/music", TrackPath: "b.flac", Artist: "B", Title: "T2"},
	)

	s.scheduleEmbeddings(context.Background(), state)

	waitForCondition(t, time.Second, func() bool {
		client.mu.Lock()
		defer client.mu.Unlock()
		return len(client.embedCalls) == 2
	})

	client.mu.Lock()
	defer client.mu.Unlock()
	if len(client.checkCalls) != 2 {
		t.Fatalf("expected 2 status checks, got %d", len(client.checkCalls))
	}
}
