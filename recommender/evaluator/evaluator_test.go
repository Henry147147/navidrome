package evaluator

import (
	"testing"
	"time"

	"github.com/navidrome/navidrome/model"
)

func TestSplitSessionsRespectsGapAndMissingTracks(t *testing.T) {
	start := time.Now().UTC().Truncate(time.Second)
	scrobbles := model.Scrobbles{
		{MediaFileID: "a", UserID: "u1", SubmissionTime: start},
		{MediaFileID: "a", UserID: "u1", SubmissionTime: start.Add(1 * time.Minute)},
		{MediaFileID: "b", UserID: "u1", SubmissionTime: start.Add(10 * time.Minute)},
		{MediaFileID: "missing", UserID: "u1", SubmissionTime: start.Add(20 * time.Minute)},
		{MediaFileID: "c", UserID: "u1", SubmissionTime: start.Add(2 * time.Hour)},
	}
	trackMap := map[string]model.MediaFile{
		"a": {ID: "a"},
		"b": {ID: "b"},
		"c": {ID: "c"},
	}

	sessions := splitSessions(scrobbles, trackMap, 30*time.Minute)

	if len(sessions) != 2 {
		t.Fatalf("expected 2 sessions, got %d", len(sessions))
	}
	if len(sessions[0]) != 2 || sessions[0][0].ID != "a" || sessions[0][1].ID != "b" {
		t.Fatalf("unexpected first session: %#v", sessions[0])
	}
	if len(sessions[1]) != 1 || sessions[1][0].ID != "c" {
		t.Fatalf("unexpected second session: %#v", sessions[1])
	}
}

func TestRecallAndNDCGAtK(t *testing.T) {
	recommended := []string{"x", "b", "a", "z"}
	relevant := map[string]struct{}{
		"a": {},
		"b": {},
	}

	recall := recallAtK(recommended, relevant)
	ndcg := ndcgAtK(recommended, relevant)

	if recall != 1 {
		t.Fatalf("expected full recall, got %f", recall)
	}
	if ndcg <= 0 || ndcg >= 1 {
		t.Fatalf("expected ndcg between 0 and 1, got %f", ndcg)
	}
}

func TestNoveltyRateUsesSeedArtists(t *testing.T) {
	seeds := model.MediaFiles{
		{ID: "seed-1", ArtistID: "artist-a"},
	}
	recommended := model.MediaFiles{
		{ID: "rec-1", ArtistID: "artist-a"},
		{ID: "rec-2", ArtistID: "artist-b"},
	}

	got := noveltyRate(seeds, recommended)
	if got != 0.5 {
		t.Fatalf("expected novelty 0.5, got %f", got)
	}
}

func TestCombineTaskReportsWeightsScenarioCounts(t *testing.T) {
	combined := combineTaskReports(
		TaskReport{Scenarios: 1, RecallAtK: 1, MeanLatencyMs: 10, P95LatencyMs: 10},
		TaskReport{Scenarios: 3, RecallAtK: 0.5, MeanLatencyMs: 40, P95LatencyMs: 60},
	)

	if combined.Scenarios != 4 {
		t.Fatalf("expected 4 scenarios, got %d", combined.Scenarios)
	}
	if combined.RecallAtK != 0.625 {
		t.Fatalf("expected weighted recall 0.625, got %f", combined.RecallAtK)
	}
	if combined.P95LatencyMs != 60 {
		t.Fatalf("expected max p95 latency 60, got %f", combined.P95LatencyMs)
	}
}
