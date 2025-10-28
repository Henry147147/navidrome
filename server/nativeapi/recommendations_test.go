package nativeapi

import (
	"strings"
	"testing"

	"github.com/navidrome/navidrome/model"
	"github.com/navidrome/navidrome/server/subsonic"
)

func TestUniqueNonEmptyStrings(t *testing.T) {
	t.Run("returns nil for empty input", func(t *testing.T) {
		if result := uniqueNonEmptyStrings(nil); result != nil {
			t.Fatalf("expected nil, got %#v", result)
		}
	})

	t.Run("filters empty and duplicate values", func(t *testing.T) {
		input := []string{"", "  ", "id1", "id2", "id1", "ID2", "Id3"}
		result := uniqueNonEmptyStrings(input)
		expected := []string{"id1", "id2", "ID2", "Id3"}
		if len(result) != len(expected) {
			t.Fatalf("expected length %d, got %d (%#v)", len(expected), len(result), result)
		}
		for idx, value := range expected {
			if result[idx] != value {
				t.Fatalf("expected value %q at index %d, got %q", value, idx, result[idx])
			}
		}
	})
}

func TestSeedsFromMediaFilesSkipsLowRatings(t *testing.T) {
	files := model.MediaFiles{
		{ID: "keep-0", Annotations: model.Annotations{Rating: 0}},
		{ID: "skip-1", Annotations: model.Annotations{Rating: 1}},
		{ID: "skip-2", Annotations: model.Annotations{Rating: 2}},
		{ID: "keep-5", Annotations: model.Annotations{Rating: 5}},
	}
	seeds := seedsFromMediaFiles(files, "test")
	if len(seeds) != 2 {
		t.Fatalf("expected 2 seeds, got %d", len(seeds))
	}
	if seeds[0].TrackID != "keep-0" || seeds[1].TrackID != "keep-5" {
		t.Fatalf("unexpected seeds returned: %#v", seeds)
	}
}

func TestFilterDislikedTracksRemovesMatches(t *testing.T) {
	signals := dislikeSignals{
		trackRatings: map[string]int{"t1": 1},
		albumRatings: map[string]int{"a1": 1},
		artistRatings: map[string]int{
			"art1": 2,
		},
	}
	tracks := []model.MediaFile{
		{ID: "t1", AlbumID: "a1", ArtistID: "art1"},
		{ID: "t2", AlbumID: "a1"},
		{ID: "t3", ArtistID: "art1"},
		{ID: "t4"},
	}
	ids := []string{"t1", "t2", "t3", "t4"}
	filteredTracks, filteredIDs, warning := filterDislikedTracks(tracks, ids, signals, nil, 0.85)
	if len(filteredIDs) != 1 || filteredIDs[0] != "t4" {
		t.Fatalf("expected only t4 to remain, got ids %#v", filteredIDs)
	}
	if len(filteredTracks) != 1 || filteredTracks[0].ID != "t4" {
		t.Fatalf("expected only track t4, got %#v", filteredTracks)
	}
	if !strings.Contains(warning, "skipped because you rated them poorly") {
		t.Fatalf("expected warning to mention disliked skip, got %q", warning)
	}
}

func TestFallbackTrackIDsSkipsDisliked(t *testing.T) {
	seeds := []subsonic.RecommendationSeed{
		{TrackID: "a"},
		{TrackID: "b"},
		{TrackID: "c"},
	}
	blocked := map[string]struct{}{"b": {}}
	result := fallbackTrackIDs(seeds, 3, blocked)
	if len(result) != 2 {
		t.Fatalf("expected 2 fallback ids, got %d", len(result))
	}
	if result[0] != "a" || result[1] != "c" {
		t.Fatalf("unexpected fallback ids: %#v", result)
	}
}

func TestFilterBlockedTracksRemovesMatches(t *testing.T) {
	tracks := []model.MediaFile{
		{ID: "t1"},
		{ID: "t2"},
		{ID: "t3"},
	}
	ids := []string{"t1", "t2", "t3"}
	blocked := map[string]struct{}{"t2": {}, "t3": {}}
	filteredTracks, filteredIDs, warning := filterDislikedTracks(tracks, ids, dislikeSignals{}, blocked, 0.5)
	if len(filteredIDs) != 1 || filteredIDs[0] != "t1" {
		t.Fatalf("expected only t1 to remain, got %#v", filteredIDs)
	}
	if len(filteredTracks) != 1 || filteredTracks[0].ID != "t1" {
		t.Fatalf("expected only track t1, got %#v", filteredTracks)
	}
	if !strings.Contains(warning, "removed because they belong to excluded playlists") {
		t.Fatalf("expected warning to mention playlist exclusion, got %q", warning)
	}
}

func TestSelectionBaseWeightClamps(t *testing.T) {
	w := selectionBaseWeight(0, 3)
	if w != 1 {
		t.Fatalf("expected first selection weight 1, got %f", w)
	}
	low := selectionBaseWeight(10, 3)
	if low < 0.1 || low > 0.11 {
		t.Fatalf("expected weight to clamp near 0.1, got %f", low)
	}
}

func TestAlbumSeedWeightDecay(t *testing.T) {
	base := 0.9
	w0 := albumSeedWeight(base, 0)
	if w0 != clamp(base, 0.05, 1) {
		t.Fatalf("expected base weight with clamp, got %f", w0)
	}
	w3 := albumSeedWeight(base, 3)
	if !(w3 < base && w3 > 0.05) {
		t.Fatalf("expected decay between base and clamp floor, got %f", w3)
	}
}

func TestMakeCustomSeedClampsWeights(t *testing.T) {
	mf := model.MediaFile{ID: "track"}
	seed := makeCustomSeed(mf, 5, "custom")
	if seed.Weight > 1 {
		t.Fatalf("weight should clamp to 1, got %f", seed.Weight)
	}
	seedLow := makeCustomSeed(mf, 0.001, "custom")
	if seedLow.Weight < 0.05 {
		t.Fatalf("weight should clamp to >= 0.05, got %f", seedLow.Weight)
	}
}
