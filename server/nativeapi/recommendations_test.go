package nativeapi

import (
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
	filteredTracks, filteredIDs, warning := filterDislikedTracks(tracks, ids, signals, 0.85)
	if len(filteredIDs) != 1 || filteredIDs[0] != "t4" {
		t.Fatalf("expected only t4 to remain, got ids %#v", filteredIDs)
	}
	if len(filteredTracks) != 1 || filteredTracks[0].ID != "t4" {
		t.Fatalf("expected only track t4, got %#v", filteredTracks)
	}
	if warning == "" {
		t.Fatalf("expected warning to mention filtered tracks")
	}
}

func TestFallbackTrackIDsSkipsDisliked(t *testing.T) {
	seeds := []subsonic.RecommendationSeed{
		{TrackID: "a"},
		{TrackID: "b"},
		{TrackID: "c"},
	}
	disliked := map[string]int{"b": 1}
	result := fallbackTrackIDs(seeds, 3, disliked)
	if len(result) != 2 {
		t.Fatalf("expected 2 fallback ids, got %d", len(result))
	}
	if result[0] != "a" || result[1] != "c" {
		t.Fatalf("unexpected fallback ids: %#v", result)
	}
}
