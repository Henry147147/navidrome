package nativeapi

import (
	"testing"

	"github.com/navidrome/navidrome/model"
)

func TestRerankRecommendationTracksCalibrationBonusPrefersProfileMatch(t *testing.T) {
	profile := buildRecommendationTasteProfile([]model.MediaFile{
		{
			ID:     "seed-1",
			Artist: "Seed Artist",
			Genre:  "Jazz",
			Year:   1994,
			Tags: model.Tags{
				model.TagMood: []string{"Calm"},
			},
		},
	})

	candidates := []recommendationTrack{
		{
			MediaFile: model.MediaFile{
				ID:     "pop-track",
				Artist: "Pop Artist",
				Genre:  "Pop",
				Year:   2010,
				Path:   "pop-track.mp3",
				Album:  "Pop Album",
			},
		},
		{
			MediaFile: model.MediaFile{
				ID:     "jazz-track",
				Artist: "Jazz Artist",
				Genre:  "Jazz",
				Year:   1991,
				Path:   "jazz-track.mp3",
				Album:  "Jazz Album",
				Tags: model.Tags{
					model.TagMood: []string{"Calm"},
				},
			},
		},
	}

	reranked := rerankRecommendationTracks(candidates, map[string]float64{
		"pop-track":  1.00,
		"jazz-track": 1.00,
	}, profile)

	if len(reranked) != 2 {
		t.Fatalf("expected 2 reranked tracks, got %d", len(reranked))
	}
	if reranked[0].ID != "jazz-track" {
		t.Fatalf("expected jazz track to move ahead due to calibration bonus, got %#v", reranked)
	}
}

func TestRerankRecommendationTracksAppliesAlbumPenalty(t *testing.T) {
	candidates := []recommendationTrack{
		{
			MediaFile: model.MediaFile{
				ID:      "album-a-1",
				AlbumID: "album-a",
				Album:   "Album A",
				Path:    "album-a-1.mp3",
			},
		},
		{
			MediaFile: model.MediaFile{
				ID:      "album-a-2",
				AlbumID: "album-a",
				Album:   "Album A",
				Path:    "album-a-2.mp3",
			},
		},
		{
			MediaFile: model.MediaFile{
				ID:      "album-b-1",
				AlbumID: "album-b",
				Album:   "Album B",
				Path:    "album-b-1.mp3",
			},
		},
	}

	reranked := rerankRecommendationTracks(candidates, map[string]float64{
		"album-a-1": 1.00,
		"album-a-2": 0.99,
		"album-b-1": 0.99,
	}, recommendationTasteProfile{})

	if len(reranked) != 3 {
		t.Fatalf("expected 3 reranked tracks, got %d", len(reranked))
	}
	if reranked[1].ID != "album-b-1" {
		t.Fatalf("expected album penalty to push different album earlier, got %#v", reranked)
	}
}

func TestSelectRecommendationTracksSkipsDurationAndPathDuplicatesSilently(t *testing.T) {
	settings := defaultRecommendationSettings()
	settings.MinTrackDurationSeconds = 30
	settings.MaxTrackDurationSeconds = 300

	candidates := []recommendationTrack{
		{MediaFile: model.MediaFile{ID: "short", Duration: 10, Path: "dup.mp3"}},
		{MediaFile: model.MediaFile{ID: "good-a", Duration: 200, Path: "dup.mp3"}},
		{MediaFile: model.MediaFile{ID: "good-b", Duration: 220, Path: "good-b.mp3"}},
	}

	selected, warning := selectRecommendationTracks(candidates, 2, nil, dislikeSignals{}, settings)

	if warning != "" {
		t.Fatalf("expected no warning for duration/path filtering, got %q", warning)
	}
	if len(selected) != 2 {
		t.Fatalf("expected 2 selected tracks, got %#v", selected)
	}
	if selected[0].ID != "good-a" || selected[1].ID != "good-b" {
		t.Fatalf("unexpected selected tracks: %#v", selected)
	}
}
