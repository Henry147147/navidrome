package nativeapi

import "testing"

func TestAutoPlaySettingsApplyDefaults(t *testing.T) {
	defaults := defaultAutoPlaySettings()
	settings := autoPlaySettings{
		Mode:      "  FAVORITES  ",
		BatchSize: 0,
	}
	settings.applyDefaults(defaults)
	if settings.Mode != modeFavoritesRecommendations {
		t.Fatalf("expected mode to normalize to %q, got %q", modeFavoritesRecommendations, settings.Mode)
	}
	if settings.BatchSize != defaults.BatchSize {
		t.Fatalf("expected batch size default %d, got %d", defaults.BatchSize, settings.BatchSize)
	}
	if settings.ExcludePlaylistIDs == nil {
		t.Fatalf("expected exclude list to initialize")
	}
}

func TestDefaultAutoPlayBatchMatchesMinimum(t *testing.T) {
	defaults := defaultAutoPlaySettings()
	if defaults.BatchSize != autoPlayBatchMin {
		t.Fatalf("expected default batch size %d, got %d", autoPlayBatchMin, defaults.BatchSize)
	}
}

func TestAutoPlaySettingsValidate(t *testing.T) {
	valid := autoPlaySettings{
		Mode:      modeAllRecommendations,
		BatchSize: autoPlayBatchMin,
	}
	if err := valid.validate(); err != nil {
		t.Fatalf("expected valid settings, got error %v", err)
	}

	invalidMode := autoPlaySettings{Mode: "unknown", BatchSize: 10}
	if err := invalidMode.validate(); err == nil {
		t.Fatalf("expected invalid mode error")
	}

	invalidBatch := autoPlaySettings{Mode: modeRecentRecommendations, BatchSize: autoPlayBatchMax + 1}
	if err := invalidBatch.validate(); err == nil {
		t.Fatalf("expected invalid batch size error")
	}

	div := 1.5
	invalidDiversity := autoPlaySettings{Mode: modeRecentRecommendations, BatchSize: 10, DiversityOverride: &div}
	if err := invalidDiversity.validate(); err == nil {
		t.Fatalf("expected invalid diversity error")
	}
}
