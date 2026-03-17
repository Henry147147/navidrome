package nativeapi

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/navidrome/navidrome/model/request"
)

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
	if defaults.Enabled {
		t.Fatalf("expected autoplay to default to disabled")
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

func TestAutoPlaySettingsRoundTripEnabled(t *testing.T) {
	h := newTextRecommendationHarness(t, &captureRecommendationClient{})

	getReq := httptest.NewRequest(http.MethodGet, "/autoplay/settings", bytes.NewReader(nil))
	getReq = getReq.WithContext(request.WithUser(getReq.Context(), h.user))
	getRes := httptest.NewRecorder()
	h.native.handleGetAutoPlaySettings(getRes, getReq)
	if getRes.Code != http.StatusOK {
		t.Fatalf("expected GET status %d, got %d", http.StatusOK, getRes.Code)
	}

	var initial autoPlaySettings
	if err := json.Unmarshal(getRes.Body.Bytes(), &initial); err != nil {
		t.Fatalf("failed to decode initial settings: %v", err)
	}
	if initial.Enabled {
		t.Fatalf("expected initial enabled to be false")
	}

	payload := autoPlaySettings{
		Enabled:            true,
		Mode:               modeDiscoveryRecommendations,
		TextPrompt:         "night drive",
		ExcludePlaylistIDs: []string{"pl-1"},
		BatchSize:          autoPlayBatchMin,
	}
	body, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("failed to marshal payload: %v", err)
	}
	putReq := httptest.NewRequest(http.MethodPut, "/autoplay/settings", bytes.NewReader(body))
	putReq.Header.Set("Content-Type", "application/json")
	putReq = putReq.WithContext(request.WithUser(putReq.Context(), h.user))
	putRes := httptest.NewRecorder()
	h.native.handleUpdateAutoPlaySettings(putRes, putReq)
	if putRes.Code != http.StatusOK {
		t.Fatalf("expected PUT status %d, got %d: %s", http.StatusOK, putRes.Code, putRes.Body.String())
	}

	var updated autoPlaySettings
	if err := json.Unmarshal(putRes.Body.Bytes(), &updated); err != nil {
		t.Fatalf("failed to decode updated settings: %v", err)
	}
	if !updated.Enabled {
		t.Fatalf("expected updated enabled to be true")
	}
	if updated.Mode != modeDiscoveryRecommendations {
		t.Fatalf("expected mode %q, got %q", modeDiscoveryRecommendations, updated.Mode)
	}

	verifyReq := httptest.NewRequest(http.MethodGet, "/autoplay/settings", bytes.NewReader(nil))
	verifyReq = verifyReq.WithContext(request.WithUser(verifyReq.Context(), h.user))
	verifyRes := httptest.NewRecorder()
	h.native.handleGetAutoPlaySettings(verifyRes, verifyReq)
	if verifyRes.Code != http.StatusOK {
		t.Fatalf("expected verify GET status %d, got %d", http.StatusOK, verifyRes.Code)
	}
	if err := json.Unmarshal(verifyRes.Body.Bytes(), &updated); err != nil {
		t.Fatalf("failed to decode verified settings: %v", err)
	}
	if !updated.Enabled {
		t.Fatalf("expected persisted enabled to stay true")
	}
	if updated.TextPrompt != "night drive" {
		t.Fatalf("expected persisted text prompt, got %q", updated.TextPrompt)
	}
}
