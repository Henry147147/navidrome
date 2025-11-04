package nativeapi

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/go-chi/chi/v5"
	"github.com/navidrome/navidrome/log"
	"github.com/navidrome/navidrome/model"
	"github.com/navidrome/navidrome/model/request"
)

const (
	autoPlaySettingsKey = "autoplay.settings"
	autoPlayBatchMin    = 5
	autoPlayBatchMax    = 50
	autoPlayModeText    = "text"
)

var autoPlayAllowedModes = map[string]struct{}{
	modeRecentRecommendations:    {},
	modeFavoritesRecommendations: {},
	modeAllRecommendations:       {},
	modeDiscoveryRecommendations: {},
	modeCustomRecommendations:    {},
	autoPlayModeText:             {},
}

type autoPlaySettings struct {
	Mode               string   `json:"mode"`
	TextPrompt         string   `json:"textPrompt,omitempty"`
	ExcludePlaylistIDs []string `json:"excludePlaylistIds"`
	BatchSize          int      `json:"batchSize"`
	DiversityOverride  *float64 `json:"diversityOverride,omitempty"`
}

func defaultAutoPlaySettings() autoPlaySettings {
	return autoPlaySettings{
		Mode:               modeRecentRecommendations,
		BatchSize:          autoPlayBatchMin,
		ExcludePlaylistIDs: []string{},
	}
}

func (s *autoPlaySettings) applyDefaults(defaults autoPlaySettings) {
	if trimmed := strings.ToLower(strings.TrimSpace(s.Mode)); trimmed != "" {
		s.Mode = trimmed
	}
	if s.Mode == "" {
		s.Mode = defaults.Mode
	}
	if s.BatchSize == 0 {
		s.BatchSize = defaults.BatchSize
	}
	if s.ExcludePlaylistIDs == nil {
		s.ExcludePlaylistIDs = []string{}
	}
	s.TextPrompt = strings.TrimSpace(s.TextPrompt)
}

func (s autoPlaySettings) validate() error {
	if _, ok := autoPlayAllowedModes[s.Mode]; !ok {
		return fmt.Errorf("invalid mode '%s'", s.Mode)
	}
	if s.BatchSize < autoPlayBatchMin || s.BatchSize > autoPlayBatchMax {
		return fmt.Errorf("batchSize must be between %d and %d", autoPlayBatchMin, autoPlayBatchMax)
	}
	if s.DiversityOverride != nil {
		if *s.DiversityOverride < 0 || *s.DiversityOverride > 1 {
			return fmt.Errorf("diversityOverride must be between %.2f and %.2f", 0.0, 1.0)
		}
	}
	return nil
}

func (n *Router) getAutoPlaySettings(ctx context.Context, user model.User) (autoPlaySettings, error) {
	defaults := defaultAutoPlaySettings()
	repo := n.ds.UserProps(ctx)
	value, err := repo.Get(user.ID, autoPlaySettingsKey)
	if err != nil {
		if errors.Is(err, model.ErrNotFound) {
			return defaults, nil
		}
		return autoPlaySettings{}, err
	}
	if strings.TrimSpace(value) == "" {
		return defaults, nil
	}
	var settings autoPlaySettings
	if err := json.Unmarshal([]byte(value), &settings); err != nil {
		log.Warn(ctx, "Failed to decode stored autoplay settings", "error", err)
		return defaults, nil
	}
	settings.applyDefaults(defaults)
	settings.ExcludePlaylistIDs = uniqueNonEmptyStrings(settings.ExcludePlaylistIDs)
	if err := settings.validate(); err != nil {
		log.Warn(ctx, "Stored autoplay settings invalid, using defaults", "error", err)
		return defaults, nil
	}
	return settings, nil
}

func (n *Router) saveAutoPlaySettings(ctx context.Context, user model.User, settings autoPlaySettings) (autoPlaySettings, error) {
	bytes, err := json.Marshal(settings)
	if err != nil {
		return autoPlaySettings{}, err
	}
	if err := n.ds.UserProps(ctx).Put(user.ID, autoPlaySettingsKey, string(bytes)); err != nil {
		return autoPlaySettings{}, err
	}
	return settings, nil
}

func (n *Router) handleGetAutoPlaySettings(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	user, ok := request.UserFrom(ctx)
	if !ok {
		http.Error(w, "user not found in context", http.StatusUnauthorized)
		return
	}
	settings, err := n.getAutoPlaySettings(ctx, user)
	if err != nil {
		log.Error(ctx, "Failed to load autoplay settings", "error", err)
		http.Error(w, "failed to load autoplay settings", http.StatusInternalServerError)
		return
	}
	writeJSON(w, http.StatusOK, settings)
}

func (n *Router) handleUpdateAutoPlaySettings(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	user, ok := request.UserFrom(ctx)
	if !ok {
		http.Error(w, "user not found in context", http.StatusUnauthorized)
		return
	}
	var payload autoPlaySettings
	if err := decodeJSON(r, &payload); err != nil {
		http.Error(w, "invalid JSON payload", http.StatusBadRequest)
		return
	}
	payload.ExcludePlaylistIDs = uniqueNonEmptyStrings(payload.ExcludePlaylistIDs)
	defaults := defaultAutoPlaySettings()
	payload.applyDefaults(defaults)
	if err := payload.validate(); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	settings, err := n.saveAutoPlaySettings(ctx, user, payload)
	if err != nil {
		log.Error(ctx, "Failed to persist autoplay settings", "error", err)
		http.Error(w, "failed to save autoplay settings", http.StatusInternalServerError)
		return
	}
	writeJSON(w, http.StatusOK, settings)
}

func (n *Router) addAutoPlayRoute(r chi.Router) {
	r.Route("/autoplay", func(r chi.Router) {
		r.Route("/settings", func(r chi.Router) {
			r.Get("/", n.handleGetAutoPlaySettings)
			r.Put("/", n.handleUpdateAutoPlaySettings)
		})
	})
}
