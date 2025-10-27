package nativeapi

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"time"

	sq "github.com/Masterminds/squirrel"
	"github.com/go-chi/chi/v5"
	"github.com/navidrome/navidrome/conf"
	"github.com/navidrome/navidrome/log"
	"github.com/navidrome/navidrome/model"
	"github.com/navidrome/navidrome/model/request"
	"github.com/navidrome/navidrome/server/subsonic"
)

type recommendationRequestPayload struct {
	Limit     int      `json:"limit"`
	SongIDs   []string `json:"songIds"`
	Name      string   `json:"name"`
	Diversity *float64 `json:"diversity"`
}

type recommendationResponsePayload struct {
	Name     string            `json:"name"`
	Mode     string            `json:"mode"`
	TrackIDs []string          `json:"trackIds"`
	Warnings []string          `json:"warnings,omitempty"`
	Tracks   []model.MediaFile `json:"tracks"`
}

const (
	modeRecentRecommendations    = "recent"
	modeCustomRecommendations    = "custom"
	modeFavoritesRecommendations = "favorites"
	modeAllRecommendations       = "all"
	modeDiscoveryRecommendations = "discovery"

	recommendationSettingsKey = "recommendations.settings"

	mixLengthMin          = 10
	mixLengthMax          = 100
	seedRecencyMinDays    = 7
	seedRecencyMaxDays    = 120
	discoveryMinDiversity = 0.3
	favoritesBlendMin     = 0.1
)

type recommendationSettings struct {
	MixLength             int     `json:"mixLength"`
	BaseDiversity         float64 `json:"baseDiversity"`
	DiscoveryExploration  float64 `json:"discoveryExploration"`
	SeedRecencyWindowDays int     `json:"seedRecencyWindowDays"`
	FavoritesBlendWeight  float64 `json:"favoritesBlendWeight"`
}

func defaultRecommendationSettings() recommendationSettings {
	limit := conf.Server.Recommendations.DefaultLimit
	if limit < mixLengthMin {
		limit = mixLengthMin
	}
	if limit > mixLengthMax {
		limit = mixLengthMax
	}
	diversity := conf.Server.Recommendations.Diversity
	if diversity < 0 {
		diversity = 0
	}
	if diversity > 1 {
		diversity = 1
	}
	return recommendationSettings{
		MixLength:             limit,
		BaseDiversity:         diversity,
		DiscoveryExploration:  0.6,
		SeedRecencyWindowDays: 60,
		FavoritesBlendWeight:  0.85,
	}
}

func (s *recommendationSettings) applyDefaults(defaults recommendationSettings) {
	if s.MixLength == 0 {
		s.MixLength = defaults.MixLength
	}
	if s.DiscoveryExploration == 0 {
		s.DiscoveryExploration = defaults.DiscoveryExploration
	}
	if s.SeedRecencyWindowDays == 0 {
		s.SeedRecencyWindowDays = defaults.SeedRecencyWindowDays
	}
	if s.FavoritesBlendWeight == 0 {
		s.FavoritesBlendWeight = defaults.FavoritesBlendWeight
	}
}

func (s recommendationSettings) validate() error {
	if s.MixLength < mixLengthMin || s.MixLength > mixLengthMax {
		return fmt.Errorf("mixLength must be between %d and %d", mixLengthMin, mixLengthMax)
	}
	if s.BaseDiversity < 0 || s.BaseDiversity > 1 {
		return fmt.Errorf("baseDiversity must be between %.2f and %.2f", 0.0, 1.0)
	}
	if s.DiscoveryExploration < discoveryMinDiversity || s.DiscoveryExploration > 1 {
		return fmt.Errorf("discoveryExploration must be between %.2f and %.2f", discoveryMinDiversity, 1.0)
	}
	if s.SeedRecencyWindowDays < seedRecencyMinDays || s.SeedRecencyWindowDays > seedRecencyMaxDays {
		return fmt.Errorf("seedRecencyWindowDays must be between %d and %d", seedRecencyMinDays, seedRecencyMaxDays)
	}
	if s.FavoritesBlendWeight < favoritesBlendMin || s.FavoritesBlendWeight > 1 {
		return fmt.Errorf("favoritesBlendWeight must be between %.2f and %.2f", favoritesBlendMin, 1.0)
	}
	return nil
}

func (n *Router) addRecommendationRoutes(r chi.Router) {
	r.Route("/recommendations", func(r chi.Router) {
		r.Route("/settings", func(r chi.Router) {
			r.Get("/", n.handleGetRecommendationSettings)
			r.Put("/", n.handleUpdateRecommendationSettings)
		})
		r.Post("/recent", n.handleRecentRecommendations)
		r.Post("/favorites", n.handleFavoritesRecommendations)
		r.Post("/all", n.handleAllRecommendations)
		r.Post("/discovery", n.handleDiscoveryRecommendations)
		r.Post("/custom", n.handleCustomRecommendations)
	})
}

func (n *Router) handleGetRecommendationSettings(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	user, ok := request.UserFrom(ctx)
	if !ok {
		http.Error(w, "user not found in context", http.StatusUnauthorized)
		return
	}
	settings, err := n.getRecommendationSettings(ctx, user)
	if err != nil {
		log.Error(ctx, "Failed to load recommendation settings", "error", err)
		http.Error(w, "failed to load recommendation settings", http.StatusInternalServerError)
		return
	}
	writeJSON(w, http.StatusOK, settings)
}

func (n *Router) handleUpdateRecommendationSettings(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	user, ok := request.UserFrom(ctx)
	if !ok {
		http.Error(w, "user not found in context", http.StatusUnauthorized)
		return
	}
	var payload recommendationSettings
	if err := decodeJSON(r, &payload); err != nil {
		http.Error(w, "invalid JSON payload", http.StatusBadRequest)
		return
	}
	defaults := defaultRecommendationSettings()
	payload.applyDefaults(defaults)
	if err := payload.validate(); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	settings, err := n.saveRecommendationSettings(ctx, user, payload)
	if err != nil {
		log.Error(ctx, "Failed to persist recommendation settings", "error", err)
		http.Error(w, "failed to save recommendation settings", http.StatusInternalServerError)
		return
	}
	writeJSON(w, http.StatusOK, settings)
}

func (n *Router) getRecommendationSettings(ctx context.Context, user model.User) (recommendationSettings, error) {
	defaults := defaultRecommendationSettings()
	repo := n.ds.UserProps(ctx)
	value, err := repo.Get(user.ID, recommendationSettingsKey)
	if err != nil {
		if errors.Is(err, model.ErrNotFound) {
			return defaults, nil
		}
		return recommendationSettings{}, err
	}
	if strings.TrimSpace(value) == "" {
		return defaults, nil
	}
	var settings recommendationSettings
	if err := json.Unmarshal([]byte(value), &settings); err != nil {
		log.Warn(ctx, "Failed to decode stored recommendation settings", "error", err)
		return defaults, nil
	}
	settings.applyDefaults(defaults)
	if err := settings.validate(); err != nil {
		log.Warn(ctx, "Stored recommendation settings invalid, using defaults", "error", err)
		return defaults, nil
	}
	return settings, nil
}

func (n *Router) saveRecommendationSettings(ctx context.Context, user model.User, settings recommendationSettings) (recommendationSettings, error) {
	bytes, err := json.Marshal(settings)
	if err != nil {
		return recommendationSettings{}, err
	}
	if err := n.ds.UserProps(ctx).Put(user.ID, recommendationSettingsKey, string(bytes)); err != nil {
		return recommendationSettings{}, err
	}
	return settings, nil
}

func (n *Router) handleRecentRecommendations(w http.ResponseWriter, r *http.Request) {
	if n.recommender == nil {
		http.Error(w, "recommendation service unavailable", http.StatusServiceUnavailable)
		return
	}
	ctx := r.Context()
	user, ok := request.UserFrom(ctx)
	if !ok {
		http.Error(w, "user not found in context", http.StatusUnauthorized)
		return
	}
	settings, err := n.getRecommendationSettings(ctx, user)
	if err != nil {
		log.Error(ctx, "Failed to load recommendation settings", "error", err)
		http.Error(w, "failed to load recommendation settings", http.StatusInternalServerError)
		return
	}
	var payload recommendationRequestPayload
	if err := decodeJSON(r, &payload); err != nil {
		http.Error(w, "invalid JSON payload", http.StatusBadRequest)
		return
	}
	limit := normalizeLimit(payload.Limit, settings.MixLength)
	seeds, err := n.buildRecentSeeds(ctx, user, limit*2, settings)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if len(seeds) == 0 {
		seeds, err = n.buildLikedSeeds(ctx, user, limit*2)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
	}
	if len(seeds) == 0 {
		writeError(w, http.StatusBadRequest, "We need a little more listening history before we can build this mix.")
		return
	}
	resp, err := n.executeRecommendation(ctx, user, modeRecentRecommendations, payload.Name, seeds, limit, payload.Diversity, settings.BaseDiversity, 0)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

func (n *Router) handleFavoritesRecommendations(w http.ResponseWriter, r *http.Request) {
	if n.recommender == nil {
		http.Error(w, "recommendation service unavailable", http.StatusServiceUnavailable)
		return
	}
	ctx := r.Context()
	user, ok := request.UserFrom(ctx)
	if !ok {
		http.Error(w, "user not found in context", http.StatusUnauthorized)
		return
	}
	settings, err := n.getRecommendationSettings(ctx, user)
	if err != nil {
		log.Error(ctx, "Failed to load recommendation settings", "error", err)
		http.Error(w, "failed to load recommendation settings", http.StatusInternalServerError)
		return
	}
	var payload recommendationRequestPayload
	if err := decodeJSON(r, &payload); err != nil {
		http.Error(w, "invalid JSON payload", http.StatusBadRequest)
		return
	}
	limit := normalizeLimit(payload.Limit, settings.MixLength)
	seeds, err := n.buildLikedSeeds(ctx, user, limit*2)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if len(seeds) == 0 {
		writeError(w, http.StatusBadRequest, "Star or rate a few songs and try again.")
		return
	}
	resp, err := n.executeRecommendation(ctx, user, modeFavoritesRecommendations, payload.Name, seeds, limit, payload.Diversity, settings.BaseDiversity, 0)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

func (n *Router) handleAllRecommendations(w http.ResponseWriter, r *http.Request) {
	if n.recommender == nil {
		http.Error(w, "recommendation service unavailable", http.StatusServiceUnavailable)
		return
	}
	ctx := r.Context()
	user, ok := request.UserFrom(ctx)
	if !ok {
		http.Error(w, "user not found in context", http.StatusUnauthorized)
		return
	}
	settings, err := n.getRecommendationSettings(ctx, user)
	if err != nil {
		log.Error(ctx, "Failed to load recommendation settings", "error", err)
		http.Error(w, "failed to load recommendation settings", http.StatusInternalServerError)
		return
	}
	var payload recommendationRequestPayload
	if err := decodeJSON(r, &payload); err != nil {
		http.Error(w, "invalid JSON payload", http.StatusBadRequest)
		return
	}
	limit := normalizeLimit(payload.Limit, settings.MixLength)
	seeds, err := n.buildAllSeeds(ctx, user, limit*2, settings)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if len(seeds) == 0 {
		writeError(w, http.StatusBadRequest, "We didn't find enough signals to build this mix yet.")
		return
	}
	if len(seeds) > limit {
		seeds = seeds[:limit]
	}
	resp, err := n.executeRecommendation(ctx, user, modeAllRecommendations, payload.Name, seeds, limit, payload.Diversity, settings.BaseDiversity, 0)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

func (n *Router) handleDiscoveryRecommendations(w http.ResponseWriter, r *http.Request) {
	if n.recommender == nil {
		http.Error(w, "recommendation service unavailable", http.StatusServiceUnavailable)
		return
	}
	ctx := r.Context()
	user, ok := request.UserFrom(ctx)
	if !ok {
		http.Error(w, "user not found in context", http.StatusUnauthorized)
		return
	}
	settings, err := n.getRecommendationSettings(ctx, user)
	if err != nil {
		log.Error(ctx, "Failed to load recommendation settings", "error", err)
		http.Error(w, "failed to load recommendation settings", http.StatusInternalServerError)
		return
	}
	var payload recommendationRequestPayload
	if err := decodeJSON(r, &payload); err != nil {
		http.Error(w, "invalid JSON payload", http.StatusBadRequest)
		return
	}
	limit := normalizeLimit(payload.Limit, settings.MixLength)
	seeds, err := n.buildAllSeeds(ctx, user, limit*3, settings)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if len(seeds) == 0 {
		writeError(w, http.StatusBadRequest, "We need a little more listening data before exploring.")
		return
	}
	maxSeeds := limit * 3
	if len(seeds) > maxSeeds {
		seeds = seeds[:maxSeeds]
	}
	resp, err := n.executeRecommendation(ctx, user, modeDiscoveryRecommendations, payload.Name, seeds, limit, payload.Diversity, settings.DiscoveryExploration, discoveryMinDiversity)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

func (n *Router) handleCustomRecommendations(w http.ResponseWriter, r *http.Request) {
	if n.recommender == nil {
		http.Error(w, "recommendation service unavailable", http.StatusServiceUnavailable)
		return
	}
	ctx := r.Context()
	user, ok := request.UserFrom(ctx)
	if !ok {
		http.Error(w, "user not found in context", http.StatusUnauthorized)
		return
	}
	settings, err := n.getRecommendationSettings(ctx, user)
	if err != nil {
		log.Error(ctx, "Failed to load recommendation settings", "error", err)
		http.Error(w, "failed to load recommendation settings", http.StatusInternalServerError)
		return
	}
	var payload recommendationRequestPayload
	if err := decodeJSON(r, &payload); err != nil {
		http.Error(w, "invalid JSON payload", http.StatusBadRequest)
		return
	}
	if len(payload.SongIDs) == 0 {
		http.Error(w, "songIds is required", http.StatusBadRequest)
		return
	}
	limit := normalizeLimit(payload.Limit, settings.MixLength)
	seeds, err := n.buildCustomSeeds(ctx, user, payload.SongIDs)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if len(seeds) == 0 {
		writeError(w, http.StatusBadRequest, "Those songs aren't available right now. Try a different selection.")
		return
	}
	resp, err := n.executeRecommendation(ctx, user, modeCustomRecommendations, payload.Name, seeds, limit, payload.Diversity, settings.BaseDiversity, 0)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

func (n *Router) executeRecommendation(ctx context.Context, user model.User, mode string, name string, seeds []subsonic.RecommendationSeed, limit int, diversityOverride *float64, fallback float64, min float64) (recommendationResponsePayload, error) {
	req := subsonic.RecommendationRequest{
		UserID:    user.ID,
		UserName:  user.UserName,
		Limit:     limit,
		Mode:      mode,
		Seeds:     seeds,
		Diversity: normalizeDiversity(diversityOverride, fallback, min),
	}
	result, err := n.recommender.Recommend(ctx, mode, req)
	if err != nil {
		return recommendationResponsePayload{}, err
	}
	trackIDs := result.TrackIDs()
	if len(trackIDs) == 0 {
		trackIDs = fallbackTrackIDs(seeds, limit)
	}
	tracks, err := n.loadTracks(ctx, trackIDs)
	if err != nil {
		return recommendationResponsePayload{}, err
	}
	playlistName := name
	if strings.TrimSpace(playlistName) == "" {
		playlistName = defaultPlaylistName(mode)
	}
	return recommendationResponsePayload{
		Name:     playlistName,
		Mode:     mode,
		TrackIDs: trackIDs,
		Warnings: result.Warnings,
		Tracks:   tracks,
	}, nil
}

func (n *Router) buildRecentSeeds(ctx context.Context, user model.User, limit int, settings recommendationSettings) ([]subsonic.RecommendationSeed, error) {
	filter := sq.Gt{"play_date": time.Time{}}
	if settings.SeedRecencyWindowDays > 0 {
		cutoff := time.Now().Add(-time.Duration(settings.SeedRecencyWindowDays) * 24 * time.Hour)
		filter = sq.Gt{"play_date": cutoff}
	}
	options := model.QueryOptions{
		Sort:    "playDate",
		Order:   "desc",
		Max:     limit,
		Filters: filter,
	}
	options.Filters = withLibraryFilter(options.Filters, user)
	mfs, err := n.ds.MediaFile(ctx).GetAll(options)
	if err != nil {
		return nil, err
	}
	return seedsFromMediaFiles(mfs, "recent"), nil
}

func (n *Router) buildCustomSeeds(ctx context.Context, user model.User, songIDs []string) ([]subsonic.RecommendationSeed, error) {
	filters := withLibraryFilter(sq.Eq{"media_file.id": songIDs}, user)
	mfs, err := n.ds.MediaFile(ctx).GetAll(model.QueryOptions{Filters: filters, Max: len(songIDs)})
	if err != nil {
		return nil, err
	}
	order := make(map[string]int, len(songIDs))
	for idx, id := range songIDs {
		order[id] = idx
	}
	seeds := make([]subsonic.RecommendationSeed, 0, len(mfs))
	seen := make(map[string]struct{}, len(mfs))
	length := float64(len(order))
	if length == 0 {
		return []subsonic.RecommendationSeed{}, nil
	}
	for _, mf := range mfs {
		pos, ok := order[mf.ID]
		if !ok {
			continue
		}
		if _, dup := seen[mf.ID]; dup {
			continue
		}
		seen[mf.ID] = struct{}{}
		weight := 1.0 - (float64(pos) / (length + 1))
		if weight < 0.1 {
			weight = 0.1
		}
		seed := subsonic.RecommendationSeed{
			TrackID:  mf.ID,
			Weight:   weight,
			Source:   "custom",
			PlayedAt: mf.PlayDate,
		}
		seeds = append(seeds, seed)
	}
	sort.SliceStable(seeds, func(i, j int) bool {
		iPos := order[seeds[i].TrackID]
		jPos := order[seeds[j].TrackID]
		return iPos < jPos
	})
	return seeds, nil
}

func (n *Router) buildLikedSeeds(ctx context.Context, user model.User, limit int) ([]subsonic.RecommendationSeed, error) {
	preferenceFilter := sq.Or{
		sq.Eq{"starred": true},
		sq.Gt{"rating": 0},
	}
	filters := withLibraryFilter(preferenceFilter, user)
	options := model.QueryOptions{
		Sort:    "starred_at",
		Order:   "desc",
		Max:     limit,
		Filters: filters,
	}
	mfs, err := n.ds.MediaFile(ctx).GetAll(options)
	if err != nil {
		return nil, err
	}
	if len(mfs) == 0 {
		return []subsonic.RecommendationSeed{}, nil
	}
	return seedsFromMediaFiles(mfs, "favorites"), nil
}

func (n *Router) buildAllSeeds(ctx context.Context, user model.User, limit int, settings recommendationSettings) ([]subsonic.RecommendationSeed, error) {
	recent, err := n.buildRecentSeeds(ctx, user, limit, settings)
	if err != nil {
		return nil, err
	}
	favorites, err := n.buildLikedSeeds(ctx, user, limit)
	if err != nil {
		return nil, err
	}
	combined, seen := appendUniqueSeeds(nil, recent, nil, 1.0)
	combined, _ = appendUniqueSeeds(combined, favorites, seen, settings.FavoritesBlendWeight)
	return combined, nil
}

func (n *Router) loadTracks(ctx context.Context, ids []string) ([]model.MediaFile, error) {
	if len(ids) == 0 {
		return []model.MediaFile{}, nil
	}
	mfs, err := n.ds.MediaFile(ctx).GetAll(model.QueryOptions{
		Filters: sq.Eq{"media_file.id": ids},
		Max:     len(ids),
	})
	if err != nil {
		return nil, err
	}
	m := make(map[string]model.MediaFile, len(mfs))
	for _, mf := range mfs {
		m[mf.ID] = mf
	}
	ordered := make([]model.MediaFile, 0, len(ids))
	for _, id := range ids {
		if mf, ok := m[id]; ok {
			ordered = append(ordered, mf)
		}
	}
	return ordered, nil
}

func seedsFromMediaFiles(files model.MediaFiles, source string) []subsonic.RecommendationSeed {
	seeds := make([]subsonic.RecommendationSeed, 0, len(files))
	for idx, mf := range files {
		if mf.ID == "" {
			continue
		}
		weight := 1.0 / float64(idx+1)
		if weight < 0.1 {
			weight = 0.1
		}
		seeds = append(seeds, subsonic.RecommendationSeed{
			TrackID:  mf.ID,
			Weight:   weight,
			Source:   source,
			PlayedAt: mf.PlayDate,
		})
	}
	return seeds
}

func fallbackTrackIDs(seeds []subsonic.RecommendationSeed, limit int) []string {
	ids := make([]string, 0, len(seeds))
	for _, seed := range seeds {
		if seed.TrackID == "" {
			continue
		}
		ids = append(ids, seed.TrackID)
		if len(ids) >= limit {
			break
		}
	}
	return ids
}

func appendUniqueSeeds(dst []subsonic.RecommendationSeed, src []subsonic.RecommendationSeed, seen map[string]struct{}, scale float64) ([]subsonic.RecommendationSeed, map[string]struct{}) {
	if len(src) == 0 {
		return dst, seen
	}
	if seen == nil {
		seen = make(map[string]struct{}, len(dst)+len(src))
	}
	for _, seed := range src {
		id := seed.TrackID
		if id == "" {
			continue
		}
		if _, exists := seen[id]; exists {
			continue
		}
		seen[id] = struct{}{}
		if scale > 0 && scale != 1 {
			seed.Weight *= scale
			if seed.Weight < 0.05 {
				seed.Weight = 0.05
			}
		}
		dst = append(dst, seed)
	}
	return dst, seen
}

func withLibraryFilter(filter sq.Sqlizer, user model.User) sq.Sqlizer {
	if user.IsAdmin || len(user.Libraries) == 0 {
		return filter
	}
	ids := make([]int, 0, len(user.Libraries))
	for _, lib := range user.Libraries {
		ids = append(ids, lib.ID)
	}
	libraryFilter := sq.Eq{"media_file.library_id": ids}
	if filter == nil {
		return libraryFilter
	}
	switch existing := filter.(type) {
	case sq.And:
		return append(existing, libraryFilter)
	default:
		return sq.And{existing, libraryFilter}
	}
}

func defaultPlaylistName(mode string) string {
	switch mode {
	case modeRecentRecommendations:
		return "Recent Mix"
	case modeCustomRecommendations:
		return "Custom Mix"
	case modeFavoritesRecommendations:
		return "Favorites Mix"
	case modeAllRecommendations:
		return "All Metrics Mix"
	case modeDiscoveryRecommendations:
		return "Discovery Mix"
	default:
		return "Mix"
	}
}

func normalizeLimit(requested int, fallback int) int {
	limit := fallback
	if requested > 0 {
		limit = requested
	}
	if limit <= 0 {
		limit = conf.Server.Recommendations.DefaultLimit
	}
	if limit > 200 {
		limit = 200
	}
	return limit
}

func normalizeDiversity(override *float64, fallback float64, min float64) float64 {
	diversity := fallback
	if override != nil {
		diversity = *override
	}
	if diversity < 0 {
		diversity = 0
	}
	if diversity > 1 {
		diversity = 1
	}
	if diversity < min {
		diversity = min
	}
	return diversity
}

func decodeJSON(r *http.Request, dst interface{}) error {
	if r.Body == nil {
		return nil
	}
	if err := json.NewDecoder(r.Body).Decode(dst); err != nil && err.Error() != "EOF" {
		log.Error(r.Context(), "Failed to decode request body", "error", err)
		return err
	}
	return nil
}

func writeJSON(w http.ResponseWriter, status int, payload interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

func writeError(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, map[string]string{"message": message})
}
