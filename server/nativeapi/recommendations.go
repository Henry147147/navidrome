package nativeapi

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
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
	"slices"
)

type recommendationRequestPayload struct {
	Limit              int      `json:"limit"`
	SongIDs            []string `json:"songIds"`
	Name               string   `json:"name"`
	Diversity          *float64 `json:"diversity"`
	ExcludeTrackIDs    []string `json:"excludeTrackIds"`
	ExcludePlaylistIDs []string `json:"excludePlaylistIds"`
	PositiveTrackIDs   []string `json:"positiveTrackIds"`
	NegativeTrackIDs   []string `json:"negativeTrackIds"`
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

	positiveSeedBoost = 1.12

	recommendationSettingsKey = "recommendations.settings"

	mixLengthMin           = 10
	mixLengthMax           = 100
	seedRecencyMinDays     = 7
	seedRecencyMaxDays     = 120
	discoveryMinDiversity  = 0.3
	favoritesBlendMin      = 0.1
	lowRatingDislikeMax    = 2
	dislikeRejectThreshold = 0.5
)

type recommendationSettings struct {
	MixLength             int     `json:"mixLength"`
	BaseDiversity         float64 `json:"baseDiversity"`
	DiscoveryExploration  float64 `json:"discoveryExploration"`
	SeedRecencyWindowDays int     `json:"seedRecencyWindowDays"`
	FavoritesBlendWeight  float64 `json:"favoritesBlendWeight"`
	LowRatingPenalty      float64 `json:"lowRatingPenalty"`
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
		LowRatingPenalty:      0.85,
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
	if s.LowRatingPenalty == 0 {
		s.LowRatingPenalty = defaults.LowRatingPenalty
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
	if s.LowRatingPenalty < 0.3 || s.LowRatingPenalty > 1 {
		return fmt.Errorf("lowRatingPenalty must be between %.2f and %.2f", 0.3, 1.0)
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
	seeds = n.addPositiveSeeds(ctx, user, seeds, payload.PositiveTrackIDs)
	if len(seeds) == 0 {
		writeError(w, http.StatusBadRequest, "We need a little more listening history before we can build this mix.")
		return
	}
	excludeIDs := combineExcludeTrackIDs(payload)
	resp, err := n.executeRecommendation(ctx, user, modeRecentRecommendations, payload.Name, seeds, limit, excludeIDs, payload.ExcludePlaylistIDs, payload.Diversity, settings.BaseDiversity, 0, settings)
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
	seeds = n.addPositiveSeeds(ctx, user, seeds, payload.PositiveTrackIDs)
	if len(seeds) == 0 {
		writeError(w, http.StatusBadRequest, "Star or rate a few songs and try again.")
		return
	}
	excludeIDs := combineExcludeTrackIDs(payload)
	resp, err := n.executeRecommendation(ctx, user, modeFavoritesRecommendations, payload.Name, seeds, limit, excludeIDs, payload.ExcludePlaylistIDs, payload.Diversity, settings.BaseDiversity, 0, settings)
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
	seeds = n.addPositiveSeeds(ctx, user, seeds, payload.PositiveTrackIDs)
	if len(seeds) == 0 {
		writeError(w, http.StatusBadRequest, "We didn't find enough signals to build this mix yet.")
		return
	}
	if len(seeds) > limit {
		seeds = seeds[:limit]
	}
	excludeIDs := combineExcludeTrackIDs(payload)
	resp, err := n.executeRecommendation(ctx, user, modeAllRecommendations, payload.Name, seeds, limit, excludeIDs, payload.ExcludePlaylistIDs, payload.Diversity, settings.BaseDiversity, 0, settings)
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
	seeds = n.addPositiveSeeds(ctx, user, seeds, payload.PositiveTrackIDs)
	if len(seeds) == 0 {
		writeError(w, http.StatusBadRequest, "We need a little more listening data before exploring.")
		return
	}
	maxSeeds := limit * 3
	if len(seeds) > maxSeeds {
		seeds = seeds[:maxSeeds]
	}
	excludeIDs := combineExcludeTrackIDs(payload)
	resp, err := n.executeRecommendation(ctx, user, modeDiscoveryRecommendations, payload.Name, seeds, limit, excludeIDs, payload.ExcludePlaylistIDs, payload.Diversity, settings.DiscoveryExploration, discoveryMinDiversity, settings)
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
	baseIDs := uniqueNonEmptyStrings(payload.SongIDs)
	seeds, err := n.buildCustomSeeds(ctx, user, baseIDs)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if len(seeds) == 0 {
		writeError(w, http.StatusBadRequest, "Those songs aren't available right now. Try a different selection.")
		return
	}
	seeds = n.addPositiveSeeds(ctx, user, seeds, payload.PositiveTrackIDs)
	excludeIDs := combineExcludeTrackIDs(payload)
	resp, err := n.executeRecommendation(ctx, user, modeCustomRecommendations, payload.Name, seeds, limit, excludeIDs, payload.ExcludePlaylistIDs, payload.Diversity, settings.BaseDiversity, 0, settings)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

func (n *Router) executeRecommendation(ctx context.Context, user model.User, mode string, name string, seeds []subsonic.RecommendationSeed, limit int, excludeTrackIDs []string, excludePlaylistIDs []string, diversityOverride *float64, fallback float64, min float64, settings recommendationSettings) (recommendationResponsePayload, error) {
	dislikes, err := n.buildDislikeSignals(ctx, user, settings)
	if err != nil {
		log.Warn(ctx, "Failed to load low-rating signals", "error", err)
	}
	var combinedWarnings []string
	playlistIDs := uniqueNonEmptyStrings(excludePlaylistIDs)
	playlistTrackIDs, playlistWarnings, err := n.collectPlaylistTrackIDs(ctx, playlistIDs)
	if err != nil {
		return recommendationResponsePayload{}, err
	}
	if len(playlistWarnings) > 0 {
		combinedWarnings = append(combinedWarnings, playlistWarnings...)
	}
	directExclude := uniqueNonEmptyStrings(excludeTrackIDs)
	combinedExclude := uniqueNonEmptyStrings(append(directExclude, playlistTrackIDs...))
	dislikedTrackIDs := dislikes.trackIDs()
	combinedExclude = uniqueNonEmptyStrings(append(combinedExclude, dislikedTrackIDs...))
	blocked := make(map[string]struct{}, len(combinedExclude))
	for _, id := range combinedExclude {
		blocked[id] = struct{}{}
	}
	req := subsonic.RecommendationRequest{
		UserID:             user.ID,
		UserName:           user.UserName,
		Limit:              limit,
		Mode:               mode,
		Seeds:              seeds,
		Diversity:          normalizeDiversity(diversityOverride, fallback, min),
		ExcludeTrackIDs:    combinedExclude,
		DislikedTrackIDs:   dislikedTrackIDs,
		DislikedArtistIDs:  dislikes.artistIDs(),
		DislikeStrength:    settings.LowRatingPenalty,
		ExcludePlaylistIDs: playlistIDs,
	}
	result, err := n.recommender.Recommend(ctx, mode, req)
	if err != nil {
		return recommendationResponsePayload{}, err
	}
	trackIDs := result.TrackIDs()
	if len(trackIDs) == 0 {
		trackIDs = fallbackTrackIDs(seeds, limit, blocked)
	}
	tracks, err := n.loadTracks(ctx, trackIDs)
	if err != nil {
		return recommendationResponsePayload{}, err
	}
	playlistName := name
	if strings.TrimSpace(playlistName) == "" {
		playlistName = defaultPlaylistName(mode)
	}
	filteredTracks, filteredIDs, filteredWarnings := filterDislikedTracks(tracks, trackIDs, dislikes, blocked, settings.LowRatingPenalty)
	if len(result.Warnings) > 0 {
		combinedWarnings = append(combinedWarnings, result.Warnings...)
	}
	if filteredWarnings != "" {
		combinedWarnings = append(combinedWarnings, filteredWarnings)
	}
	finalTrackIDs := filteredIDs
	finalTracks := filteredTracks
	if len(finalTrackIDs) < limit {
		additional := fallbackTrackIDs(seeds, limit, blocked)
		additional = difference(finalTrackIDs, additional)
		if len(additional) > 0 {
			extraIDs := make([]string, 0, limit-len(finalTrackIDs))
			existing := make(map[string]struct{}, len(finalTrackIDs))
			for _, id := range finalTrackIDs {
				existing[id] = struct{}{}
			}
			for _, id := range additional {
				if _, skip := blocked[id]; skip {
					continue
				}
				if _, dup := existing[id]; dup {
					continue
				}
				if len(finalTrackIDs)+len(extraIDs) >= limit {
					break
				}
				extraIDs = append(extraIDs, id)
			}
			if len(extraIDs) > 0 {
				extraTracks, loadErr := n.loadTracks(ctx, extraIDs)
				if loadErr == nil {
					extraTrackMap := make(map[string]model.MediaFile, len(extraTracks))
					for _, mf := range extraTracks {
						extraTrackMap[mf.ID] = mf
					}
					for _, id := range extraIDs {
						mf, ok := extraTrackMap[id]
						if !ok {
							continue
						}
						if _, skip := blocked[id]; skip {
							continue
						}
						if dislikes.shouldReject(mf, id, settings.LowRatingPenalty) {
							continue
						}
						finalTrackIDs = append(finalTrackIDs, id)
						finalTracks = append(finalTracks, mf)
						if len(finalTrackIDs) >= limit {
							break
						}
					}
				}
			}
		}
	}
	return recommendationResponsePayload{
		Name:     playlistName,
		Mode:     mode,
		TrackIDs: finalTrackIDs,
		Warnings: combinedWarnings,
		Tracks:   finalTracks,
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
	if len(songIDs) == 0 {
		return []subsonic.RecommendationSeed{}, nil
	}
	filters := withLibraryFilter(sq.Eq{"media_file.id": songIDs}, user)
	mfs, err := n.ds.MediaFile(ctx).GetAll(model.QueryOptions{Filters: filters, Max: len(songIDs)})
	if err != nil {
		return nil, err
	}
	trackCache := make(map[string]model.MediaFile, len(mfs))
	for _, mf := range mfs {
		trackCache[mf.ID] = mf
	}
	albumCache := make(map[string][]model.MediaFile)
	totalSelections := len(songIDs)
	seeds := make([]subsonic.RecommendationSeed, 0, totalSelections)
	seen := make(map[string]struct{}, len(trackCache))

	for idx, rawID := range songIDs {
		id := strings.TrimSpace(rawID)
		if id == "" {
			continue
		}
		baseWeight := selectionBaseWeight(idx, totalSelections)
		if track, ok := trackCache[id]; ok {
			if _, dup := seen[track.ID]; dup {
				continue
			}
			seeds = append(seeds, makeCustomSeed(track, baseWeight, "custom"))
			seen[track.ID] = struct{}{}
			continue
		}
		albumTracks, err := n.loadAlbumTracks(ctx, user, id, albumCache)
		if err != nil {
			return nil, err
		}
		if len(albumTracks) == 0 {
			continue
		}
		for pos, track := range albumTracks {
			if track.ID == "" {
				continue
			}
			if _, dup := seen[track.ID]; dup {
				continue
			}
			weight := albumSeedWeight(baseWeight, pos)
			seeds = append(seeds, makeCustomSeed(track, weight, "custom-album"))
			seen[track.ID] = struct{}{}
		}
	}
	if len(seeds) == 0 {
		return []subsonic.RecommendationSeed{}, nil
	}
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

func (n *Router) collectPlaylistTrackIDs(ctx context.Context, playlistIDs []string) ([]string, []string, error) {
	if len(playlistIDs) == 0 {
		return nil, nil, nil
	}
	uniqueIDs := uniqueNonEmptyStrings(playlistIDs)
	if len(uniqueIDs) == 0 {
		return nil, nil, nil
	}
	trackSet := make(map[string]struct{})
	warnings := make([]string, 0)
	repo := n.ds.Playlist(ctx)
	for _, playlistID := range uniqueIDs {
		displayName := playlistID
		playlist, err := repo.Get(playlistID)
		if err != nil {
			if errors.Is(err, model.ErrNotFound) {
				warnings = append(warnings, fmt.Sprintf("Playlist %s is unavailable for exclusions.", playlistID))
				continue
			}
			return nil, nil, err
		}
		if strings.TrimSpace(playlist.Name) != "" {
			displayName = playlist.Name
		}
		tracks, err := repo.Tracks(playlistID, false).GetAll(model.QueryOptions{})
		if err != nil {
			if errors.Is(err, model.ErrNotFound) {
				warnings = append(warnings, fmt.Sprintf("Playlist %s is unavailable for exclusions.", displayName))
				continue
			}
			return nil, nil, err
		}
		if len(tracks) == 0 {
			warnings = append(warnings, fmt.Sprintf("Playlist %s has no available tracks to exclude.", displayName))
			continue
		}
		for _, track := range tracks {
			id := strings.TrimSpace(track.MediaFileID)
			if id == "" {
				id = strings.TrimSpace(track.MediaFile.ID)
			}
			if id == "" {
				continue
			}
			trackSet[id] = struct{}{}
		}
	}
	if len(trackSet) == 0 {
		return nil, warnings, nil
	}
	ids := make([]string, 0, len(trackSet))
	for id := range trackSet {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids, warnings, nil
}

func (n *Router) loadAlbumTracks(ctx context.Context, user model.User, albumID string, cache map[string][]model.MediaFile) ([]model.MediaFile, error) {
	if albumID == "" {
		return nil, nil
	}
	if cached, ok := cache[albumID]; ok {
		return cached, nil
	}
	filters := withLibraryFilter(sq.Eq{"media_file.album_id": albumID}, user)
	tracks, err := n.ds.MediaFile(ctx).GetAll(model.QueryOptions{
		Filters: filters,
		Max:     500,
	})
	if err != nil {
		if errors.Is(err, model.ErrNotFound) {
			cache[albumID] = nil
			return nil, nil
		}
		return nil, err
	}
	sort.SliceStable(tracks, func(i, j int) bool {
		if tracks[i].DiscNumber != tracks[j].DiscNumber {
			return tracks[i].DiscNumber < tracks[j].DiscNumber
		}
		if tracks[i].TrackNumber != tracks[j].TrackNumber {
			return tracks[i].TrackNumber < tracks[j].TrackNumber
		}
		return strings.Compare(tracks[i].OrderTitle, tracks[j].OrderTitle) < 0
	})
	cache[albumID] = tracks
	return tracks, nil
}

func makeCustomSeed(mf model.MediaFile, weight float64, source string) subsonic.RecommendationSeed {
	if weight > 1 {
		weight = 1
	}
	if weight < 0.05 {
		weight = 0.05
	}
	return subsonic.RecommendationSeed{
		TrackID:  mf.ID,
		Weight:   weight,
		Source:   source,
		PlayedAt: mf.PlayDate,
	}
}

func selectionBaseWeight(idx int, total int) float64 {
	if total <= 0 {
		return 1
	}
	weight := 1.0 - (float64(idx) / float64(total+1))
	if weight < 0.1 {
		return 0.1
	}
	if weight > 1 {
		return 1
	}
	return weight
}

func albumSeedWeight(base float64, position int) float64 {
	if base <= 0 {
		base = 0.1
	}
	if position <= 0 {
		return clamp(base, 0.05, 1)
	}
	decay := math.Pow(0.85, float64(position))
	return clamp(base*decay, 0.05, 1)
}

func seedsFromMediaFiles(files model.MediaFiles, source string) []subsonic.RecommendationSeed {
	seeds := make([]subsonic.RecommendationSeed, 0, len(files))
	position := 0
	for _, mf := range files {
		if mf.ID == "" {
			continue
		}
		if mf.Rating > 0 && mf.Rating <= lowRatingDislikeMax {
			continue
		}
		weight := 1.0 / float64(position+1)
		if weight < 0.1 {
			weight = 0.1
		}
		seeds = append(seeds, subsonic.RecommendationSeed{
			TrackID:  mf.ID,
			Weight:   weight,
			Source:   source,
			PlayedAt: mf.PlayDate,
		})
		position++
	}
	return seeds
}

func fallbackTrackIDs(seeds []subsonic.RecommendationSeed, limit int, blocked map[string]struct{}) []string {
	ids := make([]string, 0, len(seeds))
	seen := make(map[string]struct{}, len(seeds))
	for _, seed := range seeds {
		if seed.TrackID == "" {
			continue
		}
		if blocked != nil {
			if _, skip := blocked[seed.TrackID]; skip {
				continue
			}
		}
		if _, dup := seen[seed.TrackID]; dup {
			continue
		}
		seen[seed.TrackID] = struct{}{}
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
		for _, seed := range dst {
			if id := strings.TrimSpace(seed.TrackID); id != "" {
				seen[id] = struct{}{}
			}
		}
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

func combineExcludeTrackIDs(payload recommendationRequestPayload) []string {
	if len(payload.ExcludeTrackIDs) == 0 && len(payload.NegativeTrackIDs) == 0 {
		return nil
	}
	combined := make([]string, 0, len(payload.ExcludeTrackIDs)+len(payload.NegativeTrackIDs))
	combined = append(combined, payload.ExcludeTrackIDs...)
	combined = append(combined, payload.NegativeTrackIDs...)
	return uniqueNonEmptyStrings(combined)
}

func seedSeenSet(seeds []subsonic.RecommendationSeed) map[string]struct{} {
	if len(seeds) == 0 {
		return nil
	}
	seen := make(map[string]struct{}, len(seeds))
	for _, seed := range seeds {
		if id := strings.TrimSpace(seed.TrackID); id != "" {
			seen[id] = struct{}{}
		}
	}
	return seen
}

func (n *Router) addPositiveSeeds(ctx context.Context, user model.User, seeds []subsonic.RecommendationSeed, trackIDs []string) []subsonic.RecommendationSeed {
	ids := uniqueNonEmptyStrings(trackIDs)
	if len(ids) == 0 {
		return seeds
	}
	extras, err := n.buildCustomSeeds(ctx, user, ids)
	if err != nil {
		log.Warn(ctx, "Failed to build positive feedback seeds", "error", err)
		return seeds
	}
	if len(extras) == 0 {
		return seeds
	}
	seen := seedSeenSet(seeds)
	enriched, _ := appendUniqueSeeds(seeds, extras, seen, positiveSeedBoost)
	return enriched
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

func uniqueNonEmptyStrings(values []string) []string {
	if len(values) == 0 {
		return nil
	}
	seen := make(map[string]struct{}, len(values))
	result := make([]string, 0, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" {
			continue
		}
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		result = append(result, value)
	}
	if len(result) == 0 {
		return nil
	}
	return result
}

const (
	dislikeTrackStrongMultiplier    = 1.25
	dislikeTrackModerateMultiplier  = 0.95
	dislikeAlbumStrongMultiplier    = 0.95
	dislikeAlbumModerateMultiplier  = 0.7
	dislikeArtistStrongMultiplier   = 0.85
	dislikeArtistModerateMultiplier = 0.65
)

type dislikeSignals struct {
	trackRatings  map[string]int
	albumRatings  map[string]int
	artistRatings map[string]int
}

func (s dislikeSignals) empty() bool {
	return len(s.trackRatings) == 0 && len(s.albumRatings) == 0 && len(s.artistRatings) == 0
}

func (s dislikeSignals) trackIDs() []string {
	if len(s.trackRatings) == 0 {
		return nil
	}
	ids := make([]string, 0, len(s.trackRatings))
	for id := range s.trackRatings {
		ids = append(ids, id)
	}
	slices.Sort(ids)
	return ids
}

func (s dislikeSignals) artistIDs() []string {
	if len(s.artistRatings) == 0 {
		return nil
	}
	ids := make([]string, 0, len(s.artistRatings))
	for id := range s.artistRatings {
		ids = append(ids, id)
	}
	slices.Sort(ids)
	return ids
}

func (s *dislikeSignals) recordTrack(id string, rating int) {
	if id == "" {
		return
	}
	if s.trackRatings == nil {
		s.trackRatings = make(map[string]int)
	}
	if existing, ok := s.trackRatings[id]; !ok || rating < existing {
		s.trackRatings[id] = rating
	}
}

func (s *dislikeSignals) recordAlbum(id string, rating int) {
	if id == "" {
		return
	}
	if s.albumRatings == nil {
		s.albumRatings = make(map[string]int)
	}
	if existing, ok := s.albumRatings[id]; !ok || rating < existing {
		s.albumRatings[id] = rating
	}
}

func (s *dislikeSignals) recordArtist(id string, rating int) {
	if id == "" {
		return
	}
	if s.artistRatings == nil {
		s.artistRatings = make(map[string]int)
	}
	if existing, ok := s.artistRatings[id]; !ok || rating < existing {
		s.artistRatings[id] = rating
	}
}

func (s dislikeSignals) shouldReject(track model.MediaFile, trackID string, scale float64) bool {
	if s.empty() || scale <= 0 {
		return false
	}
	if trackID == "" {
		trackID = track.ID
	}
	if trackID != "" {
		if rating, ok := s.trackRatings[trackID]; ok && rating > 0 && rating <= lowRatingDislikeMax {
			return true
		}
	}
	scale = clamp(scale, 0.0, 1.0)
	if pen := s.albumPenalty(track.AlbumID, scale); pen >= dislikeRejectThreshold {
		return true
	}
	if pen := s.artistPenalty(track.AlbumArtistID, scale); pen >= dislikeRejectThreshold {
		return true
	}
	if pen := s.artistPenalty(track.ArtistID, scale); pen >= dislikeRejectThreshold {
		return true
	}
	for _, artistID := range track.Participants.AllIDs() {
		if pen := s.artistPenalty(artistID, scale); pen >= dislikeRejectThreshold {
			return true
		}
	}
	return false
}

func (s dislikeSignals) albumPenalty(id string, scale float64) float64 {
	if id == "" {
		return 0
	}
	rating, ok := s.albumRatings[id]
	if !ok {
		return 0
	}
	return penaltyFromRating(rating, scale, dislikeAlbumStrongMultiplier, dislikeAlbumModerateMultiplier)
}

func (s dislikeSignals) artistPenalty(id string, scale float64) float64 {
	if id == "" {
		return 0
	}
	rating, ok := s.artistRatings[id]
	if !ok {
		return 0
	}
	return penaltyFromRating(rating, scale, dislikeArtistStrongMultiplier, dislikeArtistModerateMultiplier)
}

func penaltyFromRating(rating int, scale float64, strongMultiplier float64, moderateMultiplier float64) float64 {
	if rating <= 0 || rating > lowRatingDislikeMax {
		return 0
	}
	var multiplier float64
	if rating == 1 {
		multiplier = strongMultiplier
	} else {
		multiplier = moderateMultiplier
	}
	value := scale * multiplier
	return math.Min(1.0, value)
}

func clamp(value float64, minValue float64, maxValue float64) float64 {
	if value < minValue {
		return minValue
	}
	if value > maxValue {
		return maxValue
	}
	return value
}

func filterDislikedTracks(tracks []model.MediaFile, trackIDs []string, signals dislikeSignals, blocked map[string]struct{}, scale float64) ([]model.MediaFile, []string, string) {
	if len(trackIDs) == 0 {
		return tracks, trackIDs, ""
	}
	trackMap := make(map[string]model.MediaFile, len(tracks))
	for _, mf := range tracks {
		trackMap[mf.ID] = mf
	}
	filteredIDs := make([]string, 0, len(trackIDs))
	filteredTracks := make([]model.MediaFile, 0, len(tracks))
	blockedRemoved := 0
	dislikedRemoved := 0
	for _, id := range trackIDs {
		mf, ok := trackMap[id]
		if !ok {
			continue
		}
		if blocked != nil {
			if _, skip := blocked[id]; skip {
				blockedRemoved++
				continue
			}
		}
		if signals.shouldReject(mf, id, scale) {
			dislikedRemoved++
			continue
		}
		filteredIDs = append(filteredIDs, id)
		filteredTracks = append(filteredTracks, mf)
	}
	if blockedRemoved == 0 && dislikedRemoved == 0 {
		return filteredTracks, filteredIDs, ""
	}
	parts := make([]string, 0, 2)
	if blockedRemoved > 0 {
		parts = append(parts, fmt.Sprintf("%d tracks removed because they belong to excluded playlists.", blockedRemoved))
	}
	if dislikedRemoved > 0 {
		parts = append(parts, fmt.Sprintf("%d tracks skipped because you rated them poorly.", dislikedRemoved))
	}
	warning := strings.Join(parts, " ")
	return filteredTracks, filteredIDs, warning
}

func (n *Router) buildDislikeSignals(ctx context.Context, user model.User, settings recommendationSettings) (dislikeSignals, error) {
	if settings.LowRatingPenalty <= 0 {
		return dislikeSignals{}, nil
	}
	baseFilters := sq.And{
		sq.Gt{"rating": 0},
		sq.LtOrEq{"rating": lowRatingDislikeMax},
	}
	filters := withLibraryFilter(baseFilters, user)
	files, err := n.ds.MediaFile(ctx).GetAll(model.QueryOptions{
		Sort:    "rating",
		Order:   "asc",
		Filters: filters,
	})
	if err != nil {
		if errors.Is(err, model.ErrNotFound) {
			return dislikeSignals{}, nil
		}
		return dislikeSignals{}, err
	}
	signals := dislikeSignals{}
	for _, mf := range files {
		rating := mf.Rating
		if rating <= 0 || rating > lowRatingDislikeMax {
			continue
		}
		signals.recordTrack(mf.ID, rating)
		signals.recordAlbum(mf.AlbumID, rating)
		signals.recordArtist(mf.AlbumArtistID, rating)
		signals.recordArtist(mf.ArtistID, rating)
		for _, artistID := range mf.Participants.AllIDs() {
			signals.recordArtist(artistID, rating)
		}
	}
	return signals, nil
}

func difference(existing []string, candidates []string) []string {
	if len(candidates) == 0 {
		return candidates
	}
	seen := make(map[string]struct{}, len(existing))
	for _, id := range existing {
		seen[id] = struct{}{}
	}
	filtered := make([]string, 0, len(candidates))
	for _, id := range candidates {
		if _, ok := seen[id]; ok {
			continue
		}
		filtered = append(filtered, id)
	}
	return filtered
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
