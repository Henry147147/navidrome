package nativeapi

import (
	"context"
	"encoding/json"
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
	modeRecentRecommendations = "recent"
	modeCustomRecommendations = "custom"
)

func (n *Router) addRecommendationRoutes(r chi.Router) {
	r.Route("/recommendations", func(r chi.Router) {
		r.Post("/recent", n.handleRecentRecommendations)
		r.Post("/custom", n.handleCustomRecommendations)
	})
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
	var payload recommendationRequestPayload
	if err := decodeJSON(r, &payload); err != nil {
		http.Error(w, "invalid JSON payload", http.StatusBadRequest)
		return
	}
	limit := normalizeLimit(payload.Limit)
	seeds, err := n.buildRecentSeeds(ctx, user, limit*2)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	resp, err := n.executeRecommendation(ctx, user, modeRecentRecommendations, payload.Name, seeds, limit, payload.Diversity)
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
	var payload recommendationRequestPayload
	if err := decodeJSON(r, &payload); err != nil {
		http.Error(w, "invalid JSON payload", http.StatusBadRequest)
		return
	}
	if len(payload.SongIDs) == 0 {
		http.Error(w, "songIds is required", http.StatusBadRequest)
		return
	}
	limit := normalizeLimit(payload.Limit)
	seeds, err := n.buildCustomSeeds(ctx, user, payload.SongIDs)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	resp, err := n.executeRecommendation(ctx, user, modeCustomRecommendations, payload.Name, seeds, limit, payload.Diversity)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

func (n *Router) executeRecommendation(ctx context.Context, user model.User, mode string, name string, seeds []subsonic.RecommendationSeed, limit int, diversityOverride *float64) (recommendationResponsePayload, error) {
	req := subsonic.RecommendationRequest{
		UserID:    user.ID,
		UserName:  user.UserName,
		Limit:     limit,
		Mode:      mode,
		Seeds:     seeds,
		Diversity: normalizeDiversity(diversityOverride),
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

func (n *Router) buildRecentSeeds(ctx context.Context, user model.User, limit int) ([]subsonic.RecommendationSeed, error) {
	options := model.QueryOptions{
		Sort:    "playDate",
		Order:   "desc",
		Max:     limit,
		Filters: sq.Gt{"play_date": time.Time{}},
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
	seeds := make([]subsonic.RecommendationSeed, 0, len(mfs))
	seen := make(map[string]struct{}, len(mfs))
	for _, mf := range mfs {
		if _, ok := seen[mf.ID]; ok {
			continue
		}
		seen[mf.ID] = struct{}{}
		seed := subsonic.RecommendationSeed{
			TrackID:  mf.ID,
			Weight:   1.0,
			Source:   "custom",
			PlayedAt: mf.PlayDate,
		}
		seeds = append(seeds, seed)
	}
	sort.SliceStable(seeds, func(i, j int) bool {
		return strings.ToLower(seeds[i].TrackID) < strings.ToLower(seeds[j].TrackID)
	})
	return seeds, nil
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
		return sq.And{libraryFilter}
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
	default:
		return "Mix"
	}
}

func normalizeLimit(limit int) int {
	if limit <= 0 {
		limit = conf.Server.Recommendations.DefaultLimit
	}
	if limit > 200 {
		limit = 200
	}
	return limit
}

func normalizeDiversity(override *float64) float64 {
	diversity := conf.Server.Recommendations.Diversity
	if override != nil {
		diversity = *override
	}
	if diversity < 0 {
		diversity = 0
	}
	if diversity > 1 {
		diversity = 1
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
