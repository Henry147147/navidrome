package subsonic

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"

	sq "github.com/Masterminds/squirrel"
	"github.com/navidrome/navidrome/conf"
	"github.com/navidrome/navidrome/log"
	"github.com/navidrome/navidrome/model"
	"github.com/navidrome/navidrome/server/subsonic/filter"
	"github.com/navidrome/navidrome/server/subsonic/responses"
	"github.com/navidrome/navidrome/utils/req"
)

const (
	recommendationRecentMode    = "recent"
	recommendationPlaylistMode  = "playlists"
	recommendationAllMode       = "all"
	recommendationDiscoveryMode = "discovery"
)

/* Creates playlist from recent listens */
func (api *Router) MakePlaylistFromRecentListens(r *http.Request) (*responses.Subsonic, error) {
	ctx := r.Context()
	limit := api.recommendationLimit(r)
	seeds, err := api.recentSeeds(ctx, limit*2)
	if err != nil {
		log.Error(r, "Failed to build recent seeds", err)
		return nil, err
	}
	playlist, warnings, err := api.generateRecommendationPlaylist(ctx, recommendationRecentMode, seeds, limit, conf.Server.Recommendations.Diversity)
	if err != nil {
		log.Error(r, "Recommendation service failed", err)
		return nil, err
	}
	api.logRecommendationWarnings(r, recommendationRecentMode, warnings)
	response := newResponse()
	response.Playlist = playlist
	return response, nil
}

/* Creates playlist from other playlists */
func (api *Router) MakePlaylistFromOtherPlaylists(r *http.Request) (*responses.Subsonic, error) {
	ctx := r.Context()
	limit := api.recommendationLimit(r)
	seeds, err := api.playlistSeeds(ctx, limit*3)
	if err != nil {
		log.Error(r, "Failed to build playlist seeds", err)
		return nil, err
	}
	playlist, warnings, err := api.generateRecommendationPlaylist(ctx, recommendationPlaylistMode, seeds, limit, conf.Server.Recommendations.Diversity)
	if err != nil {
		log.Error(r, "Recommendation service failed", err)
		return nil, err
	}
	api.logRecommendationWarnings(r, recommendationPlaylistMode, warnings)
	response := newResponse()
	response.Playlist = playlist
	return response, nil
}

/* Makes playlist from both the recent and the stars/hearts */
func (api *Router) MakePlaylistFromAllMetrics(r *http.Request) (*responses.Subsonic, error) {
	ctx := r.Context()
	limit := api.recommendationLimit(r)
	seeds, err := api.combinedSeeds(ctx, limit*4, 0.2)
	if err != nil {
		log.Error(r, "Failed to gather combined seeds", err)
		return nil, err
	}
	playlist, warnings, err := api.generateRecommendationPlaylist(ctx, recommendationAllMode, seeds, limit, conf.Server.Recommendations.Diversity)
	if err != nil {
		log.Error(r, "Recommendation service failed", err)
		return nil, err
	}
	api.logRecommendationWarnings(r, recommendationAllMode, warnings)
	response := newResponse()
	response.Playlist = playlist
	return response, nil
}

/* Creates playlist from all metrics, but slightly perturbs it so its more of a discovery playlist */
func (api *Router) MakeDiscoveryPlaylist(r *http.Request) (*responses.Subsonic, error) {
	ctx := r.Context()
	limit := api.recommendationLimit(r)
	seeds, err := api.combinedSeeds(ctx, limit*4, 0.4)
	if err != nil {
		log.Error(r, "Failed to gather discovery seeds", err)
		return nil, err
	}
	diversity := conf.Server.Recommendations.Diversity
	if diversity < 0.3 {
		diversity = 0.3
	}
	playlist, warnings, err := api.generateRecommendationPlaylist(ctx, recommendationDiscoveryMode, seeds, limit, diversity)
	if err != nil {
		log.Error(r, "Recommendation service failed", err)
		return nil, err
	}
	api.logRecommendationWarnings(r, recommendationDiscoveryMode, warnings)
	response := newResponse()
	response.Playlist = playlist
	return response, nil
}

func (api *Router) recommendationLimit(r *http.Request) int {
	p := req.Params(r)
	requested := p.IntOr("count", conf.Server.Recommendations.DefaultLimit)
	if requested <= 0 {
		requested = conf.Server.Recommendations.DefaultLimit
	}
	if requested > 500 {
		requested = 500
	}
	return requested
}

func (api *Router) generateRecommendationPlaylist(ctx context.Context, mode string, seeds []RecommendationSeed, limit int, diversity float64) (*responses.PlaylistWithSongs, []string, error) {
	user := getUser(ctx)
	if len(seeds) == 0 {
		return api.playlistFromSeeds(ctx, user.UserName, mode, nil, limit), nil, nil
	}
	req := RecommendationRequest{
		UserID:    user.ID,
		UserName:  user.UserName,
		Limit:     limit,
		Mode:      mode,
		Seeds:     seeds,
		Diversity: diversity,
	}
	resp, err := api.recommender.Recommend(ctx, mode, req)
	if err != nil {
		return nil, nil, err
	}
	trackIDs := resp.TrackIDs()
	if len(trackIDs) == 0 {
		trackIDs = compactSeedIDs(seeds, limit)
	}
	files, err := api.loadMediaFiles(ctx, trackIDs)
	if err != nil {
		return nil, nil, err
	}
	entries := make([]responses.Child, 0, len(trackIDs))
	var durationSeconds float64
	for _, id := range trackIDs {
		mf, ok := files[id]
		if !ok {
			continue
		}
		entries = append(entries, childFromMediaFile(ctx, mf))
		durationSeconds += float64(mf.Duration)
	}
	playlist := api.playlistFromSeeds(ctx, user.UserName, mode, entries, limit)
	playlist.SongCount = int32(len(entries))
	playlist.Duration = int32(durationSeconds)
	return playlist, resp.Warnings, nil
}

func (api *Router) playlistFromSeeds(ctx context.Context, owner string, mode string, entries []responses.Child, limit int) *responses.PlaylistWithSongs {
	now := time.Now()
	name := fmt.Sprintf("%s mix", strings.Title(mode))
	return &responses.PlaylistWithSongs{
		Playlist: responses.Playlist{
			Id:        fmt.Sprintf("recommender:%s", mode),
			Name:      name,
			Public:    false,
			Owner:     owner,
			Created:   now,
			Changed:   now,
			SongCount: int32(len(entries)),
		},
		Entry: entries,
	}
}

func (api *Router) loadMediaFiles(ctx context.Context, ids []string) (map[string]model.MediaFile, error) {
	if len(ids) == 0 {
		return map[string]model.MediaFile{}, nil
	}
	mfs, err := api.ds.MediaFile(ctx).GetAll(model.QueryOptions{
		Filters: sq.Eq{"media_file.id": ids},
		Max:     len(ids),
	})
	if err != nil {
		return nil, err
	}
	result := make(map[string]model.MediaFile, len(mfs))
	for _, mf := range mfs {
		result[mf.ID] = mf
	}
	return result, nil
}

func (api *Router) recentSeeds(ctx context.Context, limit int) ([]RecommendationSeed, error) {
	options := model.QueryOptions{
		Sort:    "playDate",
		Order:   "desc",
		Max:     limit,
		Filters: sq.Gt{"play_date": time.Time{}},
	}
	mfs, err := api.ds.MediaFile(ctx).GetAll(options)
	if err != nil {
		return nil, err
	}
	return seedsFromMediaFiles(mfs, "recent"), nil
}

func (api *Router) favoriteSeeds(ctx context.Context, limit int) ([]RecommendationSeed, error) {
	options := filter.ByStarred()
	options.Max = limit
	mfs, err := api.ds.MediaFile(ctx).GetAll(options)
	if err != nil {
		return nil, err
	}
	return seedsFromMediaFiles(mfs, "favorite"), nil
}

func (api *Router) playlistSeeds(ctx context.Context, limit int) ([]RecommendationSeed, error) {
	playlists, err := api.ds.Playlist(ctx).GetAll(model.QueryOptions{Sort: "updated_at", Order: "desc", Max: 10})
	if err != nil {
		return nil, err
	}
	seen := make(map[string]struct{})
	var seeds []RecommendationSeed
	for _, pls := range playlists {
		tracks, err := api.ds.Playlist(ctx).Tracks(pls.ID, false).GetAll(model.QueryOptions{Max: limit})
		if err != nil {
			return nil, err
		}
		for _, track := range tracks {
			if track.MediaFileID == "" {
				continue
			}
			if _, ok := seen[track.MediaFileID]; ok {
				continue
			}
			seen[track.MediaFileID] = struct{}{}
			seed := RecommendationSeed{TrackID: track.MediaFileID, Weight: 0.7, Source: "playlist"}
			if track.MediaFile.PlayDate != nil {
				seed.PlayedAt = track.MediaFile.PlayDate
			}
			seeds = append(seeds, seed)
			if len(seeds) >= limit {
				return seeds, nil
			}
		}
	}
	return seeds, nil
}

func (api *Router) combinedSeeds(ctx context.Context, limit int, playlistWeight float64) ([]RecommendationSeed, error) {
	recentSeeds, err := api.recentSeeds(ctx, limit)
	if err != nil {
		return nil, err
	}
	favoriteSeeds, err := api.favoriteSeeds(ctx, limit)
	if err != nil {
		return nil, err
	}
	playlistSeeds, err := api.playlistSeeds(ctx, limit)
	if err != nil {
		return nil, err
	}
	seen := make(map[string]struct{})
	seeds := make([]RecommendationSeed, 0, limit)
	seeds = appendUniqueSeeds(seeds, recentSeeds, seen)
	seeds = appendUniqueSeeds(seeds, scaleSeeds(favoriteSeeds, 1.2), seen)
	seeds = appendUniqueSeeds(seeds, scaleSeeds(playlistSeeds, playlistWeight), seen)
	if len(seeds) > limit {
		seeds = seeds[:limit]
	}
	return seeds, nil
}

func appendUniqueSeeds(dst []RecommendationSeed, src []RecommendationSeed, seen map[string]struct{}) []RecommendationSeed {
	for _, seed := range src {
		if seed.TrackID == "" {
			continue
		}
		if _, ok := seen[seed.TrackID]; ok {
			continue
		}
		seen[seed.TrackID] = struct{}{}
		dst = append(dst, seed)
	}
	return dst
}

func scaleSeeds(seeds []RecommendationSeed, factor float64) []RecommendationSeed {
	if factor == 0 {
		return seeds
	}
	for i := range seeds {
		seeds[i].Weight *= factor
	}
	return seeds
}

func seedsFromMediaFiles(files model.MediaFiles, source string) []RecommendationSeed {
	seeds := make([]RecommendationSeed, 0, len(files))
	for idx, mf := range files {
		weight := 1.0 / float64(idx+1)
		if weight < 0.1 {
			weight = 0.1
		}
		seed := RecommendationSeed{
			TrackID:  mf.ID,
			Weight:   weight,
			Source:   source,
			PlayedAt: mf.PlayDate,
		}
		seeds = append(seeds, seed)
	}
	return seeds
}

func compactSeedIDs(seeds []RecommendationSeed, limit int) []string {
	ids := make([]string, 0, min(limit, len(seeds)))
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

func (api *Router) logRecommendationWarnings(r *http.Request, mode string, warnings []string) {
	if len(warnings) == 0 {
		return
	}
	for _, warning := range warnings {
		log.Warn(r, "Recommendation warning", "mode", mode, "warning", warning)
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
