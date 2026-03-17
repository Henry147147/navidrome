package nativeapi

import (
	"context"
	"fmt"
	"math"
	"strings"

	"github.com/navidrome/navidrome/model"
	"github.com/navidrome/navidrome/server/subsonic"
)

type recommendationTasteProfile struct {
	artists map[string]float64
	genres  map[string]float64
	moods   map[string]float64
	decades map[string]float64
}

const (
	recommendationArtistAffinityWeight  = 0.48
	recommendationGenreAffinityWeight   = 0.22
	recommendationMoodAffinityWeight    = 0.08
	recommendationDecadeAffinityWeight  = 0.12
	recommendationBaseScoreWeight       = 0.60
	recommendationProfileBonusWeight    = 0.75
	recommendationCalibrationBonusDelta = 0.08
)

func (n *Router) buildRecommendationTasteProfile(ctx context.Context, seeds []subsonic.RecommendationSeed) (recommendationTasteProfile, error) {
	seedIDs := make([]string, 0, len(seeds))
	for _, seed := range seeds {
		id := strings.TrimSpace(seed.TrackID)
		if id == "" || strings.HasPrefix(id, "_") {
			continue
		}
		seedIDs = append(seedIDs, id)
	}
	seedIDs = uniqueNonEmptyStrings(seedIDs)
	if len(seedIDs) == 0 {
		return recommendationTasteProfile{}, nil
	}

	tracks, err := n.loadTracks(ctx, seedIDs)
	if err != nil {
		return recommendationTasteProfile{}, err
	}
	return buildRecommendationTasteProfile(tracks), nil
}

func buildRecommendationTasteProfile(tracks []model.MediaFile) recommendationTasteProfile {
	artistCounts := make(map[string]float64)
	genreCounts := make(map[string]float64)
	moodCounts := make(map[string]float64)
	decadeCounts := make(map[string]float64)

	for _, track := range tracks {
		accumulateFeatureCounts(artistCounts, recommendationArtistValues(track))
		accumulateFeatureCounts(genreCounts, recommendationGenreValues(track))
		accumulateFeatureCounts(moodCounts, recommendationMoodValues(track))
		accumulateFeatureCounts(decadeCounts, recommendationDecadeValues(track))
	}

	return recommendationTasteProfile{
		artists: normalizeFeatureCounts(artistCounts),
		genres:  normalizeFeatureCounts(genreCounts),
		moods:   normalizeFeatureCounts(moodCounts),
		decades: normalizeFeatureCounts(decadeCounts),
	}
}

func recommendationScoreMap(items []subsonic.RecommendationItem) map[string]float64 {
	scores := make(map[string]float64, len(items))
	for _, item := range items {
		scores[item.TrackID] = item.Score
	}
	return scores
}

func rerankRecommendationTracks(candidates []recommendationTrack, baseScores map[string]float64, profile recommendationTasteProfile) []recommendationTrack {
	if len(candidates) <= 1 {
		return candidates
	}

	normalizedBase := normalizeRecommendationScores(candidates, baseScores)
	selected := make([]recommendationTrack, 0, len(candidates))
	used := make([]bool, len(candidates))
	albumCounts := make(map[string]int)
	pathSet := make(map[string]struct{})

	for len(selected) < len(candidates) {
		currentDivergence := profile.divergence(selected)
		bestIdx := -1
		bestScore := math.Inf(-1)
		bestBase := math.Inf(-1)

		for idx, candidate := range candidates {
			if used[idx] {
				continue
			}
			if pathKey(candidate.MediaFile) != "" {
				if _, dup := pathSet[pathKey(candidate.MediaFile)]; dup {
					continue
				}
			}

			score := normalizedBase[candidate.ID]
			if profile.hasSignal() {
				score = normalizedBase[candidate.ID]*recommendationBaseScoreWeight +
					profile.affinity(candidate.MediaFile)*recommendationProfileBonusWeight
				nextSelection := append(append([]recommendationTrack(nil), selected...), candidate)
				nextDivergence := profile.divergence(nextSelection)
				if nextDivergence < currentDivergence {
					score += recommendationCalibrationBonusDelta
				} else if nextDivergence > currentDivergence {
					score -= recommendationCalibrationBonusDelta
				}
			}
			if albumRepeatPenalty(candidate.MediaFile, albumCounts) > 0 {
				score -= 0.15
			}

			baseScore := normalizedBase[candidate.ID]
			if score > bestScore || (score == bestScore && baseScore > bestBase) {
				bestIdx = idx
				bestScore = score
				bestBase = baseScore
			}
		}

		if bestIdx < 0 {
			break
		}

		chosen := candidates[bestIdx]
		selected = append(selected, chosen)
		used[bestIdx] = true
		if key := albumKey(chosen.MediaFile); key != "" {
			albumCounts[key]++
		}
		if key := pathKey(chosen.MediaFile); key != "" {
			pathSet[key] = struct{}{}
		}
	}

	return selected
}

func selectRecommendationTracks(candidates []recommendationTrack, limit int, blocked map[string]struct{}, dislikes dislikeSignals, settings recommendationSettings) ([]recommendationTrack, string) {
	if limit <= 0 || len(candidates) == 0 {
		return []recommendationTrack{}, ""
	}

	capacity := limit
	if len(candidates) < capacity {
		capacity = len(candidates)
	}
	selected := make([]recommendationTrack, 0, capacity)
	seenIDs := make(map[string]struct{}, limit)
	seenPaths := make(map[string]struct{}, limit)
	blockedRemoved := 0
	dislikedRemoved := 0

	for _, candidate := range candidates {
		if len(selected) >= limit {
			break
		}
		if _, ok := seenIDs[candidate.ID]; ok {
			continue
		}
		if _, skip := blocked[candidate.ID]; skip {
			blockedRemoved++
			continue
		}
		if dislikes.shouldReject(candidate.MediaFile, candidate.ID, settings.LowRatingPenalty) {
			dislikedRemoved++
			continue
		}
		if !settings.isDurationAllowed(candidate.Duration) {
			continue
		}
		if key := pathKey(candidate.MediaFile); key != "" {
			if _, dup := seenPaths[key]; dup {
				continue
			}
			seenPaths[key] = struct{}{}
		}
		seenIDs[candidate.ID] = struct{}{}
		selected = append(selected, candidate)
	}

	return selected, recommendationFilterWarning(blockedRemoved, dislikedRemoved)
}

func recommendationTrackIDs(tracks []recommendationTrack) []string {
	ids := make([]string, 0, len(tracks))
	for _, track := range tracks {
		if strings.TrimSpace(track.ID) == "" {
			continue
		}
		ids = append(ids, track.ID)
	}
	return ids
}

func normalizeRecommendationScores(candidates []recommendationTrack, baseScores map[string]float64) map[string]float64 {
	normalized := make(map[string]float64, len(candidates))
	if len(candidates) == 0 {
		return normalized
	}

	minScore := 0.0
	maxScore := 0.0
	first := true
	for _, candidate := range candidates {
		score := baseScores[candidate.ID]
		if first {
			minScore = score
			maxScore = score
			first = false
			continue
		}
		if score < minScore {
			minScore = score
		}
		if score > maxScore {
			maxScore = score
		}
	}

	if maxScore == minScore {
		for _, candidate := range candidates {
			normalized[candidate.ID] = 1
		}
		return normalized
	}

	rangeScore := maxScore - minScore
	for _, candidate := range candidates {
		normalized[candidate.ID] = (baseScores[candidate.ID] - minScore) / rangeScore
	}
	return normalized
}

func recommendationFilterWarning(blockedRemoved int, dislikedRemoved int) string {
	if blockedRemoved == 0 && dislikedRemoved == 0 {
		return ""
	}
	parts := make([]string, 0, 2)
	if blockedRemoved > 0 {
		parts = append(parts, fmt.Sprintf("%d tracks removed because they belong to excluded playlists.", blockedRemoved))
	}
	if dislikedRemoved > 0 {
		parts = append(parts, fmt.Sprintf("%d tracks skipped because you rated them poorly.", dislikedRemoved))
	}
	return strings.Join(parts, " ")
}

func (p recommendationTasteProfile) hasSignal() bool {
	return len(p.artists) > 0 || len(p.genres) > 0 || len(p.moods) > 0 || len(p.decades) > 0
}

func (p recommendationTasteProfile) affinity(track model.MediaFile) float64 {
	if !p.hasSignal() {
		return 0
	}

	total := 0.0
	weight := 0.0

	if len(p.artists) > 0 {
		total += featureAffinityScore(recommendationArtistValues(track), p.artists) * recommendationArtistAffinityWeight
		weight += recommendationArtistAffinityWeight
	}
	if len(p.genres) > 0 {
		total += featureAffinityScore(recommendationGenreValues(track), p.genres) * recommendationGenreAffinityWeight
		weight += recommendationGenreAffinityWeight
	}
	if len(p.moods) > 0 {
		total += featureAffinityScore(recommendationMoodValues(track), p.moods) * recommendationMoodAffinityWeight
		weight += recommendationMoodAffinityWeight
	}
	if len(p.decades) > 0 {
		total += featureAffinityScore(recommendationDecadeValues(track), p.decades) * recommendationDecadeAffinityWeight
		weight += recommendationDecadeAffinityWeight
	}
	if weight == 0 {
		return 0
	}
	return total / weight
}

func (p recommendationTasteProfile) divergence(tracks []recommendationTrack) float64 {
	if !p.hasSignal() {
		return 0
	}

	artistCounts := make(map[string]float64)
	genreCounts := make(map[string]float64)
	moodCounts := make(map[string]float64)
	decadeCounts := make(map[string]float64)

	for _, track := range tracks {
		accumulateFeatureCounts(artistCounts, recommendationArtistValues(track.MediaFile))
		accumulateFeatureCounts(genreCounts, recommendationGenreValues(track.MediaFile))
		accumulateFeatureCounts(moodCounts, recommendationMoodValues(track.MediaFile))
		accumulateFeatureCounts(decadeCounts, recommendationDecadeValues(track.MediaFile))
	}

	total := 0.0
	count := 0.0
	if len(p.artists) > 0 {
		total += jensenShannonDivergence(p.artists, normalizeFeatureCounts(artistCounts))
		count++
	}
	if len(p.genres) > 0 {
		total += jensenShannonDivergence(p.genres, normalizeFeatureCounts(genreCounts))
		count++
	}
	if len(p.moods) > 0 {
		total += jensenShannonDivergence(p.moods, normalizeFeatureCounts(moodCounts))
		count++
	}
	if len(p.decades) > 0 {
		total += jensenShannonDivergence(p.decades, normalizeFeatureCounts(decadeCounts))
		count++
	}
	if count == 0 {
		return 0
	}
	return total / count
}

func accumulateFeatureCounts(target map[string]float64, values []string) {
	for _, value := range values {
		if value == "" {
			continue
		}
		target[value]++
	}
}

func normalizeFeatureCounts(counts map[string]float64) map[string]float64 {
	if len(counts) == 0 {
		return nil
	}
	total := 0.0
	for _, value := range counts {
		total += value
	}
	if total == 0 {
		return nil
	}
	normalized := make(map[string]float64, len(counts))
	for key, value := range counts {
		normalized[key] = value / total
	}
	return normalized
}

func featureAffinityScore(values []string, weights map[string]float64) float64 {
	if len(values) == 0 || len(weights) == 0 {
		return 0
	}
	best := 0.0
	for _, value := range values {
		if value == "" {
			continue
		}
		if score := weights[value]; score > best {
			best = score
		}
	}
	return clamp(best, 0, 1)
}

func recommendationArtistValues(mf model.MediaFile) []string {
	artist := mf.Participants.First(model.RoleArtist)
	switch {
	case strings.TrimSpace(artist.ID) != "":
		return []string{"artist:" + strings.ToLower(strings.TrimSpace(artist.ID))}
	case strings.TrimSpace(mf.ArtistID) != "":
		return []string{"artist:" + strings.ToLower(strings.TrimSpace(mf.ArtistID))}
	case strings.TrimSpace(mf.Artist) != "":
		return []string{"artist:" + normalizeFeatureValue(mf.Artist)}
	default:
		return nil
	}
}

func recommendationGenreValues(mf model.MediaFile) []string {
	genres := mf.Tags.Values(model.TagGenre)
	if len(genres) == 0 && strings.TrimSpace(mf.Genre) != "" {
		genres = []string{mf.Genre}
	}
	return normalizeFeatureValues("genre:", genres)
}

func recommendationMoodValues(mf model.MediaFile) []string {
	return normalizeFeatureValues("mood:", mf.Tags.Values(model.TagMood))
}

func recommendationDecadeValues(mf model.MediaFile) []string {
	year := mf.Year
	if year <= 0 {
		year = mf.ReleaseYear
	}
	if year <= 0 {
		year = mf.OriginalYear
	}
	if year <= 0 {
		return nil
	}
	return []string{fmt.Sprintf("decade:%d", (year/10)*10)}
}

func normalizeFeatureValues(prefix string, values []string) []string {
	normalized := make([]string, 0, len(values))
	seen := make(map[string]struct{}, len(values))
	for _, value := range values {
		key := prefix + normalizeFeatureValue(value)
		if key == prefix {
			continue
		}
		if _, dup := seen[key]; dup {
			continue
		}
		seen[key] = struct{}{}
		normalized = append(normalized, key)
	}
	return normalized
}

func normalizeFeatureValue(value string) string {
	return strings.ToLower(strings.TrimSpace(value))
}

func albumKey(mf model.MediaFile) string {
	if strings.TrimSpace(mf.AlbumID) != "" {
		return "album:" + strings.ToLower(strings.TrimSpace(mf.AlbumID))
	}
	if strings.TrimSpace(mf.Album) == "" {
		return ""
	}
	return "album:" + normalizeFeatureValue(mf.Album)
}

func albumRepeatPenalty(mf model.MediaFile, albumCounts map[string]int) float64 {
	key := albumKey(mf)
	if key == "" || albumCounts[key] == 0 {
		return 0
	}
	return 0.15
}

func pathKey(mf model.MediaFile) string {
	return strings.ToLower(strings.TrimSpace(mf.Path))
}

func jensenShannonDivergence(left map[string]float64, right map[string]float64) float64 {
	if len(left) == 0 && len(right) == 0 {
		return 0
	}
	merged := make(map[string]struct{}, len(left)+len(right))
	for key := range left {
		merged[key] = struct{}{}
	}
	for key := range right {
		merged[key] = struct{}{}
	}

	kl := func(a map[string]float64, midpoint map[string]float64) float64 {
		total := 0.0
		for key, value := range a {
			if value <= 0 {
				continue
			}
			m := midpoint[key]
			if m <= 0 {
				continue
			}
			total += value * math.Log(value/m)
		}
		return total
	}

	midpoint := make(map[string]float64, len(merged))
	for key := range merged {
		midpoint[key] = 0.5 * (left[key] + right[key])
	}
	return 0.5 * (kl(left, midpoint) + kl(right, midpoint)) / math.Log(2)
}
