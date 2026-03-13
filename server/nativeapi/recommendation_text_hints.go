package nativeapi

import (
	"context"
	"strings"
	"unicode"

	"github.com/navidrome/navidrome/model"
)

const inferredTextPositiveSeedLimit = 4

func (n *Router) inferTextPromptPositiveTrackIDs(ctx context.Context, user model.User, prompt string, settings recommendationSettings, existingIDs []string) ([]string, error) {
	if strings.TrimSpace(prompt) == "" {
		return nil, nil
	}

	candidateSeeds, err := n.buildAllSeeds(ctx, user, 24, settings)
	if err != nil {
		return nil, err
	}
	if len(candidateSeeds) == 0 {
		return nil, nil
	}

	seedIDs := make([]string, 0, len(candidateSeeds))
	for _, seed := range candidateSeeds {
		id := strings.TrimSpace(seed.TrackID)
		if id == "" || strings.HasPrefix(id, "_") {
			continue
		}
		seedIDs = append(seedIDs, id)
	}
	seedIDs = uniqueNonEmptyStrings(seedIDs)
	if len(seedIDs) == 0 {
		return nil, nil
	}

	tracks, err := n.loadTracks(ctx, seedIDs)
	if err != nil {
		return nil, err
	}
	if len(tracks) == 0 {
		return nil, nil
	}

	normalizedPrompt := normalizePromptText(prompt)
	if normalizedPrompt == "" {
		return nil, nil
	}
	existing := make(map[string]struct{}, len(existingIDs))
	for _, id := range existingIDs {
		id = strings.TrimSpace(id)
		if id != "" {
			existing[id] = struct{}{}
		}
	}

	matched := make([]string, 0, inferredTextPositiveSeedLimit)
	for _, track := range tracks {
		if track.ID == "" {
			continue
		}
		if _, skip := existing[track.ID]; skip {
			continue
		}
		if !promptMentionsTrackArtist(normalizedPrompt, track) {
			continue
		}
		matched = append(matched, track.ID)
		existing[track.ID] = struct{}{}
		if len(matched) >= inferredTextPositiveSeedLimit {
			break
		}
	}
	if len(matched) == 0 {
		return nil, nil
	}
	return matched, nil
}

func promptMentionsTrackArtist(normalizedPrompt string, track model.MediaFile) bool {
	for _, name := range recommendationTrackArtistNames(track) {
		if promptContainsPhrase(normalizedPrompt, name) {
			return true
		}
	}
	return false
}

func recommendationTrackArtistNames(track model.MediaFile) []string {
	values := make([]string, 0, 4)
	values = append(values, track.Artist, track.AlbumArtist)
	for _, participant := range track.Participants[model.RoleArtist] {
		values = append(values, participant.Name)
	}
	for _, participant := range track.Participants[model.RoleAlbumArtist] {
		values = append(values, participant.Name)
	}
	return uniqueNormalizedArtistNames(values)
}

func uniqueNormalizedArtistNames(values []string) []string {
	if len(values) == 0 {
		return nil
	}
	result := make([]string, 0, len(values))
	seen := make(map[string]struct{}, len(values))
	for _, value := range values {
		normalized := normalizePromptText(value)
		if normalized == "" {
			continue
		}
		if _, ok := seen[normalized]; ok {
			continue
		}
		seen[normalized] = struct{}{}
		result = append(result, normalized)
	}
	return result
}

func promptContainsPhrase(normalizedPrompt string, phrase string) bool {
	if phrase == "" || normalizedPrompt == "" {
		return false
	}
	return strings.Contains(" "+normalizedPrompt+" ", " "+phrase+" ")
}

func normalizePromptText(value string) string {
	if strings.TrimSpace(value) == "" {
		return ""
	}
	var b strings.Builder
	b.Grow(len(value))
	prevSpace := true
	for _, r := range strings.ToLower(value) {
		switch {
		case unicode.IsLetter(r) || unicode.IsDigit(r):
			b.WriteRune(r)
			prevSpace = false
		default:
			if !prevSpace {
				b.WriteByte(' ')
				prevSpace = true
			}
		}
	}
	return strings.TrimSpace(b.String())
}
