// Package resolver provides track name to ID resolution for the recommender.
package resolver

import (
	"context"
	"strings"
	"sync"

	"github.com/Masterminds/squirrel"
	"github.com/navidrome/navidrome/log"
	"github.com/navidrome/navidrome/model"
)

// Resolver maps canonical track names to track IDs using the database.
type Resolver struct {
	ds    model.DataStore
	cache sync.Map // Simple in-memory cache: name -> trackID
}

// NewResolver creates a new Resolver with the given data store.
func NewResolver(ds model.DataStore) *Resolver {
	return &Resolver{
		ds: ds,
	}
}

// CanonicalName creates a canonical name from artist and title.
func CanonicalName(artist, title string) string {
	if artist == "" {
		return title
	}
	return artist + " - " + title
}

// ResolveTrackID resolves a canonical name to a track ID.
func (r *Resolver) ResolveTrackID(ctx context.Context, name string) (string, error) {
	resolved, err := r.ResolveTrackIDs(ctx, []string{name})
	if err != nil {
		return "", err
	}
	return resolved[name], nil
}

// ResolveTrackIDs resolves multiple canonical names to track IDs.
func (r *Resolver) ResolveTrackIDs(ctx context.Context, names []string) (map[string]string, error) {
	result := make(map[string]string, len(names))

	var uncached []string
	for _, name := range names {
		trimmed := strings.TrimSpace(name)
		if trimmed == "" {
			continue
		}
		if cached, ok := r.cache.Load(trimmed); ok {
			result[name] = cached.(string)
		} else {
			uncached = append(uncached, trimmed)
		}
	}

	if len(uncached) == 0 {
		return result, nil
	}

	repo := r.ds.MediaFile(ctx)
	remaining := uniqueNames(uncached)

	if len(remaining) > 0 {
		idMatches, err := repo.GetAll(model.QueryOptions{
			Filters: squirrel.Eq{"media_file.id": remaining},
			Max:     len(remaining),
		})
		if err != nil {
			log.Debug(ctx, "Batch ID lookup failed", "error", err)
		} else {
			for _, track := range idMatches {
				r.storeResolvedKeys(track.ID, track)
				result[track.ID] = track.ID
			}
			remaining = subtractResolvedNames(remaining, result)
		}
	}

	type canonicalLookup struct {
		artist string
		title  string
	}
	canonicalKeys := make(map[string]canonicalLookup)
	titleNames := make([]string, 0)
	pathNames := make([]string, 0)
	for _, name := range remaining {
		artist, title := parseCanonicalName(name)
		switch {
		case artist != "" && title != "":
			canonicalKeys[name] = canonicalLookup{artist: artist, title: title}
		case strings.Contains(name, "/"):
			pathNames = append(pathNames, name)
		case title != "":
			titleNames = append(titleNames, title)
		}
	}

	if len(canonicalKeys) > 0 {
		orFilters := make(squirrel.Or, 0, len(canonicalKeys))
		for _, lookup := range canonicalKeys {
			orFilters = append(orFilters, squirrel.And{
				squirrel.Eq{"artist": lookup.artist},
				squirrel.Eq{"title": lookup.title},
			})
		}
		tracks, err := repo.GetAll(model.QueryOptions{
			Filters: orFilters,
			Max:     len(canonicalKeys),
		})
		if err != nil {
			log.Debug(ctx, "Batch canonical lookup failed", "error", err)
		} else {
			byCanonical := make(map[string]model.MediaFile, len(tracks))
			for _, track := range tracks {
				byCanonical[CanonicalName(track.Artist, track.Title)] = track
			}
			for requested, lookup := range canonicalKeys {
				if track, ok := byCanonical[CanonicalName(lookup.artist, lookup.title)]; ok {
					r.storeResolvedKeys(requested, track)
					result[requested] = track.ID
				}
			}
		}
	}

	if len(titleNames) > 0 {
		tracks, err := repo.GetAll(model.QueryOptions{
			Filters: squirrel.Eq{"title": uniqueNames(titleNames)},
			Max:     len(titleNames),
		})
		if err != nil {
			log.Debug(ctx, "Batch title lookup failed", "error", err)
		} else {
			byTitle := make(map[string]model.MediaFile, len(tracks))
			for _, track := range tracks {
				if _, exists := byTitle[track.Title]; !exists {
					byTitle[track.Title] = track
				}
			}
			for _, title := range titleNames {
				if track, ok := byTitle[title]; ok {
					r.storeResolvedKeys(title, track)
					result[title] = track.ID
				}
			}
		}
	}

	if len(pathNames) > 0 {
		tracks, err := repo.GetAll(model.QueryOptions{
			Filters: squirrel.Eq{"path": uniqueNames(pathNames)},
			Max:     len(pathNames),
		})
		if err != nil {
			log.Debug(ctx, "Batch path lookup failed", "error", err)
		} else {
			byPath := make(map[string]model.MediaFile, len(tracks))
			for _, track := range tracks {
				byPath[track.Path] = track
			}
			for _, path := range pathNames {
				if track, ok := byPath[path]; ok {
					r.storeResolvedKeys(path, track)
					result[path] = track.ID
				}
			}
		}
	}

	for _, name := range remaining {
		if _, ok := result[name]; ok {
			continue
		}
		log.Debug(ctx, "Could not resolve track name", "name", name)
	}

	return result, nil
}

// ClearCache clears the resolver cache.
func (r *Resolver) ClearCache() {
	r.cache = sync.Map{}
}

// parseCanonicalName parses a canonical name into artist and title.
// Format: "Artist - Title" or just "Title"
func parseCanonicalName(name string) (artist, title string) {
	parts := strings.SplitN(name, " - ", 2)
	if len(parts) == 2 {
		return strings.TrimSpace(parts[0]), strings.TrimSpace(parts[1])
	}
	return "", strings.TrimSpace(name)
}

func (r *Resolver) storeResolvedKeys(requested string, track model.MediaFile) {
	if strings.TrimSpace(requested) != "" {
		r.cache.Store(requested, track.ID)
	}
	if canonical := CanonicalName(track.Artist, track.Title); canonical != "" {
		r.cache.Store(canonical, track.ID)
	}
	if title := strings.TrimSpace(track.Title); title != "" {
		r.cache.Store(title, track.ID)
	}
	if path := strings.TrimSpace(track.Path); path != "" {
		r.cache.Store(path, track.ID)
	}
}

func subtractResolvedNames(names []string, resolved map[string]string) []string {
	filtered := make([]string, 0, len(names))
	for _, name := range names {
		if _, ok := resolved[name]; ok {
			continue
		}
		filtered = append(filtered, name)
	}
	return filtered
}

func uniqueNames(names []string) []string {
	seen := make(map[string]struct{}, len(names))
	result := make([]string, 0, len(names))
	for _, name := range names {
		trimmed := strings.TrimSpace(name)
		if trimmed == "" {
			continue
		}
		if _, ok := seen[trimmed]; ok {
			continue
		}
		seen[trimmed] = struct{}{}
		result = append(result, trimmed)
	}
	return result
}
