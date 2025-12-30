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
	// Check cache first
	if cached, ok := r.cache.Load(name); ok {
		return cached.(string), nil
	}

	// Parse canonical name (format: "Artist - Title")
	artist, title := parseCanonicalName(name)

	// Query database
	repo := r.ds.MediaFile(ctx)

	// Try exact match first
	if artist != "" && title != "" {
		tracks, err := repo.GetAll(model.QueryOptions{
			Filters: squirrel.And{
				squirrel.Eq{"artist": artist},
				squirrel.Eq{"title": title},
			},
			Max: 1,
		})
		if err != nil {
			log.Warn(ctx, "Query tracks failed", "error", err)
		} else if len(tracks) > 0 {
			r.cache.Store(name, tracks[0].ID)
			return tracks[0].ID, nil
		}
	}

	// Try with just title if no artist
	if title != "" && artist == "" {
		tracks, err := repo.GetAll(model.QueryOptions{
			Filters: squirrel.Eq{"title": title},
			Max:     1,
		})
		if err != nil {
			log.Warn(ctx, "Query tracks failed", "error", err)
		} else if len(tracks) > 0 {
			r.cache.Store(name, tracks[0].ID)
			return tracks[0].ID, nil
		}
	}

	// Try with just the name as ID (some embeddings use track ID directly)
	exists, err := repo.Exists(name)
	if err == nil && exists {
		r.cache.Store(name, name)
		return name, nil
	}

	log.Debug(ctx, "Could not resolve track name", "name", name, "artist", artist, "title", title)
	return "", nil
}

// ResolveTrackIDs resolves multiple canonical names to track IDs.
func (r *Resolver) ResolveTrackIDs(ctx context.Context, names []string) (map[string]string, error) {
	result := make(map[string]string, len(names))

	// First, check cache
	var uncached []string
	for _, name := range names {
		if cached, ok := r.cache.Load(name); ok {
			result[name] = cached.(string)
		} else {
			uncached = append(uncached, name)
		}
	}

	if len(uncached) == 0 {
		return result, nil
	}

	// Resolve individually for simplicity (batch would be more complex with squirrel)
	repo := r.ds.MediaFile(ctx)
	for _, name := range uncached {
		artist, title := parseCanonicalName(name)

		var tracks model.MediaFiles
		var err error

		if artist != "" && title != "" {
			tracks, err = repo.GetAll(model.QueryOptions{
				Filters: squirrel.And{
					squirrel.Eq{"artist": artist},
					squirrel.Eq{"title": title},
				},
				Max: 1,
			})
		} else if title != "" {
			tracks, err = repo.GetAll(model.QueryOptions{
				Filters: squirrel.Eq{"title": title},
				Max:     1,
			})
		}

		if err != nil {
			log.Debug(ctx, "Query tracks failed", "name", name, "error", err)
			continue
		}

		if len(tracks) > 0 {
			canonicalName := CanonicalName(tracks[0].Artist, tracks[0].Title)
			r.cache.Store(canonicalName, tracks[0].ID)
			result[name] = tracks[0].ID
			continue
		}

		// Try name as track ID directly
		exists, err := repo.Exists(name)
		if err == nil && exists {
			r.cache.Store(name, name)
			result[name] = name
		}
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
