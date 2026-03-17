package main

import (
	"context"
	"database/sql"
	"fmt"
	"strings"

	_ "github.com/lib/pq"           // PostgreSQL driver
	_ "github.com/mattn/go-sqlite3" // SQLite driver
)

// Track represents a music track from the database.
type Track struct {
	ID          string
	Path        string
	Title       string
	Artist      string
	Album       string
	AlbumArtist string
	Lyrics      string
}

// OpenDatabase opens a connection to the Navidrome database.
func OpenDatabase(ctx context.Context, cfg DatabaseConfig) (*sql.DB, error) {
	var db *sql.DB
	var err error

	switch cfg.Type {
	case "postgres":
		db, err = sql.Open("postgres", cfg.Path)
	case "sqlite3":
		// Use same configuration as Navidrome for SQLite
		db, err = sql.Open("sqlite3", cfg.Path)
	default:
		return nil, fmt.Errorf("unsupported database type: %s", cfg.Type)
	}

	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	// Configure connection pool (similar to Navidrome's settings)
	db.SetMaxOpenConns(4)
	db.SetMaxIdleConns(2)

	// Test connection
	if err := db.PingContext(ctx); err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}

	return db, nil
}

// GetTotalTrackCount returns the total number of tracks in the database.
func GetTotalTrackCount(ctx context.Context, db *sql.DB, filter string) (int, error) {
	query := "SELECT COUNT(*) FROM media_file WHERE path IS NOT NULL AND path != ''"

	// Add optional filter
	if filter != "" {
		query += " AND (" + filter + ")"
	}

	var count int
	err := db.QueryRowContext(ctx, query).Scan(&count)
	if err != nil {
		return 0, fmt.Errorf("failed to get track count: %w", err)
	}

	return count, nil
}

// FetchTrackBatch fetches a batch of tracks from the database.
// offset: starting position (0-indexed)
// limit: number of tracks to fetch
// filter: optional SQL WHERE clause to filter tracks
func FetchTrackBatch(ctx context.Context, db *sql.DB, offset, limit int, filter string) ([]Track, error) {
	query := `
		SELECT
			id,
			path,
			COALESCE(title, '') as title,
			COALESCE(artist, '') as artist,
			COALESCE(album, '') as album,
			COALESCE(album_artist, '') as album_artist,
			COALESCE(lyrics, '') as lyrics
		FROM media_file
		WHERE path IS NOT NULL AND path != ''
	`

	// Add optional filter
	if filter != "" {
		query += " AND (" + filter + ")"
	}

	query += " ORDER BY id LIMIT ? OFFSET ?"

	rows, err := db.QueryContext(ctx, query, limit, offset)
	if err != nil {
		return nil, fmt.Errorf("failed to query tracks: %w", err)
	}
	defer rows.Close()

	var tracks []Track
	for rows.Next() {
		var track Track
		err := rows.Scan(
			&track.ID,
			&track.Path,
			&track.Title,
			&track.Artist,
			&track.Album,
			&track.AlbumArtist,
			&track.Lyrics,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan track row: %w", err)
		}
		tracks = append(tracks, track)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating track rows: %w", err)
	}

	return tracks, nil
}

// GetTrackByID fetches a single track by its ID.
func GetTrackByID(ctx context.Context, db *sql.DB, id string) (*Track, error) {
	query := `
		SELECT
			id,
			path,
			COALESCE(title, '') as title,
			COALESCE(artist, '') as artist,
			COALESCE(album, '') as album,
			COALESCE(album_artist, '') as album_artist,
			COALESCE(lyrics, '') as lyrics
		FROM media_file
		WHERE id = ?
	`

	var track Track
	err := db.QueryRowContext(ctx, query, id).Scan(
		&track.ID,
		&track.Path,
		&track.Title,
		&track.Artist,
		&track.Album,
		&track.AlbumArtist,
		&track.Lyrics,
	)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("track not found: %s", id)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get track: %w", err)
	}

	return &track, nil
}

// CanonicalName generates a canonical name for a track (for Milvus).
// Format: "Artist - Title" (or fallback to filename if metadata is missing)
func CanonicalName(track Track) string {
	artist := strings.TrimSpace(track.Artist)
	title := strings.TrimSpace(track.Title)

	if artist != "" && title != "" {
		return fmt.Sprintf("%s - %s", artist, title)
	}

	if title != "" {
		return title
	}

	// Fallback to filename if no metadata
	return track.Path
}
