package resolver

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCanonicalName(t *testing.T) {
	tests := []struct {
		name     string
		artist   string
		title    string
		expected string
	}{
		{"artist and title", "The Beatles", "Hey Jude", "The Beatles - Hey Jude"},
		{"only artist", "The Beatles", "", "The Beatles - "},
		{"only title", "", "Hey Jude", "Hey Jude"},
		{"neither", "", "", ""},
		{"long artist and title", "The Red Hot Chili Peppers", "Under the Bridge", "The Red Hot Chili Peppers - Under the Bridge"},
		{"unicode characters", "日本語アーティスト", "日本語タイトル", "日本語アーティスト - 日本語タイトル"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := CanonicalName(tt.artist, tt.title)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestParseCanonicalName(t *testing.T) {
	tests := []struct {
		name           string
		input          string
		expectedArtist string
		expectedTitle  string
	}{
		{"artist and title", "The Beatles - Hey Jude", "The Beatles", "Hey Jude"},
		{"only title", "Hey Jude", "", "Hey Jude"},
		{"multiple dashes", "AC/DC - Back In Black", "AC/DC", "Back In Black"},
		{"dash in artist name", "Artist - With - Dash - Title", "Artist", "With - Dash - Title"},
		{"empty string", "", "", ""},
		{"only whitespace", "   ", "", ""},
		{"title with leading/trailing spaces", "  Artist  -  Title  ", "Artist", "Title"},
		{"no spaces around dash", "Artist-Title", "", "Artist-Title"}, // Requires " - " with spaces
		{"unicode", "Künstler - Titel", "Künstler", "Titel"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			artist, title := parseCanonicalName(tt.input)
			assert.Equal(t, tt.expectedArtist, artist)
			assert.Equal(t, tt.expectedTitle, title)
		})
	}
}

func TestNewResolver(t *testing.T) {
	r := NewResolver(nil)

	assert.NotNil(t, r)
	assert.Nil(t, r.ds)
}

func TestResolverClearCache(t *testing.T) {
	r := NewResolver(nil)

	// Add something to cache
	r.cache.Store("test", "value")

	// Verify it's there
	_, exists := r.cache.Load("test")
	assert.True(t, exists)

	// Clear cache
	r.ClearCache()

	// Verify it's gone
	_, exists = r.cache.Load("test")
	assert.False(t, exists)
}

func TestResolverCacheHit(t *testing.T) {
	r := NewResolver(nil)

	// Pre-populate cache
	r.cache.Store("Artist - Title", "track-id-123")

	// Verify cache hit returns value
	value, ok := r.cache.Load("Artist - Title")
	assert.True(t, ok)
	assert.Equal(t, "track-id-123", value)
}

func TestResolverCacheMiss(t *testing.T) {
	r := NewResolver(nil)

	// Empty cache should miss
	value, ok := r.cache.Load("Unknown Artist - Unknown Title")
	assert.False(t, ok)
	assert.Nil(t, value)
}

func TestResolverCacheConcurrency(t *testing.T) {
	r := NewResolver(nil)

	// sync.Map is safe for concurrent use
	done := make(chan bool, 100)
	for i := 0; i < 100; i++ {
		go func(id int) {
			key := string(rune('A' + id%26))
			r.cache.Store(key, id)
			_, _ = r.cache.Load(key)
			done <- true
		}(i)
	}

	for i := 0; i < 100; i++ {
		<-done
	}
}

func TestParseCanonicalNameEdgeCases(t *testing.T) {
	// Edge case: exactly " - " is preserved
	artist, title := parseCanonicalName(" - ")
	assert.Equal(t, "", artist)
	assert.Equal(t, "", title)

	// Edge case: just spaces around dash
	artist, title = parseCanonicalName("  -  ")
	assert.Equal(t, "", artist)
	assert.Equal(t, "", title)

	// Edge case: dash without spaces isn't split
	artist, title = parseCanonicalName("Artist-Title")
	assert.Equal(t, "", artist)
	assert.Equal(t, "Artist-Title", title)
}

func TestCanonicalNameRoundTrip(t *testing.T) {
	// Test that CanonicalName followed by parseCanonicalName is consistent
	testCases := []struct {
		artist string
		title  string
	}{
		{"The Beatles", "Hey Jude"},
		{"AC/DC", "Back In Black"},
		{"Single", "Word"},
		{"Artist With Many Words", "Title With Many Words"},
	}

	for _, tc := range testCases {
		canonical := CanonicalName(tc.artist, tc.title)
		parsedArtist, parsedTitle := parseCanonicalName(canonical)
		assert.Equal(t, tc.artist, parsedArtist)
		assert.Equal(t, tc.title, parsedTitle)
	}
}
