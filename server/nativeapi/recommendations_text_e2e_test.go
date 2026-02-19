package nativeapi

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/navidrome/navidrome/conf"
	"github.com/navidrome/navidrome/conf/configtest"
	"github.com/navidrome/navidrome/consts"
	"github.com/navidrome/navidrome/core"
	"github.com/navidrome/navidrome/core/auth"
	"github.com/navidrome/navidrome/model"
	"github.com/navidrome/navidrome/server"
	"github.com/navidrome/navidrome/server/subsonic"
	"github.com/navidrome/navidrome/tests"
)

type captureRecommendationClient struct {
	response  *subsonic.RecommendationResponse
	err       error
	callCount int
	lastMode  string
	lastReq   subsonic.RecommendationRequest
}

func (c *captureRecommendationClient) Recommend(_ context.Context, mode string, req subsonic.RecommendationRequest) (*subsonic.RecommendationResponse, error) {
	c.callCount++
	c.lastMode = mode
	c.lastReq = req
	if c.err != nil {
		return nil, c.err
	}
	if c.response != nil {
		return c.response, nil
	}
	return &subsonic.RecommendationResponse{}, nil
}

type staticPlaylistTrackRepo struct {
	model.PlaylistTrackRepository
	tracks model.PlaylistTracks
	err    error
}

func (r *staticPlaylistTrackRepo) GetAll(...model.QueryOptions) (model.PlaylistTracks, error) {
	if r.err != nil {
		return nil, r.err
	}
	return r.tracks, nil
}

type textRecommendationHarness struct {
	ds     *tests.MockDataStore
	router http.Handler
	user   model.User
	rec    *captureRecommendationClient
}

func newTextRecommendationHarness(t *testing.T, rec *captureRecommendationClient) textRecommendationHarness {
	t.Helper()

	restore := configtest.SetupConfig()
	t.Cleanup(restore)

	ds := &tests.MockDataStore{}
	auth.Init(ds)

	// Keep text model dimensions aligned in tests, matching how text query seeds are fanned out
	// into lyrics and description targets.
	conf.Server.Recommendations.Milvus.Dimensions.Lyrics = 16
	conf.Server.Recommendations.Milvus.Dimensions.Description = 16

	embeddingServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/embeddings" {
			http.NotFound(w, r)
			return
		}
		var reqBody struct {
			Input string `json:"input"`
			Model string `json:"model"`
		}
		if err := json.NewDecoder(r.Body).Decode(&reqBody); err != nil {
			http.Error(w, "invalid json", http.StatusBadRequest)
			return
		}
		vector := mockEmbeddingServiceVector(reqBody.Input, reqBody.Model, conf.Server.Recommendations.Milvus.Dimensions.Lyrics)
		writeJSON(w, http.StatusOK, map[string]any{
			"data": []map[string]any{
				{
					"embedding": vector,
					"index":     0,
					"object":    "embedding",
				},
			},
			"model": reqBody.Model,
		})
	}))
	t.Cleanup(embeddingServer.Close)
	conf.Server.Recommendations.TextBaseURL = embeddingServer.URL

	user := model.User{
		ID:          "user-1",
		UserName:    "tester",
		Name:        "Test User",
		NewPassword: "testpass",
	}
	if err := ds.User(context.Background()).Put(&user); err != nil {
		t.Fatalf("failed to create test user: %v", err)
	}

	nativeRouter := New(
		ds,
		nil,
		nil,
		nil,
		tests.NewMockLibraryService(),
		tests.NewMockUserService(),
		core.NewMaintenance(ds),
		nil,
		rec,
	)

	return textRecommendationHarness{
		ds:     ds,
		router: server.JWTVerifier(nativeRouter),
		user:   user,
		rec:    rec,
	}
}

func authenticatedJSONRequest(t *testing.T, user model.User, method string, path string, payload any) *http.Request {
	t.Helper()
	body, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("failed to marshal payload: %v", err)
	}
	return authenticatedRequest(t, user, method, path, bytes.NewReader(body))
}

func authenticatedRequest(t *testing.T, user model.User, method string, path string, body *bytes.Reader) *http.Request {
	t.Helper()
	token, err := auth.CreateToken(&user)
	if err != nil {
		t.Fatalf("failed to create token: %v", err)
	}
	req := httptest.NewRequest(method, path, body)
	req.Header.Set(consts.UIAuthorizationHeader, "Bearer "+token)
	req.Header.Set("Content-Type", "application/json")
	return req
}

func putMediaFiles(t *testing.T, ds model.DataStore, ids ...string) {
	t.Helper()
	repo := ds.MediaFile(context.Background())
	for _, id := range ids {
		mf := model.MediaFile{
			ID:        id,
			Title:     "Title " + id,
			Artist:    "Artist " + id,
			Album:     "Album " + id,
			LibraryID: 1,
		}
		if err := repo.Put(&mf); err != nil {
			t.Fatalf("failed to insert media file %s: %v", id, err)
		}
	}
}

func TestTextRecommendationsRequiresAuthentication(t *testing.T) {
	h := newTextRecommendationHarness(t, &captureRecommendationClient{})

	req := httptest.NewRequest(http.MethodPost, "/recommendations/text", bytes.NewReader([]byte(`{"text":"chill"}`)))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.router.ServeHTTP(w, req)

	if w.Code != http.StatusUnauthorized {
		t.Fatalf("expected %d, got %d", http.StatusUnauthorized, w.Code)
	}
}

func TestTextRecommendationsReturnsServiceUnavailableWhenDisabled(t *testing.T) {
	restore := configtest.SetupConfig()
	defer restore()

	ds := &tests.MockDataStore{}
	auth.Init(ds)

	user := model.User{ID: "user-1", UserName: "tester", NewPassword: "testpass"}
	if err := ds.User(context.Background()).Put(&user); err != nil {
		t.Fatalf("failed to create test user: %v", err)
	}

	nativeRouter := New(
		ds,
		nil,
		nil,
		nil,
		tests.NewMockLibraryService(),
		tests.NewMockUserService(),
		core.NewMaintenance(ds),
		nil,
		nil,
	)
	router := server.JWTVerifier(nativeRouter)

	req := authenticatedJSONRequest(t, user, http.MethodPost, "/recommendations/text", map[string]any{
		"text": "chill",
	})
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)

	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected %d, got %d", http.StatusServiceUnavailable, w.Code)
	}
}

func TestTextRecommendationsValidatesInput(t *testing.T) {
	t.Run("invalid json", func(t *testing.T) {
		h := newTextRecommendationHarness(t, &captureRecommendationClient{})
		req := authenticatedRequest(t, h.user, http.MethodPost, "/recommendations/text", bytes.NewReader([]byte("{")))
		w := httptest.NewRecorder()

		h.router.ServeHTTP(w, req)

		if w.Code != http.StatusBadRequest {
			t.Fatalf("expected %d, got %d", http.StatusBadRequest, w.Code)
		}
		if h.rec.callCount != 0 {
			t.Fatalf("expected recommender to not be called")
		}
	})

	t.Run("missing text", func(t *testing.T) {
		h := newTextRecommendationHarness(t, &captureRecommendationClient{})
		req := authenticatedJSONRequest(t, h.user, http.MethodPost, "/recommendations/text", map[string]any{
			"text": "   ",
		})
		w := httptest.NewRecorder()

		h.router.ServeHTTP(w, req)

		if w.Code != http.StatusBadRequest {
			t.Fatalf("expected %d, got %d", http.StatusBadRequest, w.Code)
		}
		if !strings.Contains(w.Body.String(), "text query is required") {
			t.Fatalf("expected validation message, got %q", w.Body.String())
		}
		if h.rec.callCount != 0 {
			t.Fatalf("expected recommender to not be called")
		}
	})
}

func TestTextRecommendationsLegacyModelNormalizationAndResponseEnrichment(t *testing.T) {
	neg := 0.42
	rec := &captureRecommendationClient{
		response: &subsonic.RecommendationResponse{
			Tracks: []subsonic.RecommendationItem{
				{
					TrackID:            "result-1",
					Score:              0.91,
					Models:             []string{"lyrics", "description"},
					NegativeSimilarity: &neg,
				},
			},
			Warnings: []string{"partial coverage"},
		},
	}
	h := newTextRecommendationHarness(t, rec)
	putMediaFiles(t, h.ds, "result-1")

	req := authenticatedJSONRequest(t, h.user, http.MethodPost, "/recommendations/text", map[string]any{
		"text":   "dream pop with airy vocals",
		"model":  "qwen3",
		"models": []string{"flamingo"},
		"limit":  5,
	})
	w := httptest.NewRecorder()

	h.router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected %d, got %d: %s", http.StatusOK, w.Code, w.Body.String())
	}
	if rec.callCount != 1 {
		t.Fatalf("expected recommender call count 1, got %d", rec.callCount)
	}
	if rec.lastMode != modeTextRecommendations {
		t.Fatalf("expected mode %q, got %q", modeTextRecommendations, rec.lastMode)
	}

	assertStringSetContainsAll(t, rec.lastReq.Models, []string{"flamingo", "lyrics", "description"})
	if len(rec.lastReq.Seeds) != 1 {
		t.Fatalf("expected one text seed, got %d", len(rec.lastReq.Seeds))
	}

	seed := rec.lastReq.Seeds[0]
	if seed.TrackID != "_text_query_" || seed.Source != "text" {
		t.Fatalf("unexpected text seed: %#v", seed)
	}
	if len(seed.Embeddings) != 2 {
		t.Fatalf("expected 2 per-model embeddings, got %d", len(seed.Embeddings))
	}
	lyricsEmbedding, lyricsOK := seed.Embeddings["lyrics"]
	descEmbedding, descOK := seed.Embeddings["description"]
	if !lyricsOK || !descOK {
		t.Fatalf("expected lyrics and description embeddings, got keys %#v", mapKeys(seed.Embeddings))
	}
	if len(lyricsEmbedding) != embeddingDimensionForModel("lyrics") {
		t.Fatalf("unexpected lyrics embedding dimension %d", len(lyricsEmbedding))
	}
	if len(descEmbedding) != embeddingDimensionForModel("description") {
		t.Fatalf("unexpected description embedding dimension %d", len(descEmbedding))
	}
	if len(seed.Embedding) == 0 {
		t.Fatalf("expected legacy primary embedding to be set")
	}
	if seed.Embedding[0] != lyricsEmbedding[0] {
		t.Fatalf("expected primary embedding to use first text target (lyrics)")
	}

	var resp recommendationResponsePayload
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	if len(resp.TrackIDs) != 1 || resp.TrackIDs[0] != "result-1" {
		t.Fatalf("unexpected track IDs: %#v", resp.TrackIDs)
	}
	if len(resp.Tracks) != 1 {
		t.Fatalf("expected one enriched track, got %d", len(resp.Tracks))
	}
	if len(resp.Warnings) != 1 || resp.Warnings[0] != "partial coverage" {
		t.Fatalf("expected warning propagation, got %#v", resp.Warnings)
	}
	if len(resp.Tracks[0].Models) != 2 {
		t.Fatalf("expected enriched model metadata, got %#v", resp.Tracks[0].Models)
	}
	if resp.Tracks[0].NegativeSimilarity == nil || *resp.Tracks[0].NegativeSimilarity != neg {
		t.Fatalf("expected enriched negative similarity, got %#v", resp.Tracks[0].NegativeSimilarity)
	}
}

func TestTextRecommendationsUsesDefaultTextModelForSeedEmbeddings(t *testing.T) {
	rec := &captureRecommendationClient{
		response: &subsonic.RecommendationResponse{
			Tracks: []subsonic.RecommendationItem{
				{TrackID: "result-1", Score: 0.9, Models: []string{"lyrics", "description"}},
			},
		},
	}
	h := newTextRecommendationHarness(t, rec)
	putMediaFiles(t, h.ds, "result-1")

	prompt := "warm downtempo grooves"
	req := authenticatedJSONRequest(t, h.user, http.MethodPost, "/recommendations/text", map[string]any{
		"text": prompt,
	})
	w := httptest.NewRecorder()

	h.router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected %d, got %d: %s", http.StatusOK, w.Code, w.Body.String())
	}
	if rec.callCount != 1 {
		t.Fatalf("expected recommender call count 1, got %d", rec.callCount)
	}
	if len(rec.lastReq.Seeds) != 1 {
		t.Fatalf("expected one seed, got %d", len(rec.lastReq.Seeds))
	}

	seed := rec.lastReq.Seeds[0]
	gotLyrics := seed.Embeddings["lyrics"]
	gotDescription := seed.Embeddings["description"]
	wantVector := mockEmbeddingServiceVector(prompt, defaultTextEmbedderModel, embeddingDimensionForModel("lyrics"))

	if !vectorsEqual(gotLyrics, wantVector) {
		t.Fatalf("expected default model %q to be used for lyrics seed", defaultTextEmbedderModel)
	}
	if !vectorsEqual(gotDescription, wantVector) {
		t.Fatalf("expected default model %q to be used for description seed", defaultTextEmbedderModel)
	}
}

func TestTextRecommendationsHybridPathBuildsSeedsAndExclusions(t *testing.T) {
	rec := &captureRecommendationClient{
		response: &subsonic.RecommendationResponse{
			Tracks: []subsonic.RecommendationItem{
				{TrackID: "result-1", Score: 0.85, Models: []string{"description", "flamingo"}},
			},
		},
	}
	h := newTextRecommendationHarness(t, rec)
	putMediaFiles(t, h.ds, "seed-1", "positive-1", "playlist-excluded", "result-1")

	h.ds.MockedPlaylist = &tests.MockPlaylistRepo{
		Entity: &model.Playlist{
			ID:   "pls-1",
			Name: "Blocklist",
		},
		TracksReturn: &staticPlaylistTrackRepo{
			tracks: model.PlaylistTracks{
				{ID: "pt-1", PlaylistID: "pls-1", MediaFileID: "playlist-excluded"},
			},
		},
	}

	req := authenticatedJSONRequest(t, h.user, http.MethodPost, "/recommendations/text", map[string]any{
		"text":             "high energy electronic",
		"textTargets":      []string{"description"},
		"songIds":          []string{"seed-1"},
		"positiveTrackIds": []string{"positive-1"},
		"excludeTrackIds":  []string{"explicit-a", "dup"},
		"negativeTrackIds": []string{"dup", "explicit-b"},
		"excludePlaylistIds": []string{
			"pls-1",
		},
	})
	w := httptest.NewRecorder()

	h.router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected %d, got %d: %s", http.StatusOK, w.Code, w.Body.String())
	}
	if rec.callCount != 1 {
		t.Fatalf("expected recommender call count 1, got %d", rec.callCount)
	}

	assertStringSetContainsAll(t, rec.lastReq.Models, []string{"description", "flamingo"})
	assertStringSetContainsAll(t, rec.lastReq.ExcludeTrackIDs, []string{"explicit-a", "dup", "explicit-b", "playlist-excluded"})

	if len(rec.lastReq.Seeds) != 3 {
		t.Fatalf("expected text + song + positive seeds, got %d", len(rec.lastReq.Seeds))
	}
	assertSeedPresent(t, rec.lastReq.Seeds, "_text_query_")
	assertSeedPresent(t, rec.lastReq.Seeds, "seed-1")
	assertSeedPresent(t, rec.lastReq.Seeds, "positive-1")
}

func TestTextRecommendationsReturnsInternalErrorWhenRecommenderFails(t *testing.T) {
	rec := &captureRecommendationClient{err: errors.New("backend boom")}
	h := newTextRecommendationHarness(t, rec)

	req := authenticatedJSONRequest(t, h.user, http.MethodPost, "/recommendations/text", map[string]any{
		"text": "late night jazz",
	})
	w := httptest.NewRecorder()

	h.router.ServeHTTP(w, req)

	if w.Code != http.StatusInternalServerError {
		t.Fatalf("expected %d, got %d", http.StatusInternalServerError, w.Code)
	}
	if !strings.Contains(w.Body.String(), "recommendation failed") {
		t.Fatalf("expected recommendation error message, got %q", w.Body.String())
	}
	if rec.callCount != 1 {
		t.Fatalf("expected recommender call count 1, got %d", rec.callCount)
	}
}

func assertStringSetContainsAll(t *testing.T, got []string, expected []string) {
	t.Helper()
	seen := make(map[string]struct{}, len(got))
	for _, item := range got {
		seen[item] = struct{}{}
	}
	for _, item := range expected {
		if _, ok := seen[item]; !ok {
			t.Fatalf("expected %q in %#v", item, got)
		}
	}
}

func assertSeedPresent(t *testing.T, seeds []subsonic.RecommendationSeed, id string) {
	t.Helper()
	for _, seed := range seeds {
		if seed.TrackID == id {
			return
		}
	}
	t.Fatalf("expected seed %q in %#v", id, seeds)
}

func mapKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for key := range m {
		keys = append(keys, key)
	}
	return keys
}

func mockEmbeddingServiceVector(text string, model string, dim int) []float64 {
	// Simulates a remote embedding service: deterministic by text+model, no per-target variance.
	return deterministicTextEmbedding(text, model, "remote", dim)
}
