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
	"time"

	"github.com/navidrome/navidrome/conf"
	"github.com/navidrome/navidrome/conf/configtest"
	"github.com/navidrome/navidrome/consts"
	"github.com/navidrome/navidrome/core"
	"github.com/navidrome/navidrome/core/auth"
	"github.com/navidrome/navidrome/model"
	"github.com/navidrome/navidrome/recommender/engine"
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
	native *Router
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

	// Keep the shared MuQ-MuLan text embedding dimension stable for deterministic test vectors.
	conf.Server.Recommendations.Milvus.Dimensions.MuQMulan = 16

	embeddingServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet && r.URL.Path == "/v1/models" {
			writeJSON(w, http.StatusOK, map[string]any{"data": []map[string]any{{"id": "muq_mulan"}}})
			return
		}
		if r.Method == http.MethodGet && r.URL.Path == "/batch/progress" {
			writeJSON(w, http.StatusOK, map[string]any{"status": "idle"})
			return
		}
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
		vector := mockEmbeddingServiceVector(reqBody.Input, reqBody.Model, conf.Server.Recommendations.Milvus.Dimensions.MuQMulan)
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
	conf.Server.Recommendations.BatchBaseURL = embeddingServer.URL

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
		nil,
	)

	return textRecommendationHarness{
		ds:     ds,
		native: nativeRouter,
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
			Duration:  180,
		}
		if err := repo.Put(&mf); err != nil {
			t.Fatalf("failed to insert media file %s: %v", id, err)
		}
	}
}

func putMediaFile(t *testing.T, ds model.DataStore, mf model.MediaFile) {
	t.Helper()
	if mf.LibraryID == 0 {
		mf.LibraryID = 1
	}
	if mf.Duration == 0 {
		mf.Duration = 180
	}
	if err := ds.MediaFile(context.Background()).Put(&mf); err != nil {
		t.Fatalf("failed to insert media file %s: %v", mf.ID, err)
	}
}

func putMediaFileWithDuration(t *testing.T, ds model.DataStore, id string, duration float32) {
	t.Helper()
	mf := model.MediaFile{
		ID:        id,
		Title:     "Title " + id,
		Artist:    "Artist " + id,
		Album:     "Album " + id,
		LibraryID: 1,
		Duration:  duration,
	}
	if err := ds.MediaFile(context.Background()).Put(&mf); err != nil {
		t.Fatalf("failed to insert media file %s: %v", id, err)
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

func TestRecommendationHealthEndpointReportsAvailableModes(t *testing.T) {
	h := newTextRecommendationHarness(t, &captureRecommendationClient{})

	req := authenticatedRequest(
		t,
		h.user,
		http.MethodGet,
		"/recommendations/health",
		bytes.NewReader(nil),
	)
	w := httptest.NewRecorder()

	h.router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected %d, got %d", http.StatusOK, w.Code)
	}
	if !strings.Contains(w.Body.String(), `"status":"ready"`) {
		t.Fatalf("expected ready health payload, got %q", w.Body.String())
	}
	if !strings.Contains(w.Body.String(), `"availableModes":["recent","favorites","all","discovery","custom","text"]`) {
		t.Fatalf("expected available modes in health payload, got %q", w.Body.String())
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
					Models:             []string{engine.ModelMuQMulan},
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

	assertStringSetContainsAll(t, rec.lastReq.Models, []string{engine.ModelMuQAudio, engine.ModelMuQMulan})
	if len(rec.lastReq.Seeds) != 1 {
		t.Fatalf("expected one text seed, got %d", len(rec.lastReq.Seeds))
	}

	seed := rec.lastReq.Seeds[0]
	if seed.TrackID != "_text_query_" || seed.Source != "text" {
		t.Fatalf("unexpected text seed: %#v", seed)
	}
	if len(seed.Embeddings) != 1 {
		t.Fatalf("expected 1 per-model embedding, got %d", len(seed.Embeddings))
	}
	sharedEmbedding, ok := seed.Embeddings[engine.ModelMuQMulan]
	if !ok {
		t.Fatalf("expected muq mulan embedding, got keys %#v", mapKeys(seed.Embeddings))
	}
	if len(sharedEmbedding) != embeddingDimensionForModel(engine.ModelMuQMulan) {
		t.Fatalf("unexpected shared embedding dimension %d", len(sharedEmbedding))
	}
	if len(seed.Embedding) == 0 {
		t.Fatalf("expected primary embedding to be set")
	}
	if seed.Embedding[0] != sharedEmbedding[0] {
		t.Fatalf("expected primary embedding to mirror muq mulan seed")
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
	if len(resp.Tracks[0].Models) != 1 || resp.Tracks[0].Models[0] != engine.ModelMuQMulan {
		t.Fatalf("expected enriched model metadata, got %#v", resp.Tracks[0].Models)
	}
	if resp.Tracks[0].NegativeSimilarity == nil || *resp.Tracks[0].NegativeSimilarity != neg {
		t.Fatalf("expected enriched negative similarity, got %#v", resp.Tracks[0].NegativeSimilarity)
	}
}

func TestTextRecommendationsBackfillAfterDurationFiltering(t *testing.T) {
	rec := &captureRecommendationClient{
		response: &subsonic.RecommendationResponse{
			Tracks: []subsonic.RecommendationItem{
				{TrackID: "short-1", Score: 0.99, Models: []string{engine.ModelMuQMulan}},
				{TrackID: "short-2", Score: 0.97, Models: []string{engine.ModelMuQMulan}},
				{TrackID: "good-1", Score: 0.95, Models: []string{engine.ModelMuQMulan}},
				{TrackID: "good-2", Score: 0.93, Models: []string{engine.ModelMuQMulan}},
				{TrackID: "good-3", Score: 0.91, Models: []string{engine.ModelMuQMulan}},
			},
		},
	}
	h := newTextRecommendationHarness(t, rec)
	putMediaFileWithDuration(t, h.ds, "short-1", 10)
	putMediaFileWithDuration(t, h.ds, "short-2", 20)
	putMediaFileWithDuration(t, h.ds, "good-1", 180)
	putMediaFileWithDuration(t, h.ds, "good-2", 210)
	putMediaFileWithDuration(t, h.ds, "good-3", 240)

	settingsJSON, err := json.Marshal(recommendationSettings{
		MixLength:               30,
		BaseDiversity:           0.2,
		DiscoveryExploration:    0.6,
		SeedRecencyWindowDays:   60,
		FavoritesBlendWeight:    0.85,
		LowRatingPenalty:        0.85,
		MinTrackDurationSeconds: 30,
		MaxTrackDurationSeconds: 300,
	})
	if err != nil {
		t.Fatalf("failed to marshal recommendation settings: %v", err)
	}
	if err := h.ds.UserProps(context.Background()).Put(h.user.ID, recommendationSettingsKey, string(settingsJSON)); err != nil {
		t.Fatalf("failed to save recommendation settings: %v", err)
	}

	req := authenticatedJSONRequest(t, h.user, http.MethodPost, "/recommendations/text", map[string]any{
		"text":  "gentle synth textures",
		"limit": 3,
	})
	w := httptest.NewRecorder()

	h.router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected %d, got %d: %s", http.StatusOK, w.Code, w.Body.String())
	}
	if rec.callCount != 1 {
		t.Fatalf("expected recommender call count 1, got %d", rec.callCount)
	}
	if rec.lastReq.Limit != expandedRecommendationLimit(3) {
		t.Fatalf("expected expanded request limit %d, got %d", expandedRecommendationLimit(3), rec.lastReq.Limit)
	}

	var resp recommendationResponsePayload
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	expected := []string{"good-1", "good-2", "good-3"}
	if len(resp.TrackIDs) != len(expected) {
		t.Fatalf("expected %#v, got %#v", expected, resp.TrackIDs)
	}
	for idx, id := range expected {
		if resp.TrackIDs[idx] != id {
			t.Fatalf("expected track %q at index %d, got %#v", id, idx, resp.TrackIDs)
		}
	}
	for _, warning := range resp.Warnings {
		lower := strings.ToLower(warning)
		if strings.Contains(lower, "duration") || strings.Contains(lower, "allowed range") {
			t.Fatalf("did not expect duration warning, got %#v", resp.Warnings)
		}
	}
}

func TestTextRecommendationsUsesDefaultTextModelForSeedEmbeddings(t *testing.T) {
	rec := &captureRecommendationClient{
		response: &subsonic.RecommendationResponse{
			Tracks: []subsonic.RecommendationItem{
				{TrackID: "result-1", Score: 0.9, Models: []string{engine.ModelMuQMulan}},
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
	gotShared := seed.Embeddings[engine.ModelMuQMulan]
	wantVector := mockEmbeddingServiceVector(prompt, defaultTextEmbedderModel, embeddingDimensionForModel(engine.ModelMuQMulan))

	if !vectorsEqual(gotShared, wantVector) {
		t.Fatalf("expected default model %q to be used for shared text seed", defaultTextEmbedderModel)
	}
}

func TestTextRecommendationsHybridPathBuildsSeedsAndExclusions(t *testing.T) {
	rec := &captureRecommendationClient{
		response: &subsonic.RecommendationResponse{
			Tracks: []subsonic.RecommendationItem{
				{TrackID: "result-1", Score: 0.85, Models: []string{engine.ModelMuQMulan, engine.ModelMuQAudio}},
			},
		},
	}
	h := newTextRecommendationHarness(t, rec)
	putMediaFiles(t, h.ds, "seed-1", "positive-1", "playlist-excluded", "result-1")

	h.ds.MockedPlaylist = &tests.MockPlaylistRepo{
		Data: map[string]*model.Playlist{
			"pls-1": {
				ID:   "pls-1",
				Name: "Blocklist",
			},
		},
		TracksRepo: &staticPlaylistTrackRepo{
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

	assertStringSetContainsAll(t, rec.lastReq.Models, []string{engine.ModelMuQMulan, engine.ModelMuQAudio})
	assertStringSetContainsAll(t, rec.lastReq.ExcludeTrackIDs, []string{"explicit-a", "dup", "explicit-b", "playlist-excluded"})

	if len(rec.lastReq.Seeds) != 3 {
		t.Fatalf("expected text + song + positive seeds, got %d", len(rec.lastReq.Seeds))
	}
	assertSeedPresent(t, rec.lastReq.Seeds, "_text_query_")
	assertSeedPresent(t, rec.lastReq.Seeds, "seed-1")
	assertSeedPresent(t, rec.lastReq.Seeds, "positive-1")

	for _, seed := range rec.lastReq.Seeds {
		if seed.TrackID != "_text_query_" {
			continue
		}
		if _, ok := seed.Embeddings[engine.ModelMuQMulan]; !ok {
			t.Fatalf("expected text seed to carry muq mulan embedding, got %#v", seed.Embeddings)
		}
		if _, ok := seed.Embeddings[engine.ModelMuQAudio]; ok {
			t.Fatalf("did not expect text seed to carry muq audio embedding, got %#v", seed.Embeddings)
		}
	}
}

func TestTextRecommendationsInfersPositiveSeedsFromPromptArtists(t *testing.T) {
	rec := &captureRecommendationClient{
		response: &subsonic.RecommendationResponse{
			Tracks: []subsonic.RecommendationItem{
				{TrackID: "result-1", Score: 0.9, Models: []string{engine.ModelMuQMulan, engine.ModelMuQAudio}},
			},
		},
	}
	h := newTextRecommendationHarness(t, rec)

	now := time.Now()
	putMediaFile(t, h.ds, model.MediaFile{
		ID:          "lorde-seed",
		Title:       "400 Lux",
		Artist:      "Lorde",
		AlbumArtist: "Lorde",
		Album:       "Pure Heroine",
		LibraryID:   1,
		Duration:    220,
		Annotations: model.Annotations{
			PlayCount: 2,
			PlayDate:  &now,
			Starred:   true,
			StarredAt: &now,
			Rating:    5,
			RatedAt:   &now,
		},
	})
	putMediaFiles(t, h.ds, "result-1")

	req := authenticatedJSONRequest(t, h.user, http.MethodPost, "/recommendations/text", map[string]any{
		"text": "moody contemporary pop with lorde energy",
	})
	w := httptest.NewRecorder()

	h.router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected %d, got %d: %s", http.StatusOK, w.Code, w.Body.String())
	}
	assertSeedPresent(t, rec.lastReq.Seeds, "_text_query_")
	assertSeedPresent(t, rec.lastReq.Seeds, "lorde-seed")
	assertStringSetContainsAll(t, rec.lastReq.Models, []string{engine.ModelMuQMulan, engine.ModelMuQAudio})
}

func TestTextRecommendationsReturnsStructuredErrorWhenRecommenderFails(t *testing.T) {
	rec := &captureRecommendationClient{err: errors.New("backend boom")}
	h := newTextRecommendationHarness(t, rec)

	req := authenticatedJSONRequest(t, h.user, http.MethodPost, "/recommendations/text", map[string]any{
		"text": "late night jazz",
	})
	w := httptest.NewRecorder()

	h.router.ServeHTTP(w, req)

	if w.Code != http.StatusBadGateway {
		t.Fatalf("expected %d, got %d", http.StatusBadGateway, w.Code)
	}
	if !strings.Contains(w.Body.String(), `"code":"recommendation_backend_error"`) {
		t.Fatalf("expected structured recommendation error, got %q", w.Body.String())
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
