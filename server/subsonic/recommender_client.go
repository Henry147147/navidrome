package subsonic

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/navidrome/navidrome/conf"
)

type RecommendationClient interface {
	Recommend(ctx context.Context, mode string, payload RecommendationRequest) (*RecommendationResponse, error)
}

type RecommendationSeed struct {
	TrackID  string     `json:"track_id"`
	Weight   float64    `json:"weight,omitempty"`
	Source   string     `json:"source"`
	PlayedAt *time.Time `json:"played_at,omitempty"`
}

type RecommendationRequest struct {
	UserID            string               `json:"user_id"`
	UserName          string               `json:"user_name"`
	Limit             int                  `json:"limit"`
	Mode              string               `json:"mode"`
	Seeds             []RecommendationSeed `json:"seeds"`
	Diversity         float64              `json:"diversity"`
	ExcludeTrackIDs   []string             `json:"exclude_track_ids,omitempty"`
	LibraryIDs        []int                `json:"library_ids,omitempty"`
	DislikedTrackIDs  []string             `json:"disliked_track_ids,omitempty"`
	DislikedArtistIDs []string             `json:"disliked_artist_ids,omitempty"`
	DislikeStrength   float64              `json:"dislike_strength,omitempty"`
}

type RecommendationItem struct {
	TrackID string  `json:"track_id"`
	Score   float64 `json:"score"`
	Reason  string  `json:"reason,omitempty"`
}

type RecommendationResponse struct {
	Tracks   []RecommendationItem `json:"tracks"`
	Warnings []string             `json:"warnings"`
}

func (r RecommendationResponse) TrackIDs() []string {
	ids := make([]string, 0, len(r.Tracks))
	for _, item := range r.Tracks {
		if item.TrackID != "" {
			ids = append(ids, item.TrackID)
		}
	}
	return ids
}

type recommendationHTTPClient struct {
	baseURL    string
	httpClient *http.Client
}

func newRecommendationHTTPClient(baseURL string, timeout time.Duration) RecommendationClient {
	base := strings.TrimSuffix(strings.TrimSpace(baseURL), "/")
	if base == "" {
		return noopRecommendationClient{}
	}
	client := &http.Client{Timeout: timeout}
	return &recommendationHTTPClient{baseURL: base, httpClient: client}
}

func (c *recommendationHTTPClient) Recommend(ctx context.Context, mode string, payload RecommendationRequest) (*RecommendationResponse, error) {
	if c == nil {
		return nil, fmt.Errorf("recommendation client not configured")
	}
	url := fmt.Sprintf("%s/playlist/%s", c.baseURL, strings.Trim(strings.ToLower(mode), "/"))
	data, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("encode recommendation request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("create recommendation request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("call recommendation service: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("recommendation service returned %s", resp.Status)
	}
	var parsed RecommendationResponse
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return nil, fmt.Errorf("decode recommendation response: %w", err)
	}
	return &parsed, nil
}

type noopRecommendationClient struct{}

func (noopRecommendationClient) Recommend(context.Context, string, RecommendationRequest) (*RecommendationResponse, error) {
	return &RecommendationResponse{Warnings: []string{"recommendation service disabled"}}, nil
}

func NewRecommendationClient() RecommendationClient {
	opts := conf.Server.Recommendations
	return newRecommendationHTTPClient(opts.BaseURL, opts.Timeout)
}
