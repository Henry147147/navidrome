package nativeapi

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/navidrome/navidrome/conf"
)

const embedEndpoint = "/embed/audio"

type embedRequest struct {
	MusicName string         `json:"name"`
	MusicFile string         `json:"music_file"`
	CueFile   string         `json:"cue_file,omitempty"`
	Settings  map[string]any `json:"settings,omitempty"`
}

type EmbedClient interface {
	Embed(musicPath, musicName, cuePath string, settings map[string]any) (map[string]any, error)
}

type embedHTTPClient struct {
	baseURL    string
	httpClient *http.Client
}

func NewEmbedHTTPClient(baseURL string, timeout time.Duration) EmbedClient {
	base := strings.TrimSuffix(strings.TrimSpace(baseURL), "/")
	if base == "" {
		return noopEmbedClient{}
	}
	return &embedHTTPClient{
		baseURL:    base,
		httpClient: &http.Client{Timeout: timeout},
	}
}

func (c *embedHTTPClient) Embed(musicPath, musicName, cuePath string, settings map[string]any) (map[string]any, error) {
	if c == nil {
		return nil, fmt.Errorf("embed client not configured")
	}

	payload := embedRequest{
		MusicFile: musicPath,
		MusicName: musicName,
		Settings:  settings,
	}
	if cuePath != "" {
		payload.CueFile = cuePath
	}

	data, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("encode embed request: %w", err)
	}

	url := c.baseURL + embedEndpoint
	req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("create embed request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("call embed service: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= http.StatusMultipleChoices {
		return nil, fmt.Errorf("embed service returned %s", resp.Status)
	}

	var decoded map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		return nil, fmt.Errorf("decode embed response: %w", err)
	}

	return decoded, nil
}

type noopEmbedClient struct{}

func (noopEmbedClient) Embed(string, string, string, map[string]any) (map[string]any, error) {
	return nil, fmt.Errorf("embed service disabled")
}

var (
	embedClient     EmbedClient
	embedClientOnce sync.Once
)

func getEmbedClient() EmbedClient {
	embedClientOnce.Do(func() {
		base := conf.Server.Recommendations.BaseURL
		embedClient = NewEmbedHTTPClient(base, conf.Server.Recommendations.Timeout)
	})
	return embedClient
}
