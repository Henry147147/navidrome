package nativeapi

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/navidrome/navidrome/conf"
)

type openAIEmbeddingRequest struct {
	Input          string `json:"input"`
	Model          string `json:"model,omitempty"`
	EncodingFormat string `json:"encoding_format,omitempty"`
	Dimensions     int    `json:"dimensions,omitempty"`
}

type openAIEmbeddingResponse struct {
	Data []struct {
		Embedding []float64 `json:"embedding"`
		Index     int       `json:"index"`
		Object    string    `json:"object"`
	} `json:"data"`
	Model string `json:"model"`
	Error *struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    any    `json:"code"`
	} `json:"error,omitempty"`
}

func textEmbeddingHTTPTimeout() time.Duration {
	timeout := conf.Server.Recommendations.Timeout
	if timeout < 30*time.Second {
		timeout = 30 * time.Second
	}
	return timeout
}

// getTextEmbedding fetches a query embedding from a llama.cpp-compatible OpenAI embeddings endpoint.
func (n *Router) getTextEmbedding(ctx context.Context, text string, model string, desiredDim int) ([]float64, error) {
	baseURL := strings.TrimRight(textEmbedBaseURL(), "/")
	if baseURL == "" {
		return nil, fmt.Errorf("text embedding base URL is not configured")
	}

	payload := openAIEmbeddingRequest{
		Input:          text,
		Model:          model,
		EncodingFormat: "float",
	}
	if desiredDim > 0 {
		payload.Dimensions = desiredDim
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("encode embeddings request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/v1/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create embeddings request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: textEmbeddingHTTPTimeout()}
	resp, err := client.Do(req) // #nosec G704 -- destination URL comes from trusted server config
	if err != nil {
		return nil, fmt.Errorf("call embeddings endpoint: %w", err)
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read embeddings response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		var e openAIEmbeddingResponse
		if err := json.Unmarshal(data, &e); err == nil && e.Error != nil && strings.TrimSpace(e.Error.Message) != "" {
			return nil, fmt.Errorf("text embedding service returned status %d: %s", resp.StatusCode, e.Error.Message)
		}
		return nil, fmt.Errorf("text embedding service returned status %d", resp.StatusCode)
	}

	var parsed openAIEmbeddingResponse
	if err := json.Unmarshal(data, &parsed); err == nil {
		if parsed.Error != nil && strings.TrimSpace(parsed.Error.Message) != "" {
			return nil, fmt.Errorf("text embedding service returned error: %s", parsed.Error.Message)
		}
		if len(parsed.Data) > 0 && len(parsed.Data[0].Embedding) > 0 {
			return normalizeEmbeddingDimension(parsed.Data[0].Embedding, desiredDim)
		}
	}

	// Compatibility fallback for non-OpenAI format responses:
	// {"embedding":[...]}
	var legacy struct {
		Embedding []float64 `json:"embedding"`
	}
	if err := json.Unmarshal(data, &legacy); err == nil && len(legacy.Embedding) > 0 {
		return normalizeEmbeddingDimension(legacy.Embedding, desiredDim)
	}

	return nil, fmt.Errorf("text embedding response did not contain an embedding")
}

func normalizeEmbeddingDimension(embedding []float64, desiredDim int) ([]float64, error) {
	if desiredDim <= 0 {
		return embedding, nil
	}
	if len(embedding) == desiredDim {
		return embedding, nil
	}
	if len(embedding) > desiredDim {
		// Qwen3 embedding models support MRL-style subdimensions; truncation keeps compatibility
		// with existing smaller-dimension collections.
		return append([]float64(nil), embedding[:desiredDim]...), nil
	}
	return nil, fmt.Errorf("embedding dimension too small: expected %d got %d", desiredDim, len(embedding))
}
