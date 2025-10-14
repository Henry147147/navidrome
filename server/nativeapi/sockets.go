package nativeapi

import (
	"encoding/json"
	"fmt"
	"net"
)

const embedSocketPath = "/tmp/navidrome_embed.sock"

type embedRequest struct {
	MusicName string         `json:"name"`
	MusicFile string         `json:"music_file"`
	CueFile   string         `json:"cue_file,omitempty"`
	Settings  map[string]any `json:"settings,omitempty"`
}

type EmbedClient interface {
	Embed(musicPath, musicName, cuePath string, settings map[string]any) (map[string]any, error)
}

type embedSocketClient struct {
	socketPath string
}

func NewEmbedSocketClient(socketPath string) EmbedClient {
	return &embedSocketClient{socketPath: socketPath}
}

func (c *embedSocketClient) Embed(musicPath, musicName, cuePath string, settings map[string]any) (map[string]any, error) {
	conn, err := net.Dial("unix", c.socketPath)
	if err != nil {
		return nil, fmt.Errorf("dial embed server: %w", err)
	}
	defer func() { _ = conn.Close() }()
	reqPayload := embedRequest{
		MusicFile: musicPath,
		MusicName: musicName,
		Settings:  settings,
	}
	if cuePath != "" {
		reqPayload.CueFile = cuePath
	}

	if err := json.NewEncoder(conn).Encode(&reqPayload); err != nil {
		return nil, fmt.Errorf("send request to embed server: %w", err)
	}

	if unixConn, ok := conn.(*net.UnixConn); ok {
		_ = unixConn.CloseWrite()
	}

	var resp map[string]any
	if err := json.NewDecoder(conn).Decode(&resp); err != nil {
		return nil, fmt.Errorf("read response from embed server: %w", err)
	}

	return resp, nil
}

var embedClient EmbedClient = NewEmbedSocketClient(embedSocketPath)
