package scanner

import (
	"bufio"
	"context"
	"encoding/json"
	"net"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

// createMockSocketServer creates a mock Unix socket server that handles a single connection.
// The handler function is called with the connection for custom request/response handling.
func createMockSocketServer(t *testing.T, handler func(conn net.Conn)) string {
	socketPath := filepath.Join(t.TempDir(), "test.sock")
	listener, err := net.Listen("unix", socketPath)
	require.NoError(t, err)

	go func() {
		conn, err := listener.Accept()
		if err != nil {
			return
		}
		handler(conn)
		conn.Close()
	}()

	t.Cleanup(func() { listener.Close() })
	return socketPath
}

func TestSocketEmbeddingClientCheckEmbeddingPayload(t *testing.T) {
	var received map[string]any

	socketPath := createMockSocketServer(t, func(conn net.Conn) {
		reader := bufio.NewReader(conn)
		line, err := reader.ReadBytes('\n')
		require.NoError(t, err)

		err = json.Unmarshal(line, &received)
		require.NoError(t, err)

		response := `{"embedded":true,"hasDescription":false,"name":"Artist - Title"}` + "\n"
		_, _ = conn.Write([]byte(response))
	})

	client := &socketEmbeddingClient{
		socketPath:    socketPath,
		statusTimeout: statusCheckTimeout,
		embedTimeout:  statusCheckTimeout,
	}

	status, err := client.CheckEmbedding(context.Background(), embeddingCandidate{
		LibraryID:   1,
		LibraryPath: "/music",
		TrackPath:   "folder/song.flac",
		Artist:      "Artist",
		Title:       "Title",
		Album:       "Album",
	})
	require.NoError(t, err)
	require.True(t, status.Embedded)
	require.False(t, status.HasDescription)
	require.Equal(t, "Artist - Title", status.Name)

	// Verify the request payload
	require.Equal(t, "status", received["action"])
	require.Equal(t, "Artist", received["artist"])
	require.Equal(t, "Title", received["title"])
	require.Equal(t, "Album", received["album"])

	alt := received["alternate_names"].([]any)
	require.Len(t, alt, 1)
	require.Equal(t, "song.flac", alt[0])

	require.NotEmpty(t, received["track_id"])
}

func TestSocketEmbeddingClientEmbedSongPayload(t *testing.T) {
	var received map[string]any

	socketPath := createMockSocketServer(t, func(conn net.Conn) {
		reader := bufio.NewReader(conn)
		line, err := reader.ReadBytes('\n')
		require.NoError(t, err)

		err = json.Unmarshal(line, &received)
		require.NoError(t, err)

		response := `{"status":"ok"}` + "\n"
		_, _ = conn.Write([]byte(response))
	})

	client := &socketEmbeddingClient{
		socketPath:    socketPath,
		statusTimeout: statusCheckTimeout,
		embedTimeout:  statusCheckTimeout,
	}

	err := client.EmbedSong(context.Background(), embeddingCandidate{
		LibraryID:   2,
		LibraryPath: "/music",
		TrackPath:   "artist/song.flac",
		Artist:      "Artist",
		Title:       "Song",
		Album:       "Album",
	})
	require.NoError(t, err)

	// Verify the request payload
	require.Equal(t, "embed", received["action"])
	require.Equal(t, "Artist", received["artist"])
	require.Equal(t, "Song", received["title"])
	require.Equal(t, "Album", received["album"])
	require.Equal(t, filepath.Join("/music", "artist", "song.flac"), received["music_file"])
	require.Equal(t, "song.flac", received["name"])
	require.NotEmpty(t, received["track_id"])
}

func TestSocketEmbeddingClientEmbedSongError(t *testing.T) {
	socketPath := createMockSocketServer(t, func(conn net.Conn) {
		reader := bufio.NewReader(conn)
		_, _ = reader.ReadBytes('\n')

		response := `{"status":"error","message":"test error message"}` + "\n"
		_, _ = conn.Write([]byte(response))
	})

	client := &socketEmbeddingClient{
		socketPath:    socketPath,
		statusTimeout: statusCheckTimeout,
		embedTimeout:  statusCheckTimeout,
	}

	err := client.EmbedSong(context.Background(), embeddingCandidate{
		LibraryID:   1,
		LibraryPath: "/music",
		TrackPath:   "song.flac",
	})
	require.Error(t, err)
	require.Contains(t, err.Error(), "test error message")
}

func TestSocketEmbeddingClientConnectionError(t *testing.T) {
	client := &socketEmbeddingClient{
		socketPath:    "/nonexistent/socket.sock",
		statusTimeout: statusCheckTimeout,
		embedTimeout:  statusCheckTimeout,
	}

	_, err := client.CheckEmbedding(context.Background(), embeddingCandidate{})
	require.Error(t, err)
	require.Contains(t, err.Error(), "connect to socket")
}

func TestSocketEmbeddingClientInvalidResponse(t *testing.T) {
	socketPath := createMockSocketServer(t, func(conn net.Conn) {
		reader := bufio.NewReader(conn)
		_, _ = reader.ReadBytes('\n')
		_, _ = conn.Write([]byte("not-json\n"))
	})

	client := &socketEmbeddingClient{
		socketPath:    socketPath,
		statusTimeout: statusCheckTimeout,
		embedTimeout:  statusCheckTimeout,
	}

	_, err := client.CheckEmbedding(context.Background(), embeddingCandidate{})
	require.Error(t, err)
	require.Contains(t, err.Error(), "decode response")
}
