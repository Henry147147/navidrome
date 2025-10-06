package nativeapi

import (
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/go-chi/chi/v5"
	"github.com/navidrome/navidrome/log"
)

const embedSocketPath = "/tmp/navidrome_embed.sock"

type embedRequest struct {
	Name      string `json:"name"`
	MusicFile string `json:"music_file"`
	CueFile   string `json:"cue_file,omitempty"`
}

func isCueFile(name string) bool {
	return strings.EqualFold(filepath.Ext(name), ".cue")
}

// TODO: Change saveUploadFile function to save to absoluteLibPath with the original file name given.
func saveUploadedFile(header *multipart.FileHeader, dir string) (string, error) {
	src, err := header.Open()
	if err != nil {
		return "", fmt.Errorf("open uploaded file %q: %w", header.Filename, err)
	}
	defer func() { _ = src.Close() }()

	ext := filepath.Ext(header.Filename)
	pattern := "navidrome-upload-*"
	if ext != "" {
		pattern += ext
	}

	tmp, err := os.CreateTemp(dir, pattern)
	if err != nil {
		return "", fmt.Errorf("create temp file for %q: %w", header.Filename, err)
	}

	if _, err := io.Copy(tmp, src); err != nil {
		_ = tmp.Close()
		_ = os.Remove(tmp.Name())
		return "", fmt.Errorf("copy upload contents for %q: %w", header.Filename, err)
	}

	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmp.Name())
		return "", fmt.Errorf("close temp file for %q: %w", header.Filename, err)
	}

	return tmp.Name(), nil
}

func callEmbedServer(musicPath, cuePath string) (map[string]any, error) {
	conn, err := net.Dial("unix", embedSocketPath)
	if err != nil {
		return nil, fmt.Errorf("dial embed server: %w", err)
	}
	defer func() { _ = conn.Close() }()

	reqPayload := embedRequest{MusicFile: musicPath}
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

func (n *Router) addUploadRoute(r chi.Router) {
	r.With(adminOnlyMiddleware).Post("/upload", func(w http.ResponseWriter, req *http.Request) {
		if err := req.ParseMultipartForm(0); err != nil {
			log.Error(req.Context(), "Unable to parse upload payload", err)
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if req.MultipartForm == nil {
			http.Error(w, "no file in upload payload", http.StatusBadRequest)
			return
		}
		defer func() { _ = req.MultipartForm.RemoveAll() }()

		fileHeaders, ok := req.MultipartForm.File["file"]
		if !ok || len(fileHeaders) == 0 {
			http.Error(w, "no file in upload payload", http.StatusBadRequest)
			return
		}

		tempDir, err := os.MkdirTemp("", "navidrome-upload-*")
		if err != nil {
			log.Error(req.Context(), "Failed to create temp directory for upload", err)
			http.Error(w, "failed to process upload", http.StatusInternalServerError)
			return
		}
		defer func() { _ = os.RemoveAll(tempDir) }()

		var musicPath string
		var cuePath string

		for _, header := range fileHeaders {
			savedPath, err := saveUploadedFile(header, tempDir)
			if err != nil {
				log.Error(req.Context(), "Failed to persist uploaded file", "filename", header.Filename, err)
				http.Error(w, "failed to process upload", http.StatusInternalServerError)
				return
			}

			if isCueFile(header.Filename) {
				if cuePath != "" {
					log.Warn(req.Context(), "Multiple cue files provided; ignoring additional file", "filename", header.Filename)
					continue
				}
				cuePath = savedPath
				log.Info(req.Context(), "Stored cue file for embedding", "filename", header.Filename, "path", cuePath)
				continue
			}

			if musicPath != "" {
				log.Error(req.Context(), "Multiple audio files provided in single upload", "filename", header.Filename)
				http.Error(w, "multiple audio files in upload payload", http.StatusBadRequest)
				return
			}

			musicPath = savedPath
			log.Info(req.Context(), "Stored music file for embedding", "filename", header.Filename, "path", musicPath)
		}

		if musicPath == "" {
			http.Error(w, "no audio file in upload payload", http.StatusBadRequest)
			return
		}

		respPayload, err := callEmbedServer(musicPath, cuePath)
		if err != nil {
			log.Error(req.Context(), "Failed to contact embedding server", err)
			http.Error(w, "failed to contact embedding server", http.StatusBadGateway)
			return
		}

		statusCode := http.StatusOK
		if statusValue, ok := respPayload["status"]; ok {
			if statusText, ok := statusValue.(string); ok && !strings.EqualFold(statusText, "ok") {
				statusCode = http.StatusBadGateway
			}
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(statusCode)
		if err := json.NewEncoder(w).Encode(respPayload); err != nil {
			log.Error(req.Context(), "Failed to write embedding server response", err)
		}
	})
}
