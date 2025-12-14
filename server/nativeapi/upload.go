package nativeapi

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/go-chi/chi/v5"
	"github.com/navidrome/navidrome/conf"
	"github.com/navidrome/navidrome/log"
)

func isCueFile(name string) bool {
	return strings.EqualFold(filepath.Ext(name), ".cue")
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer func() { _ = in.Close() }()

	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}

	out, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}
	defer func() { _ = out.Close() }()

	if _, err := io.Copy(out, in); err != nil {
		return err
	}
	return out.Close()
}

func fileExists(path string) (bool, error) {
	_, err := os.Stat(path)
	if err == nil {
		return true, nil
	}
	if errors.Is(err, os.ErrNotExist) {
		return false, nil
	}
	return false, err
}

func toStringSlice(value any) []string {
	switch v := value.(type) {
	case nil:
		return nil
	case []string:
		return v
	case []any:
		result := make([]string, 0, len(v))
		for _, item := range v {
			if s, ok := item.(string); ok && s != "" {
				result = append(result, s)
			}
		}
		return result
	default:
		if s, ok := value.(string); ok && s != "" {
			return []string{s}
		}
	}
	return nil
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

type embedJobInput struct {
	MusicPath string
	CuePath   string
	MusicName string
	CueName   string
	Settings  map[string]any
	TempDir   string
}

func (n *Router) processEmbedJob(ctx context.Context, input embedJobInput) (map[string]any, error) {
	respPayload, err := getEmbedClient().Embed(input.MusicPath, input.MusicName, input.CuePath, input.Settings)
	if err != nil {
		return nil, fmt.Errorf("contact embedding server: %w", err)
	}

	statusCode := http.StatusOK
	if statusValue, ok := respPayload["status"]; ok {
		if statusText, ok := statusValue.(string); ok && !strings.EqualFold(statusText, "ok") {
			statusCode = http.StatusBadGateway
		}
	}

	duplicates := toStringSlice(respPayload["duplicates"])
	if duplicates == nil {
		duplicates = []string{}
	}
	respPayload["duplicates"] = duplicates

	allDuplicates := false
	if raw, ok := respPayload["allDuplicates"]; ok {
		if flag, ok := raw.(bool); ok {
			allDuplicates = flag
		}
	}
	respPayload["allDuplicates"] = allDuplicates

	type splitFilePayload struct {
		Path        string `json:"path"`
		DestName    string `json:"destName"`
		Title       string `json:"title"`
		Artist      string `json:"artist"`
		Album       string `json:"album"`
		AlbumArtist string `json:"albumArtist"`
	}

	var splitFiles []splitFilePayload
	if raw, ok := respPayload["splitFiles"]; ok {
		if data, err := json.Marshal(raw); err != nil {
			log.Warn(ctx, "Failed to marshal splitFiles payload", err)
		} else if err := json.Unmarshal(data, &splitFiles); err != nil {
			log.Warn(ctx, "Failed to decode splitFiles payload", err)
			splitFiles = nil
		}
	}

	copied := false
	copyConflict := false
	var copyConflicts []string
	copyError := ""

	if statusCode == http.StatusOK {
		musicFolder := strings.TrimSpace(conf.Server.MusicFolder)
		if musicFolder == "" {
			copyError = "music folder is not configured"
		} else if !allDuplicates {
			if len(splitFiles) > 0 {
				destinations := make([]string, len(splitFiles))
				for i, split := range splitFiles {
					destName := strings.TrimSpace(split.DestName)
					if destName == "" {
						destName = filepath.Base(split.Path)
					}
					if destName == "" {
						copyError = fmt.Sprintf("unable to determine destination filename for split track %d", i+1)
						break
					}
					destPath := filepath.Join(musicFolder, destName)
					destinations[i] = destPath
					if exists, err := fileExists(destPath); err != nil {
						copyError = fmt.Sprintf("check split destination: %v", err)
						break
					} else if exists {
						copyConflict = true
						copyConflicts = append(copyConflicts, destName)
					}
				}

				if copyError == "" && copyConflict {
					log.Warn(ctx, "Split track destinations already exist, skipping copy", "conflicts", copyConflicts)
				}

				if copyError == "" && !copyConflict {
					for i, split := range splitFiles {
						if err := copyFile(split.Path, destinations[i]); err != nil {
							copyError = fmt.Sprintf("copy split track: %v", err)
							log.Error(ctx, "Failed to copy split track", err, "source", split.Path, "dest", destinations[i])
							break
						}
					}
					if copyError == "" {
						copied = true
						log.Info(ctx, "Copied split tracks to music folder", "count", len(splitFiles), "folder", musicFolder)
					}
				}
			} else {
				destAudioName := filepath.Base(input.MusicName)
				if destAudioName == "" {
					copyError = "unable to determine destination filename"
				} else {
					destAudioPath := filepath.Join(musicFolder, destAudioName)

					var destCuePath string
					var destCueName string
					if input.CuePath != "" && input.CueName != "" {
						baseName := strings.TrimSuffix(destAudioName, filepath.Ext(destAudioName))
						if baseName == "" {
							baseName = strings.TrimSuffix(filepath.Base(input.MusicName), filepath.Ext(input.MusicName))
						}
						cueExt := filepath.Ext(input.CueName)
						destCueName = baseName + cueExt
						destCuePath = filepath.Join(musicFolder, destCueName)
					}

					conflicts := []string{}
					if exists, err := fileExists(destAudioPath); err != nil {
						copyError = fmt.Sprintf("check audio destination: %v", err)
					} else if exists {
						conflicts = append(conflicts, destAudioName)
					}

					if copyError == "" && destCuePath != "" {
						if exists, err := fileExists(destCuePath); err != nil {
							copyError = fmt.Sprintf("check cue destination: %v", err)
						} else if exists {
							conflicts = append(conflicts, destCueName)
						}
					}

					if copyError == "" && len(conflicts) > 0 {
						copyConflict = true
						copyConflicts = conflicts
						log.Warn(ctx, "Destination already exists, skipping copy", "conflicts", conflicts)
					}

					if copyError == "" && !copyConflict {
						if err := copyFile(input.MusicPath, destAudioPath); err != nil {
							copyError = fmt.Sprintf("copy audio: %v", err)
							log.Error(ctx, "Failed to copy uploaded audio", err, "source", input.MusicPath, "dest", destAudioPath)
						} else {
							if destCuePath != "" && input.CuePath != "" {
								if err := copyFile(input.CuePath, destCuePath); err != nil {
									copyError = fmt.Sprintf("copy cue: %v", err)
									log.Error(ctx, "Failed to copy uploaded cue", err, "source", input.CuePath, "dest", destCuePath)
									if removeErr := os.Remove(destAudioPath); removeErr != nil {
										log.Warn(ctx, "Failed to clean up audio after cue copy error", "path", destAudioPath, "error", removeErr)
									}
								}
							}
							if copyError == "" {
								copied = true
								log.Info(ctx, "Copied uploaded files to music folder", "audio", destAudioName, "folder", musicFolder)
							}
						}
					}
				}
			}
		}
	}

	respPayload["copied"] = copied
	respPayload["copyConflict"] = copyConflict
	if len(copyConflicts) > 0 {
		respPayload["copyConflicts"] = copyConflicts
	}
	if copyError != "" {
		respPayload["copyError"] = copyError
	}

	if input.TempDir != "" {
		_ = os.RemoveAll(input.TempDir)
	}

	if statusCode != http.StatusOK {
		return respPayload, fmt.Errorf("embedding server returned non-ok status")
	}

	return respPayload, nil
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

		var settings map[string]any
		if rawSettings, found := req.MultipartForm.Value["settings"]; found && len(rawSettings) > 0 {
			if strings.TrimSpace(rawSettings[0]) != "" {
				var parsed map[string]any
				if err := json.Unmarshal([]byte(rawSettings[0]), &parsed); err != nil {
					log.Error(req.Context(), "Invalid upload settings payload", err)
					http.Error(w, "invalid settings payload", http.StatusBadRequest)
					return
				}
				settings = parsed
				log.Debug(req.Context(), "Parsed upload settings", "settings", settings)
			}
		}

		tempDir, err := os.MkdirTemp("", "navidrome-upload-*")
		if err != nil {
			log.Error(req.Context(), "Failed to create temp directory for upload", err)
			http.Error(w, "failed to process upload", http.StatusInternalServerError)
			return
		}
		var musicPath string
		var cuePath string
		var musicName string
		var cueName string

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
				cueName = header.Filename
				log.Info(req.Context(), "Stored cue file for embedding", "filename", cueName, "path", cuePath)
				continue
			}

			if musicPath != "" {
				log.Error(req.Context(), "Multiple audio files provided in single upload", "filename", header.Filename)
				http.Error(w, "multiple audio files in upload payload", http.StatusBadRequest)
				return
			}

			musicPath = savedPath
			musicName = header.Filename
			log.Info(req.Context(), "Stored music file for embedding", "filename", header.Filename, "path", musicPath)
		}

		if musicPath == "" {
			http.Error(w, "no audio file in upload payload", http.StatusBadRequest)
			return
		}

		input := embedJobInput{
			MusicPath: musicPath,
			CuePath:   cuePath,
			MusicName: musicName,
			CueName:   cueName,
			Settings:  settings,
			TempDir:   tempDir,
		}

		work := &embedWork{music: musicName}
		work.run = func() error {
			resp, err := n.processEmbedJob(context.Background(), input)
			if err == nil {
				work.result = resp
			}
			return err
		}

		jobID, err := n.embedQueue.Enqueue(work)
		if err != nil {
			log.Error(req.Context(), "Failed to enqueue embed job", err)
			http.Error(w, "embedding queue is full", http.StatusServiceUnavailable)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusAccepted)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"status": "queued",
			"jobId":  jobID,
		})
	})
}

func (n *Router) addUploadStatusRoute(r chi.Router) {
	r.With(adminOnlyMiddleware).Get("/upload/jobs", func(w http.ResponseWriter, req *http.Request) {
		limit := 50
		if raw := strings.TrimSpace(req.URL.Query().Get("limit")); raw != "" {
			if parsed, err := strconv.Atoi(raw); err == nil && parsed > 0 {
				limit = parsed
			}
		}
		jobs := n.embedQueue.List(limit)
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]any{"jobs": jobs}); err != nil {
			log.Error(req.Context(), "Failed to encode jobs response", err)
		}
	})

	r.With(adminOnlyMiddleware).Get("/upload/jobs/{id}", func(w http.ResponseWriter, req *http.Request) {
		jobID := chi.URLParam(req, "id")
		job, ok := n.embedQueue.Get(jobID)
		if !ok {
			http.Error(w, "job not found", http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(job); err != nil {
			log.Error(req.Context(), "Failed to encode job status", err)
		}
	})
}
