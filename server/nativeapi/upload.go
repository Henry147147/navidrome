package nativeapi

import (
	"encoding/json"
	"net/http"

	"github.com/go-chi/chi/v5"
	"github.com/navidrome/navidrome/log"
)

type uploadResponse struct {
	Message string   `json:"message"`
	Files   []string `json:"files"`
}

func (n *Router) addUploadRoute(r chi.Router) {
	r.With(adminOnlyMiddleware).Post("/upload", func(w http.ResponseWriter, req *http.Request) {
		if err := req.ParseMultipartForm(0); err != nil {
			log.Error(req.Context(), "Unable to parse upload payload", err)
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		var filenames []string
		if req.MultipartForm != nil {
			if fileHeaders, ok := req.MultipartForm.File["files"]; ok {
				filenames = make([]string, 0, len(fileHeaders))
				for _, header := range fileHeaders {
					filenames = append(filenames, header.Filename)
				}
			}
		}

		log.Info(req.Context(), "Received stub upload request", "fileCount", len(filenames), "files", filenames)

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(uploadResponse{
			Message: "Upload received",
			Files:   filenames,
		}); err != nil {
			log.Error(req.Context(), "Failed to encode upload response", err)
		}
	})
}
