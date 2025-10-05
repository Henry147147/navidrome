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
		if req.MultipartForm == nil {
			http.Error(w, "no file in upload payload", http.StatusBadRequest)
			return
		}
		defer func() {
			_ = req.MultipartForm.RemoveAll()
		}()

		fileHeaders, ok := req.MultipartForm.File["file"]
		if !ok || len(fileHeaders) == 0 {
			http.Error(w, "no file in upload payload", http.StatusBadRequest)
			return
		}

		filenames := make([]string, 0, len(fileHeaders))
		for _, header := range fileHeaders {
			filenames = append(filenames, header.Filename)
			log.Info(req.Context(), "Received stub upload file", "filename", header.Filename, "size", header.Size)
		}
		log.Info(req.Context(), "Received upload request", "fileCount", len(fileHeaders))

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(uploadResponse{
			Message: "Upload received",
			Files:   filenames,
		}); err != nil {
			log.Error(req.Context(), "Failed to encode upload response", err)
		}
	})
}
