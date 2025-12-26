package scanner

import (
	"fmt"
	"path/filepath"
)

type embeddingCandidate struct {
	LibraryID   int
	LibraryPath string
	TrackPath   string
	Artist      string
	Title       string
	Album       string
}

func (c embeddingCandidate) key() string {
	return fmt.Sprintf("%d:%s", c.LibraryID, filepath.Clean(c.TrackPath))
}

func (c embeddingCandidate) absolutePath() string {
	if filepath.IsAbs(c.TrackPath) {
		return filepath.Clean(c.TrackPath)
	}
	path := filepath.Join(c.LibraryPath, c.TrackPath)
	if filepath.IsAbs(c.LibraryPath) {
		return filepath.Clean(path)
	}
	absPath, err := filepath.Abs(path)
	if err != nil {
		return filepath.Clean(path)
	}
	return absPath
}
