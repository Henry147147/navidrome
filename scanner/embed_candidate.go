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
		return c.TrackPath
	}
	return filepath.Join(c.LibraryPath, c.TrackPath)
}
