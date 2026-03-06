package model

import "time"

type Scrobble struct {
	MediaFileID    string
	UserID         string
	SubmissionTime time.Time
}

type Scrobbles []Scrobble

type ScrobbleRepository interface {
	RecordScrobble(mediaFileID string, submissionTime time.Time) error
	ListByUser(userID string, max int) (Scrobbles, error)
}
