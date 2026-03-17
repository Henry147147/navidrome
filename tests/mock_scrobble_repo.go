package tests

import (
	"context"
	"time"

	"github.com/navidrome/navidrome/model"
	"github.com/navidrome/navidrome/model/request"
)

type MockScrobbleRepo struct {
	RecordedScrobbles []model.Scrobble
	ctx               context.Context
}

func (m *MockScrobbleRepo) RecordScrobble(fileID string, submissionTime time.Time) error {
	user, _ := request.UserFrom(m.ctx)
	m.RecordedScrobbles = append(m.RecordedScrobbles, model.Scrobble{
		MediaFileID:    fileID,
		UserID:         user.ID,
		SubmissionTime: submissionTime,
	})
	return nil
}

func (m *MockScrobbleRepo) ListByUser(userID string, max int) (model.Scrobbles, error) {
	filtered := make(model.Scrobbles, 0, len(m.RecordedScrobbles))
	for _, scrobble := range m.RecordedScrobbles {
		if scrobble.UserID != userID {
			continue
		}
		filtered = append(filtered, scrobble)
		if max > 0 && len(filtered) >= max {
			break
		}
	}
	return filtered, nil
}
