package persistence

import (
	"context"
	"errors"
	"time"

	. "github.com/Masterminds/squirrel"
	"github.com/navidrome/navidrome/model"
	"github.com/pocketbase/dbx"
)

type scrobbleRepository struct {
	sqlRepository
}

func NewScrobbleRepository(ctx context.Context, db dbx.Builder) model.ScrobbleRepository {
	r := &scrobbleRepository{}
	r.ctx = ctx
	r.db = db
	r.tableName = "scrobbles"
	return r
}

func (r *scrobbleRepository) RecordScrobble(mediaFileID string, submissionTime time.Time) error {
	userID := loggedUser(r.ctx).ID
	values := map[string]any{
		"media_file_id":   mediaFileID,
		"user_id":         userID,
		"submission_time": submissionTime.Unix(),
	}
	insert := Insert(r.tableName).SetMap(values)
	_, err := r.executeSQL(insert)
	return err
}

func (r *scrobbleRepository) ListByUser(userID string, max int) (model.Scrobbles, error) {
	type row struct {
		MediaFileID    string `db:"media_file_id"`
		UserID         string `db:"user_id"`
		SubmissionTime int64  `db:"submission_time"`
	}

	query := Select("media_file_id", "user_id", "submission_time").
		From(r.tableName).
		Where(Eq{"user_id": userID}).
		OrderBy("submission_time asc")
	if max > 0 {
		query = query.Limit(uint64(max))
	}

	var rows []row
	if err := r.queryAll(query, &rows); err != nil {
		if errors.Is(err, model.ErrNotFound) {
			return model.Scrobbles{}, nil
		}
		return nil, err
	}

	scrobbles := make(model.Scrobbles, 0, len(rows))
	for _, item := range rows {
		scrobbles = append(scrobbles, model.Scrobble{
			MediaFileID:    item.MediaFileID,
			UserID:         item.UserID,
			SubmissionTime: time.Unix(item.SubmissionTime, 0).UTC(),
		})
	}
	return scrobbles, nil
}
