package persistence

import (
	"context"
	"time"

	"github.com/navidrome/navidrome/log"
	"github.com/navidrome/navidrome/model"
	"github.com/navidrome/navidrome/model/id"
	"github.com/navidrome/navidrome/model/request"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/pocketbase/dbx"
)

var _ = Describe("ScrobbleRepository", func() {
	var repo model.ScrobbleRepository
	var rawRepo sqlRepository
	var ctx context.Context
	var fileID string
	var userID string

	BeforeEach(func() {
		fileID = id.NewRandom()
		userID = id.NewRandom()
		ctx = request.WithUser(log.NewContext(GinkgoT().Context()), model.User{ID: userID, UserName: "johndoe", IsAdmin: true})
		db := GetDBXBuilder()
		repo = NewScrobbleRepository(ctx, db)

		rawRepo = sqlRepository{
			ctx:       ctx,
			tableName: "scrobbles",
			db:        db,
		}
	})

	AfterEach(func() {
		_, _ = rawRepo.db.Delete("scrobbles", dbx.HashExp{"media_file_id": fileID}).Execute()
		_, _ = rawRepo.db.Delete("media_file", dbx.HashExp{"id": fileID}).Execute()
		_, _ = rawRepo.db.Delete("user", dbx.HashExp{"id": userID}).Execute()
	})

	Describe("RecordScrobble", func() {
		It("records a scrobble event", func() {
			submissionTime := time.Now().UTC()

			// Insert User
			_, err := rawRepo.db.Insert("user", dbx.Params{
				"id":         userID,
				"user_name":  "user",
				"password":   "pw",
				"created_at": time.Now(),
				"updated_at": time.Now(),
			}).Execute()
			Expect(err).ToNot(HaveOccurred())

			// Insert MediaFile
			_, err = rawRepo.db.Insert("media_file", dbx.Params{
				"id":         fileID,
				"path":       "path",
				"created_at": time.Now(),
				"updated_at": time.Now(),
			}).Execute()
			Expect(err).ToNot(HaveOccurred())

			err = repo.RecordScrobble(fileID, submissionTime)
			Expect(err).ToNot(HaveOccurred())

			// Verify insertion
			var scrobble struct {
				MediaFileID    string `db:"media_file_id"`
				UserID         string `db:"user_id"`
				SubmissionTime int64  `db:"submission_time"`
			}
			err = rawRepo.db.Select("*").From("scrobbles").
				Where(dbx.HashExp{"media_file_id": fileID, "user_id": userID}).
				One(&scrobble)
			Expect(err).ToNot(HaveOccurred())
			Expect(scrobble.MediaFileID).To(Equal(fileID))
			Expect(scrobble.UserID).To(Equal(userID))
			Expect(scrobble.SubmissionTime).To(Equal(submissionTime.Unix()))
		})
	})

	Describe("ListByUser", func() {
		It("returns ordered scrobbles for the requested user", func() {
			otherUserID := id.NewRandom()
			firstTime := time.Now().UTC().Add(-2 * time.Hour).Truncate(time.Second)
			secondTime := firstTime.Add(15 * time.Minute)
			thirdTime := firstTime.Add(30 * time.Minute)
			otherTime := firstTime.Add(45 * time.Minute)

			_, err := rawRepo.db.Insert("user", dbx.Params{
				"id":         userID,
				"user_name":  "user",
				"password":   "pw",
				"created_at": time.Now(),
				"updated_at": time.Now(),
			}).Execute()
			Expect(err).ToNot(HaveOccurred())
			_, err = rawRepo.db.Insert("user", dbx.Params{
				"id":         otherUserID,
				"user_name":  "other",
				"password":   "pw",
				"created_at": time.Now(),
				"updated_at": time.Now(),
			}).Execute()
			Expect(err).ToNot(HaveOccurred())

			fileIDs := []string{id.NewRandom(), id.NewRandom(), id.NewRandom(), id.NewRandom()}
			for _, currentFileID := range fileIDs {
				_, err = rawRepo.db.Insert("media_file", dbx.Params{
					"id":         currentFileID,
					"path":       currentFileID,
					"created_at": time.Now(),
					"updated_at": time.Now(),
				}).Execute()
				Expect(err).ToNot(HaveOccurred())
			}

			_, err = rawRepo.db.Insert("scrobbles", dbx.Params{
				"media_file_id":   fileIDs[0],
				"user_id":         userID,
				"submission_time": firstTime.Unix(),
			}).Execute()
			Expect(err).ToNot(HaveOccurred())
			_, err = rawRepo.db.Insert("scrobbles", dbx.Params{
				"media_file_id":   fileIDs[1],
				"user_id":         userID,
				"submission_time": secondTime.Unix(),
			}).Execute()
			Expect(err).ToNot(HaveOccurred())
			_, err = rawRepo.db.Insert("scrobbles", dbx.Params{
				"media_file_id":   fileIDs[2],
				"user_id":         userID,
				"submission_time": thirdTime.Unix(),
			}).Execute()
			Expect(err).ToNot(HaveOccurred())
			_, err = rawRepo.db.Insert("scrobbles", dbx.Params{
				"media_file_id":   fileIDs[3],
				"user_id":         otherUserID,
				"submission_time": otherTime.Unix(),
			}).Execute()
			Expect(err).ToNot(HaveOccurred())

			scrobbles, err := repo.ListByUser(userID, 2)
			Expect(err).ToNot(HaveOccurred())
			Expect(scrobbles).To(HaveLen(2))
			Expect(scrobbles[0].MediaFileID).To(Equal(fileIDs[0]))
			Expect(scrobbles[0].SubmissionTime).To(Equal(firstTime))
			Expect(scrobbles[1].MediaFileID).To(Equal(fileIDs[1]))
			Expect(scrobbles[1].SubmissionTime).To(Equal(secondTime))

			for _, currentFileID := range fileIDs {
				_, _ = rawRepo.db.Delete("scrobbles", dbx.HashExp{"media_file_id": currentFileID}).Execute()
				_, _ = rawRepo.db.Delete("media_file", dbx.HashExp{"id": currentFileID}).Execute()
			}
			_, _ = rawRepo.db.Delete("user", dbx.HashExp{"id": otherUserID}).Execute()
		})
	})
})
