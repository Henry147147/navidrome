package scanner

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/navidrome/navidrome/conf"
	"github.com/navidrome/navidrome/conf/configtest"
	"github.com/navidrome/navidrome/core"
	"github.com/navidrome/navidrome/db"
	"github.com/navidrome/navidrome/model"
	"github.com/navidrome/navidrome/persistence"
	"github.com/navidrome/navidrome/tests"
)

func TestCallScanUsesEmbedWorkerFactory(t *testing.T) {
	ctx := context.Background()
	t.Cleanup(configtest.SetupConfig())
	conf.Server.DbPath = filepath.Join(t.TempDir(), "callscan.db?_journal_mode=WAL")
	db.Init(ctx)
	t.Cleanup(func() {
		_ = tests.ClearDB()
	})

	ds := persistence.New(db.Db())
	pls := core.NewPlaylists(ds)

	admin := model.User{
		ID:       "admin",
		UserName: "admin",
		IsAdmin:  true,
	}
	if err := ds.User(ctx).Put(&admin); err != nil {
		t.Fatalf("put admin user: %v", err)
	}

	libs, err := ds.Library(ctx).GetAll()
	if err != nil {
		t.Fatalf("get libraries: %v", err)
	}
	if len(libs) == 0 {
		lib := model.Library{ID: 1, Name: "Test Library", Path: t.TempDir()}
		if err := ds.Library(ctx).Put(&lib); err != nil {
			t.Fatalf("put library: %v", err)
		}
	}

	called := false
	cleaned := false
	prevFactory := scanEmbedWorkerFactory
	scanEmbedWorkerFactory = func(ctx context.Context) (*embeddingWorker, func(), error) {
		called = true
		return nil, func() { cleaned = true }, nil
	}
	t.Cleanup(func() { scanEmbedWorkerFactory = prevFactory })

	progress, err := CallScan(ctx, ds, pls, false, nil)
	if err != nil {
		t.Fatalf("CallScan: %v", err)
	}

	for range progress {
		// drain progress channel
	}

	if !called {
		t.Fatalf("expected embed worker factory to be called")
	}
	if !cleaned {
		t.Fatalf("expected embed worker cleanup to be called")
	}
}
