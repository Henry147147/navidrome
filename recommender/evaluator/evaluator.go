package evaluator

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	sq "github.com/Masterminds/squirrel"
	"github.com/navidrome/navidrome/model"
	"github.com/navidrome/navidrome/server/subsonic"
)

const recommendationModeCustom = "custom"

type Options struct {
	TopK               int
	PlaylistSeedWindow int
	PlaylistHoldout    int
	SessionSeedWindow  int
	SessionHoldout     int
	SessionGap         time.Duration
	MaxPlaylists       int
	MaxSessions        int
	UserIDs            []string
	Models             []string
	MergeStrategy      string
	Diversity          float64
	ModelPriorities    map[string]int
	MinModelAgreement  int
}

type Runner struct {
	ds      model.DataStore
	engines map[string]subsonic.RecommendationClient
}

type Report struct {
	GeneratedAt time.Time      `json:"generatedAt"`
	Dataset     DatasetReport  `json:"dataset"`
	Engines     []EngineReport `json:"engines"`
}

type DatasetReport struct {
	Users             int `json:"users"`
	LibraryTracks     int `json:"libraryTracks"`
	PlaylistScenarios int `json:"playlistScenarios"`
	SessionScenarios  int `json:"sessionScenarios"`
}

type EngineReport struct {
	Name     string     `json:"name"`
	Playlist TaskReport `json:"playlist"`
	Session  TaskReport `json:"session"`
	Overall  TaskReport `json:"overall"`
}

type TaskReport struct {
	Scenarios           int     `json:"scenarios"`
	Failures            int     `json:"failures"`
	EmptyResults        int     `json:"emptyResults"`
	FallbackResults     int     `json:"fallbackResults"`
	UnresolvedResults   int     `json:"unresolvedResults"`
	WarningOnlyResults  int     `json:"warningOnlyResults"`
	RecallAtK           float64 `json:"recallAtK"`
	NDCGAtK             float64 `json:"ndcgAtK"`
	ArtistRepeatRate    float64 `json:"artistRepeatRate"`
	AlbumRepeatRate     float64 `json:"albumRepeatRate"`
	GenreCalibrationJSD float64 `json:"genreCalibrationJsd"`
	MoodCalibrationJSD  float64 `json:"moodCalibrationJsd"`
	NoveltyRate         float64 `json:"noveltyRate"`
	CatalogCoverage     float64 `json:"catalogCoverage"`
	EmptyResultRate     float64 `json:"emptyResultRate"`
	FallbackRate        float64 `json:"fallbackRate"`
	UnresolvedTrackRate float64 `json:"unresolvedTrackRate"`
	SemanticCoverage    float64 `json:"semanticCoverage"`
	WarningsPerScenario float64 `json:"warningsPerScenario"`
	MeanLatencyMs       float64 `json:"meanLatencyMs"`
	P95LatencyMs        float64 `json:"p95LatencyMs"`
}

type Scenario struct {
	ID       string
	Task     string
	User     model.User
	Name     string
	Seeds    model.MediaFiles
	Relevant model.MediaFiles
}

func DefaultOptions() Options {
	return Options{
		TopK:               20,
		PlaylistSeedWindow: 10,
		PlaylistHoldout:    1,
		SessionSeedWindow:  5,
		SessionHoldout:     1,
		SessionGap:         30 * time.Minute,
		MaxPlaylists:       200,
		MaxSessions:        500,
	}
}

func NewRunner(ds model.DataStore, engines map[string]subsonic.RecommendationClient) *Runner {
	return &Runner{ds: ds, engines: engines}
}

func (r *Runner) Evaluate(ctx context.Context, opts Options) (Report, error) {
	opts = normalizeOptions(opts)
	users, err := r.loadUsers(ctx, opts.UserIDs)
	if err != nil {
		return Report{}, err
	}

	userByID := make(map[string]model.User, len(users))
	for _, user := range users {
		userByID[user.ID] = user
	}

	libraryTracks, err := r.ds.MediaFile(ctx).CountAll()
	if err != nil {
		return Report{}, err
	}

	playlistScenarios, err := r.buildPlaylistScenarios(ctx, userByID, opts)
	if err != nil {
		return Report{}, err
	}
	sessionScenarios, err := r.buildSessionScenarios(ctx, users, opts)
	if err != nil {
		return Report{}, err
	}

	report := Report{
		GeneratedAt: time.Now().UTC(),
		Dataset: DatasetReport{
			Users:             len(users),
			LibraryTracks:     int(libraryTracks),
			PlaylistScenarios: len(playlistScenarios),
			SessionScenarios:  len(sessionScenarios),
		},
		Engines: make([]EngineReport, 0, len(r.engines)),
	}

	engineNames := make([]string, 0, len(r.engines))
	for name := range r.engines {
		engineNames = append(engineNames, name)
	}
	sort.Strings(engineNames)

	for _, name := range engineNames {
		client := r.engines[name]
		playlistReport, err := r.evaluateScenarios(ctx, client, playlistScenarios, opts.TopK, int(libraryTracks), opts)
		if err != nil {
			return Report{}, fmt.Errorf("%s playlist evaluation failed: %w", name, err)
		}
		sessionReport, err := r.evaluateScenarios(ctx, client, sessionScenarios, opts.TopK, int(libraryTracks), opts)
		if err != nil {
			return Report{}, fmt.Errorf("%s session evaluation failed: %w", name, err)
		}
		report.Engines = append(report.Engines, EngineReport{
			Name:     name,
			Playlist: playlistReport,
			Session:  sessionReport,
			Overall:  combineTaskReports(playlistReport, sessionReport),
		})
	}

	return report, nil
}

func (r *Runner) loadUsers(ctx context.Context, requested []string) (model.Users, error) {
	if len(requested) == 0 {
		users, err := r.ds.User(ctx).GetAll()
		if err != nil {
			return nil, err
		}
		return users, nil
	}

	users := make(model.Users, 0, len(requested))
	for _, id := range requested {
		user, err := r.ds.User(ctx).Get(strings.TrimSpace(id))
		if err != nil {
			return nil, err
		}
		users = append(users, *user)
	}
	return users, nil
}

func (r *Runner) buildPlaylistScenarios(ctx context.Context, userByID map[string]model.User, opts Options) ([]Scenario, error) {
	playlists, err := r.ds.Playlist(ctx).GetAll(model.QueryOptions{
		Sort:  "updated_at",
		Order: "desc",
		Max:   opts.MaxPlaylists,
	})
	if err != nil {
		if errors.Is(err, model.ErrNotFound) {
			return nil, nil
		}
		return nil, err
	}

	scenarios := make([]Scenario, 0, len(playlists))
	for _, playlist := range playlists {
		user, ok := userByID[playlist.OwnerID]
		if !ok {
			continue
		}

		tracks, err := r.ds.Playlist(ctx).Tracks(playlist.ID, false).GetAll(model.QueryOptions{})
		if err != nil {
			if errors.Is(err, model.ErrNotFound) {
				continue
			}
			return nil, err
		}
		sequence := compactTrackRuns(tracks.MediaFiles())
		if len(sequence) < opts.PlaylistSeedWindow+opts.PlaylistHoldout {
			continue
		}

		holdoutStart := len(sequence) - opts.PlaylistHoldout
		seedStart := maxInt(0, holdoutStart-opts.PlaylistSeedWindow)
		seeds := dedupeTrackSequence(sequence[seedStart:holdoutStart])
		relevant := dedupeTrackSequence(sequence[holdoutStart:])
		if len(seeds) == 0 || len(relevant) == 0 {
			continue
		}

		scenarios = append(scenarios, Scenario{
			ID:       playlist.ID,
			Task:     "playlist",
			User:     user,
			Name:     playlist.Name,
			Seeds:    seeds,
			Relevant: relevant,
		})
	}
	return scenarios, nil
}

func (r *Runner) buildSessionScenarios(ctx context.Context, users model.Users, opts Options) ([]Scenario, error) {
	scenarios := make([]Scenario, 0, opts.MaxSessions)
	for _, user := range users {
		scrobbles, err := r.ds.Scrobble(ctx).ListByUser(user.ID, 0)
		if err != nil {
			return nil, err
		}
		if len(scrobbles) < opts.SessionSeedWindow+opts.SessionHoldout {
			continue
		}

		trackMap, err := r.loadTrackMap(ctx, scrobbleTrackIDs(scrobbles))
		if err != nil {
			return nil, err
		}

		sessions := splitSessions(scrobbles, trackMap, opts.SessionGap)
		for idx, session := range sessions {
			if len(session) < opts.SessionSeedWindow+opts.SessionHoldout {
				continue
			}
			holdoutStart := len(session) - opts.SessionHoldout
			seedStart := maxInt(0, holdoutStart-opts.SessionSeedWindow)
			seeds := dedupeTrackSequence(session[seedStart:holdoutStart])
			relevant := dedupeTrackSequence(session[holdoutStart:])
			if len(seeds) == 0 || len(relevant) == 0 {
				continue
			}

			scenarios = append(scenarios, Scenario{
				ID:       fmt.Sprintf("%s-session-%d", user.ID, idx),
				Task:     "session",
				User:     user,
				Name:     fmt.Sprintf("%s session %d", user.UserName, idx+1),
				Seeds:    seeds,
				Relevant: relevant,
			})
			if opts.MaxSessions > 0 && len(scenarios) >= opts.MaxSessions {
				return scenarios, nil
			}
		}
	}
	return scenarios, nil
}

func (r *Runner) evaluateScenarios(ctx context.Context, client subsonic.RecommendationClient, scenarios []Scenario, topK int, libraryTracks int, opts Options) (TaskReport, error) {
	acc := taskAccumulator{
		coverageIDs: make(map[string]struct{}),
		latencies:   make([]time.Duration, 0, len(scenarios)),
	}

	for _, scenario := range scenarios {
		request := buildRecommendationRequest(scenario, opts, topK)
		start := time.Now()
		response, err := client.Recommend(ctx, recommendationModeCustom, request)
		latency := time.Since(start)
		acc.latencies = append(acc.latencies, latency)
		acc.scenarios++

		if err != nil {
			acc.failures++
			continue
		}

		warningCount := len(response.Warnings)
		acc.warningCount += warningCount
		if warningCount > 0 {
			acc.fallbackResults++
		}

		responseTrackIDs := trimUniqueIDs(response.TrackIDs(), topK)
		if len(responseTrackIDs) == 0 {
			acc.emptyResults++
			acc.failures++
			if warningCount > 0 {
				acc.warningOnlyResults++
			}
			continue
		}

		recommendations, err := r.loadTracksOrdered(ctx, responseTrackIDs)
		if err != nil {
			return TaskReport{}, err
		}
		if len(recommendations) == 0 {
			acc.unresolvedResults++
			acc.failures++
			if warningCount > 0 {
				acc.warningOnlyResults++
			}
			continue
		}
		if len(recommendations) < len(responseTrackIDs) {
			acc.unresolvedResults++
		}
		acc.semanticSuccesses++

		relevantIDs := trackIDSet(scenario.Relevant)
		recommendedIDs := mediaFileIDs(recommendations)
		acc.recallSum += recallAtK(recommendedIDs, relevantIDs)
		acc.ndcgSum += ndcgAtK(recommendedIDs, relevantIDs)
		acc.artistRepeatSum += repeatRate(recommendations, artistKey)
		acc.albumRepeatSum += repeatRate(recommendations, albumKey)
		acc.genreCalibrationSum += featureDivergence(scenario.Seeds, recommendations, genreValues)
		acc.moodCalibrationSum += featureDivergence(scenario.Seeds, recommendations, moodValues)
		acc.noveltySum += noveltyRate(scenario.Seeds, recommendations)

		for _, id := range recommendedIDs {
			acc.coverageIDs[id] = struct{}{}
		}
	}

	if acc.scenarios > 0 && acc.semanticSuccesses == 0 {
		return TaskReport{}, fmt.Errorf("all %d scenarios produced empty, unresolved, or degraded recommendation results", acc.scenarios)
	}

	return acc.finalize(libraryTracks), nil
}

func buildRecommendationRequest(scenario Scenario, opts Options, topK int) subsonic.RecommendationRequest {
	return subsonic.RecommendationRequest{
		UserID:            scenario.User.ID,
		UserName:          scenario.User.UserName,
		Mode:              recommendationModeCustom,
		Limit:             topK,
		Seeds:             seedsFromMediaFiles(scenario.Seeds, scenario.Task),
		Models:            append([]string(nil), opts.Models...),
		MergeStrategy:     opts.MergeStrategy,
		Diversity:         opts.Diversity,
		ModelPriorities:   clonePriorities(opts.ModelPriorities),
		MinModelAgreement: opts.MinModelAgreement,
	}
}

func seedsFromMediaFiles(files model.MediaFiles, source string) []subsonic.RecommendationSeed {
	seeds := make([]subsonic.RecommendationSeed, 0, len(files))
	for idx, mf := range files {
		if strings.TrimSpace(mf.ID) == "" {
			continue
		}
		weight := 1.0 / float64(idx+1)
		if weight < 0.1 {
			weight = 0.1
		}
		seeds = append(seeds, subsonic.RecommendationSeed{
			TrackID:     mf.ID,
			LookupNames: trackLookupNames(mf),
			Weight:      weight,
			Source:      source,
			PlayedAt:    mf.PlayDate,
		})
	}
	return seeds
}

func trackLookupNames(mf model.MediaFile) []string {
	seen := make(map[string]struct{}, 3)
	names := make([]string, 0, 3)
	add := func(value string) {
		value = strings.TrimSpace(value)
		if value == "" {
			return
		}
		if _, ok := seen[value]; ok {
			return
		}
		seen[value] = struct{}{}
		names = append(names, value)
	}
	add(canonicalTrackName(mf.Artist, mf.Title, mf.Path))
	add(mf.Title)
	add(mf.Path)
	return names
}

func canonicalTrackName(artist string, title string, path string) string {
	artist = strings.TrimSpace(artist)
	title = strings.TrimSpace(title)
	path = strings.TrimSpace(path)
	switch {
	case artist != "" && title != "":
		return artist + " - " + title
	case title != "":
		return title
	default:
		return path
	}
}

func (r *Runner) loadTrackMap(ctx context.Context, ids []string) (map[string]model.MediaFile, error) {
	tracks, err := r.loadTracksOrdered(ctx, ids)
	if err != nil {
		return nil, err
	}
	trackMap := make(map[string]model.MediaFile, len(tracks))
	for _, track := range tracks {
		trackMap[track.ID] = track
	}
	return trackMap, nil
}

func (r *Runner) loadTracksOrdered(ctx context.Context, ids []string) (model.MediaFiles, error) {
	ids = trimUniqueIDs(ids, 0)
	if len(ids) == 0 {
		return nil, nil
	}
	tracks, err := r.ds.MediaFile(ctx).GetAll(model.QueryOptions{
		Filters: sq.Eq{"media_file.id": ids},
		Max:     len(ids),
	})
	if err != nil {
		return nil, err
	}
	trackMap := make(map[string]model.MediaFile, len(tracks))
	for _, track := range tracks {
		trackMap[track.ID] = track
	}
	ordered := make(model.MediaFiles, 0, len(ids))
	for _, id := range ids {
		if track, ok := trackMap[id]; ok {
			ordered = append(ordered, track)
		}
	}
	return ordered, nil
}

type taskAccumulator struct {
	scenarios           int
	failures            int
	emptyResults        int
	fallbackResults     int
	unresolvedResults   int
	warningOnlyResults  int
	semanticSuccesses   int
	warningCount        int
	recallSum           float64
	ndcgSum             float64
	artistRepeatSum     float64
	albumRepeatSum      float64
	genreCalibrationSum float64
	moodCalibrationSum  float64
	noveltySum          float64
	coverageIDs         map[string]struct{}
	latencies           []time.Duration
}

func (a taskAccumulator) finalize(libraryTracks int) TaskReport {
	report := TaskReport{
		Scenarios:          a.scenarios,
		Failures:           a.failures,
		EmptyResults:       a.emptyResults,
		FallbackResults:    a.fallbackResults,
		UnresolvedResults:  a.unresolvedResults,
		WarningOnlyResults: a.warningOnlyResults,
	}
	if a.scenarios == 0 {
		return report
	}

	divisor := float64(a.scenarios)
	report.RecallAtK = a.recallSum / divisor
	report.NDCGAtK = a.ndcgSum / divisor
	report.ArtistRepeatRate = a.artistRepeatSum / divisor
	report.AlbumRepeatRate = a.albumRepeatSum / divisor
	report.GenreCalibrationJSD = a.genreCalibrationSum / divisor
	report.MoodCalibrationJSD = a.moodCalibrationSum / divisor
	report.NoveltyRate = a.noveltySum / divisor
	if libraryTracks > 0 {
		report.CatalogCoverage = float64(len(a.coverageIDs)) / float64(libraryTracks)
	}
	report.EmptyResultRate = float64(a.emptyResults) / divisor
	report.FallbackRate = float64(a.fallbackResults) / divisor
	report.UnresolvedTrackRate = float64(a.unresolvedResults) / divisor
	report.SemanticCoverage = float64(a.semanticSuccesses) / divisor
	report.WarningsPerScenario = float64(a.warningCount) / divisor
	report.MeanLatencyMs = meanLatencyMs(a.latencies)
	report.P95LatencyMs = percentileLatencyMs(a.latencies, 0.95)
	return report
}

func combineTaskReports(left TaskReport, right TaskReport) TaskReport {
	total := left.Scenarios + right.Scenarios
	if total == 0 {
		return TaskReport{}
	}

	weight := func(valueLeft, valueRight float64) float64 {
		return ((valueLeft * float64(left.Scenarios)) + (valueRight * float64(right.Scenarios))) / float64(total)
	}

	return TaskReport{
		Scenarios:           total,
		Failures:            left.Failures + right.Failures,
		EmptyResults:        left.EmptyResults + right.EmptyResults,
		FallbackResults:     left.FallbackResults + right.FallbackResults,
		UnresolvedResults:   left.UnresolvedResults + right.UnresolvedResults,
		WarningOnlyResults:  left.WarningOnlyResults + right.WarningOnlyResults,
		RecallAtK:           weight(left.RecallAtK, right.RecallAtK),
		NDCGAtK:             weight(left.NDCGAtK, right.NDCGAtK),
		ArtistRepeatRate:    weight(left.ArtistRepeatRate, right.ArtistRepeatRate),
		AlbumRepeatRate:     weight(left.AlbumRepeatRate, right.AlbumRepeatRate),
		GenreCalibrationJSD: weight(left.GenreCalibrationJSD, right.GenreCalibrationJSD),
		MoodCalibrationJSD:  weight(left.MoodCalibrationJSD, right.MoodCalibrationJSD),
		NoveltyRate:         weight(left.NoveltyRate, right.NoveltyRate),
		CatalogCoverage:     maxFloat(left.CatalogCoverage, right.CatalogCoverage),
		EmptyResultRate:     weight(left.EmptyResultRate, right.EmptyResultRate),
		FallbackRate:        weight(left.FallbackRate, right.FallbackRate),
		UnresolvedTrackRate: weight(left.UnresolvedTrackRate, right.UnresolvedTrackRate),
		SemanticCoverage:    weight(left.SemanticCoverage, right.SemanticCoverage),
		WarningsPerScenario: weight(left.WarningsPerScenario, right.WarningsPerScenario),
		MeanLatencyMs:       weight(left.MeanLatencyMs, right.MeanLatencyMs),
		P95LatencyMs:        maxFloat(left.P95LatencyMs, right.P95LatencyMs),
	}
}

func normalizeOptions(opts Options) Options {
	defaults := DefaultOptions()
	if opts.TopK <= 0 {
		opts.TopK = defaults.TopK
	}
	if opts.PlaylistSeedWindow <= 0 {
		opts.PlaylistSeedWindow = defaults.PlaylistSeedWindow
	}
	if opts.PlaylistHoldout <= 0 {
		opts.PlaylistHoldout = defaults.PlaylistHoldout
	}
	if opts.SessionSeedWindow <= 0 {
		opts.SessionSeedWindow = defaults.SessionSeedWindow
	}
	if opts.SessionHoldout <= 0 {
		opts.SessionHoldout = defaults.SessionHoldout
	}
	if opts.SessionGap <= 0 {
		opts.SessionGap = defaults.SessionGap
	}
	if opts.MaxPlaylists <= 0 {
		opts.MaxPlaylists = defaults.MaxPlaylists
	}
	if opts.MaxSessions <= 0 {
		opts.MaxSessions = defaults.MaxSessions
	}
	return opts
}

func scrobbleTrackIDs(scrobbles model.Scrobbles) []string {
	ids := make([]string, 0, len(scrobbles))
	for _, scrobble := range scrobbles {
		if id := strings.TrimSpace(scrobble.MediaFileID); id != "" {
			ids = append(ids, id)
		}
	}
	return trimUniqueIDs(ids, 0)
}

func splitSessions(scrobbles model.Scrobbles, trackMap map[string]model.MediaFile, gap time.Duration) []model.MediaFiles {
	if len(scrobbles) == 0 {
		return nil
	}
	sessions := make([]model.MediaFiles, 0)
	current := make(model.MediaFiles, 0)
	last := scrobbles[0].SubmissionTime
	for _, scrobble := range scrobbles {
		track, ok := trackMap[scrobble.MediaFileID]
		if !ok {
			continue
		}
		if len(current) > 0 && scrobble.SubmissionTime.Sub(last) > gap {
			sessions = append(sessions, compactTrackRuns(current))
			current = make(model.MediaFiles, 0)
		}
		current = append(current, track)
		last = scrobble.SubmissionTime
	}
	if len(current) > 0 {
		sessions = append(sessions, compactTrackRuns(current))
	}
	return sessions
}

func compactTrackRuns(tracks model.MediaFiles) model.MediaFiles {
	if len(tracks) == 0 {
		return nil
	}
	compacted := make(model.MediaFiles, 0, len(tracks))
	var lastID string
	for _, track := range tracks {
		if track.ID == "" {
			continue
		}
		if track.ID == lastID {
			continue
		}
		compacted = append(compacted, track)
		lastID = track.ID
	}
	return compacted
}

func dedupeTrackSequence(tracks model.MediaFiles) model.MediaFiles {
	seen := make(map[string]struct{}, len(tracks))
	deduped := make(model.MediaFiles, 0, len(tracks))
	for _, track := range tracks {
		if track.ID == "" {
			continue
		}
		if _, ok := seen[track.ID]; ok {
			continue
		}
		seen[track.ID] = struct{}{}
		deduped = append(deduped, track)
	}
	return deduped
}

func trackIDSet(tracks model.MediaFiles) map[string]struct{} {
	set := make(map[string]struct{}, len(tracks))
	for _, track := range tracks {
		if track.ID == "" {
			continue
		}
		set[track.ID] = struct{}{}
	}
	return set
}

func mediaFileIDs(tracks model.MediaFiles) []string {
	ids := make([]string, 0, len(tracks))
	for _, track := range tracks {
		if track.ID == "" {
			continue
		}
		ids = append(ids, track.ID)
	}
	return ids
}

func trimUniqueIDs(ids []string, max int) []string {
	seen := make(map[string]struct{}, len(ids))
	result := make([]string, 0, len(ids))
	for _, id := range ids {
		id = strings.TrimSpace(id)
		if id == "" {
			continue
		}
		if _, ok := seen[id]; ok {
			continue
		}
		seen[id] = struct{}{}
		result = append(result, id)
		if max > 0 && len(result) >= max {
			break
		}
	}
	return result
}

func recallAtK(recommendedIDs []string, relevant map[string]struct{}) float64 {
	if len(relevant) == 0 {
		return 0
	}
	hits := 0
	for _, id := range recommendedIDs {
		if _, ok := relevant[id]; ok {
			hits++
		}
	}
	return float64(hits) / float64(len(relevant))
}

func ndcgAtK(recommendedIDs []string, relevant map[string]struct{}) float64 {
	if len(relevant) == 0 {
		return 0
	}
	dcg := 0.0
	for idx, id := range recommendedIDs {
		if _, ok := relevant[id]; !ok {
			continue
		}
		dcg += 1.0 / math.Log2(float64(idx)+2)
	}
	ideal := minInt(len(relevant), len(recommendedIDs))
	if ideal == 0 {
		return 0
	}
	idcg := 0.0
	for idx := 0; idx < ideal; idx++ {
		idcg += 1.0 / math.Log2(float64(idx)+2)
	}
	if idcg == 0 {
		return 0
	}
	return dcg / idcg
}

func repeatRate(tracks model.MediaFiles, keyFn func(model.MediaFile) string) float64 {
	if len(tracks) == 0 {
		return 0
	}
	seen := make(map[string]struct{}, len(tracks))
	repeats := 0
	for _, track := range tracks {
		key := keyFn(track)
		if key == "" {
			continue
		}
		if _, ok := seen[key]; ok {
			repeats++
			continue
		}
		seen[key] = struct{}{}
	}
	return float64(repeats) / float64(len(tracks))
}

func featureDivergence(seedTracks model.MediaFiles, recommended model.MediaFiles, valuesFn func(model.MediaFile) []string) float64 {
	return jensenShannonDivergence(distribution(seedTracks, valuesFn), distribution(recommended, valuesFn))
}

func noveltyRate(seedTracks model.MediaFiles, recommended model.MediaFiles) float64 {
	if len(recommended) == 0 {
		return 0
	}
	seedArtists := make(map[string]struct{}, len(seedTracks))
	for _, track := range seedTracks {
		if key := artistKey(track); key != "" {
			seedArtists[key] = struct{}{}
		}
	}
	if len(seedArtists) == 0 {
		return 0
	}
	novel := 0
	for _, track := range recommended {
		if key := artistKey(track); key != "" {
			if _, ok := seedArtists[key]; !ok {
				novel++
			}
		}
	}
	return float64(novel) / float64(len(recommended))
}

func distribution(tracks model.MediaFiles, valuesFn func(model.MediaFile) []string) map[string]float64 {
	counts := make(map[string]float64)
	for _, track := range tracks {
		for _, value := range valuesFn(track) {
			if value == "" {
				continue
			}
			counts[value]++
		}
	}
	total := 0.0
	for _, count := range counts {
		total += count
	}
	if total == 0 {
		return nil
	}
	for key, count := range counts {
		counts[key] = count / total
	}
	return counts
}

func genreValues(track model.MediaFile) []string {
	genres := track.Tags.Values(model.TagGenre)
	if len(genres) == 0 && strings.TrimSpace(track.Genre) != "" {
		genres = []string{track.Genre}
	}
	return normalizeValues("genre:", genres)
}

func moodValues(track model.MediaFile) []string {
	return normalizeValues("mood:", track.Tags.Values(model.TagMood))
}

func normalizeValues(prefix string, values []string) []string {
	seen := make(map[string]struct{}, len(values))
	normalized := make([]string, 0, len(values))
	for _, value := range values {
		value = strings.ToLower(strings.TrimSpace(value))
		if value == "" {
			continue
		}
		key := prefix + value
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		normalized = append(normalized, key)
	}
	return normalized
}

func artistKey(track model.MediaFile) string {
	switch {
	case strings.TrimSpace(track.ArtistID) != "":
		return "artist:" + strings.ToLower(strings.TrimSpace(track.ArtistID))
	case strings.TrimSpace(track.Artist) != "":
		return "artist:" + strings.ToLower(strings.TrimSpace(track.Artist))
	default:
		return ""
	}
}

func albumKey(track model.MediaFile) string {
	switch {
	case strings.TrimSpace(track.AlbumID) != "":
		return "album:" + strings.ToLower(strings.TrimSpace(track.AlbumID))
	case strings.TrimSpace(track.Album) != "":
		return "album:" + strings.ToLower(strings.TrimSpace(track.Album))
	default:
		return ""
	}
}

func jensenShannonDivergence(left map[string]float64, right map[string]float64) float64 {
	if len(left) == 0 && len(right) == 0 {
		return 0
	}
	keys := make(map[string]struct{}, len(left)+len(right))
	for key := range left {
		keys[key] = struct{}{}
	}
	for key := range right {
		keys[key] = struct{}{}
	}

	midpoint := make(map[string]float64, len(keys))
	for key := range keys {
		midpoint[key] = 0.5 * (left[key] + right[key])
	}
	return 0.5 * (klDivergence(left, midpoint) + klDivergence(right, midpoint)) / math.Log(2)
}

func klDivergence(left map[string]float64, midpoint map[string]float64) float64 {
	sum := 0.0
	for key, value := range left {
		if value <= 0 {
			continue
		}
		mid := midpoint[key]
		if mid <= 0 {
			continue
		}
		sum += value * math.Log(value/mid)
	}
	return sum
}

func meanLatencyMs(latencies []time.Duration) float64 {
	if len(latencies) == 0 {
		return 0
	}
	total := 0.0
	for _, latency := range latencies {
		total += float64(latency.Milliseconds())
	}
	return total / float64(len(latencies))
}

func percentileLatencyMs(latencies []time.Duration, percentile float64) float64 {
	if len(latencies) == 0 {
		return 0
	}
	sorted := append([]time.Duration(nil), latencies...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	index := int(math.Ceil(percentile*float64(len(sorted)))) - 1
	if index < 0 {
		index = 0
	}
	if index >= len(sorted) {
		index = len(sorted) - 1
	}
	return float64(sorted[index].Milliseconds())
}

func clonePriorities(priorities map[string]int) map[string]int {
	if len(priorities) == 0 {
		return nil
	}
	cloned := make(map[string]int, len(priorities))
	for key, value := range priorities {
		cloned[key] = value
	}
	return cloned
}

func minInt(a int, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt(a int, b int) int {
	if a > b {
		return a
	}
	return b
}

func maxFloat(a float64, b float64) float64 {
	if a > b {
		return a
	}
	return b
}
