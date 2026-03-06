package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/navidrome/navidrome/db"
	"github.com/navidrome/navidrome/log"
	"github.com/navidrome/navidrome/model"
	"github.com/navidrome/navidrome/model/request"
	"github.com/navidrome/navidrome/persistence"
	"github.com/navidrome/navidrome/recommender/evaluator"
	"github.com/navidrome/navidrome/recommender/milvus"
	recresolver "github.com/navidrome/navidrome/recommender/resolver"
	"github.com/navidrome/navidrome/server/subsonic"
	"github.com/spf13/cobra"
)

var (
	evalFormat             string
	evalUserIDs            []string
	evalEngines            []string
	evalModels             []string
	evalTopK               int
	evalPlaylistSeedWindow int
	evalPlaylistHoldout    int
	evalSessionSeedWindow  int
	evalSessionHoldout     int
	evalSessionGap         time.Duration
	evalMaxPlaylists       int
	evalMaxSessions        int
	evalMergeStrategy      string
	evalDiversity          float64
	evalMinModelAgreement  int
)

func init() {
	recommendationEvalCmd.Flags().StringVarP(&evalFormat, "format", "f", "text", "output format (text or json)")
	recommendationEvalCmd.Flags().StringArrayVar(&evalUserIDs, "user-id", nil, "restrict evaluation to specific user IDs")
	recommendationEvalCmd.Flags().StringArrayVar(&evalEngines, "engine", []string{"current", "legacy"}, "engines to evaluate (current, legacy)")
	recommendationEvalCmd.Flags().StringArrayVar(&evalModels, "model", nil, "restrict evaluation to specific recommendation models")
	recommendationEvalCmd.Flags().IntVar(&evalTopK, "top-k", 20, "number of recommendations to score")
	recommendationEvalCmd.Flags().IntVar(&evalPlaylistSeedWindow, "playlist-seed-window", 10, "number of playlist seed tracks before holdout")
	recommendationEvalCmd.Flags().IntVar(&evalPlaylistHoldout, "playlist-holdout", 1, "number of playlist holdout tracks")
	recommendationEvalCmd.Flags().IntVar(&evalSessionSeedWindow, "session-seed-window", 5, "number of session seed tracks before holdout")
	recommendationEvalCmd.Flags().IntVar(&evalSessionHoldout, "session-holdout", 1, "number of session holdout tracks")
	recommendationEvalCmd.Flags().DurationVar(&evalSessionGap, "session-gap", 30*time.Minute, "maximum gap between scrobbles in the same session")
	recommendationEvalCmd.Flags().IntVar(&evalMaxPlaylists, "max-playlists", 200, "maximum playlists to evaluate")
	recommendationEvalCmd.Flags().IntVar(&evalMaxSessions, "max-sessions", 500, "maximum listening sessions to evaluate")
	recommendationEvalCmd.Flags().StringVar(&evalMergeStrategy, "merge-strategy", "", "override merge strategy")
	recommendationEvalCmd.Flags().Float64Var(&evalDiversity, "diversity", 0, "override diversity for evaluation requests")
	recommendationEvalCmd.Flags().IntVar(&evalMinModelAgreement, "min-model-agreement", 0, "override minimum model agreement")
	rootCmd.AddCommand(recommendationEvalCmd)
}

var recommendationEvalCmd = &cobra.Command{
	Use:   "recommendation-eval",
	Short: "Evaluate recommendation quality offline",
	Long:  "Runs playlist continuation and session continuation evaluation against the recommender.",
	Run: func(cmd *cobra.Command, _ []string) {
		runRecommendationEval(cmd.Context())
	},
}

func runRecommendationEval(ctx context.Context) {
	sqlDB := db.Db()
	defer sqlDB.Close()

	ds := persistence.New(sqlDB)
	milvusClient, cleanup, err := newMilvusClientForRecommendations()
	if err != nil {
		log.Fatal(ctx, "Failed to initialize Milvus for evaluation", err)
	}
	defer cleanup()

	resolver := newResolverForRecommendations(ds)
	clients, err := buildEvaluationClients(milvusClient, resolver, evalEngines)
	if err != nil {
		log.Fatal(ctx, "Failed to build evaluation clients", err)
	}

	runner := evaluator.NewRunner(ds, clients)
	evalCtx := request.WithUser(ctx, model.User{
		ID:       "recommendation-eval",
		UserName: "recommendation-eval",
		IsAdmin:  true,
	})
	report, err := runner.Evaluate(evalCtx, evaluator.Options{
		TopK:               evalTopK,
		PlaylistSeedWindow: evalPlaylistSeedWindow,
		PlaylistHoldout:    evalPlaylistHoldout,
		SessionSeedWindow:  evalSessionSeedWindow,
		SessionHoldout:     evalSessionHoldout,
		SessionGap:         evalSessionGap,
		MaxPlaylists:       evalMaxPlaylists,
		MaxSessions:        evalMaxSessions,
		UserIDs:            append([]string(nil), evalUserIDs...),
		Models:             append([]string(nil), evalModels...),
		MergeStrategy:      strings.TrimSpace(evalMergeStrategy),
		Diversity:          evalDiversity,
		MinModelAgreement:  evalMinModelAgreement,
	})
	if err != nil {
		log.Fatal(ctx, "Recommendation evaluation failed", err)
	}

	switch strings.ToLower(strings.TrimSpace(evalFormat)) {
	case "json":
		encoder := json.NewEncoder(os.Stdout)
		encoder.SetIndent("", "  ")
		if err := encoder.Encode(report); err != nil {
			log.Fatal(ctx, "Failed to encode evaluation report", err)
		}
	default:
		printEvaluationReport(report)
	}
}

func buildEvaluationClients(milvusClient *milvus.Client, resolver *recresolver.Resolver, engineNames []string) (map[string]subsonic.RecommendationClient, error) {
	clients := make(map[string]subsonic.RecommendationClient)
	names := append([]string(nil), engineNames...)
	if len(names) == 0 {
		names = []string{"current", "legacy"}
	}
	for _, rawName := range names {
		name := strings.ToLower(strings.TrimSpace(rawName))
		if name == "" {
			continue
		}
		switch name {
		case "current":
			eng := newEngineForRecommendations(milvusClient, resolver)
			clients[name] = subsonic.NewGoRecommendationClient(eng)
		case "legacy":
			clients[name] = evaluator.NewLegacyClient(milvusClient, resolver)
		default:
			return nil, fmt.Errorf("unknown engine %q", rawName)
		}
	}
	if len(clients) == 0 {
		return nil, fmt.Errorf("no evaluation engines selected")
	}
	return clients, nil
}

func printEvaluationReport(report evaluator.Report) {
	fmt.Printf("Generated: %s\n", report.GeneratedAt.Format(time.RFC3339))
	fmt.Printf("Dataset: users=%d libraryTracks=%d playlistScenarios=%d sessionScenarios=%d\n\n",
		report.Dataset.Users,
		report.Dataset.LibraryTracks,
		report.Dataset.PlaylistScenarios,
		report.Dataset.SessionScenarios,
	)

	engines := append([]evaluator.EngineReport(nil), report.Engines...)
	sort.Slice(engines, func(i, j int) bool { return engines[i].Name < engines[j].Name })
	for _, engineReport := range engines {
		fmt.Printf("%s\n", strings.ToUpper(engineReport.Name))
		printTaskReport("overall", engineReport.Overall)
		printTaskReport("playlist", engineReport.Playlist)
		printTaskReport("session", engineReport.Session)
		fmt.Println()
	}
}

func printTaskReport(label string, report evaluator.TaskReport) {
	fmt.Printf(
		"  %-8s scenarios=%d failures=%d recall@k=%.4f ndcg@k=%.4f artistRepeat=%.4f albumRepeat=%.4f genreJSD=%.4f moodJSD=%.4f novelty=%.4f coverage=%.4f meanMs=%.2f p95Ms=%.2f\n",
		label,
		report.Scenarios,
		report.Failures,
		report.RecallAtK,
		report.NDCGAtK,
		report.ArtistRepeatRate,
		report.AlbumRepeatRate,
		report.GenreCalibrationJSD,
		report.MoodCalibrationJSD,
		report.NoveltyRate,
		report.CatalogCoverage,
		report.MeanLatencyMs,
		report.P95LatencyMs,
	)
}
