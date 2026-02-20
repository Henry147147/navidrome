// Package engine provides the recommendation engine for generating playlists.
package engine

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"github.com/navidrome/navidrome/log"
	"github.com/navidrome/navidrome/recommender/milvus"
)

// Model identifiers.
const (
	ModelLyrics      = "lyrics"      // Lyrics text embedding
	ModelDescription = "description" // Audio description text embedding
	ModelFlamingo    = "flamingo"    // Flamingo audio embedding
)

// Config holds recommendation engine configuration.
type Config struct {
	DefaultTopK      int
	DefaultModels    []string
	DefaultMerge     string
	DefaultDiversity float64
}

// DefaultConfig returns sensible defaults.
func DefaultConfig() Config {
	return Config{
		DefaultTopK:      75,
		DefaultModels:    []string{ModelLyrics, ModelDescription, ModelFlamingo},
		DefaultMerge:     "union",
		DefaultDiversity: 0.0,
	}
}

// SeedTrack represents a seed track for recommendations.
type SeedTrack struct {
	TrackID     string
	LookupNames []string
	Embedding   []float64
	Embeddings  map[string][]float64
	Weight      float64
}

// RecommendationRequest holds parameters for recommendation.
type RecommendationRequest struct {
	Seeds                 []SeedTrack
	Models                []string
	MergeStrategy         string
	Limit                 int
	ExcludeTrackIDs       []string
	DislikedTrackIDs      []string
	NegativePrompts       []string
	NegativeEmbeddings    map[string][][]float64 // model -> []embedding
	NegativePromptPenalty float64
	Diversity             float64
	ModelPriorities       map[string]int
	MinModelAgreement     int
}

// RecommendationItem represents a single track recommendation.
type RecommendationItem struct {
	TrackID            string
	Score              float64
	Models             []string
	NegativeSimilarity *float64
}

// RecommendationResponse holds recommendation results.
type RecommendationResponse struct {
	Tracks   []RecommendationItem
	Warnings []string
}

// CollectionForModel returns the Milvus collection name for a model.
func CollectionForModel(model string) string {
	switch model {
	case ModelLyrics:
		return milvus.CollectionLyrics
	case ModelDescription:
		return milvus.CollectionDescription
	case ModelFlamingo:
		return milvus.CollectionFlamingo
	default:
		return milvus.CollectionLyrics
	}
}

// Engine implements RecommendationEngine using multi-model similarity search.
type Engine struct {
	config   Config
	milvus   *milvus.Client
	resolver TrackNameResolver
}

// TrackNameResolver maps canonical names to track IDs.
type TrackNameResolver interface {
	ResolveTrackID(ctx context.Context, name string) (string, error)
	ResolveTrackIDs(ctx context.Context, names []string) (map[string]string, error)
}

// New creates a new recommendation Engine.
func New(cfg Config, milvus *milvus.Client, resolver TrackNameResolver) *Engine {
	if len(cfg.DefaultModels) == 0 {
		cfg.DefaultModels = []string{ModelLyrics, ModelDescription, ModelFlamingo}
	}
	if cfg.DefaultMerge == "" {
		cfg.DefaultMerge = "union"
	}
	if cfg.DefaultTopK <= 0 {
		cfg.DefaultTopK = 75
	}

	return &Engine{
		config:   cfg,
		milvus:   milvus,
		resolver: resolver,
	}
}

// Recommend generates track recommendations based on seeds.
func (e *Engine) Recommend(ctx context.Context, req RecommendationRequest) (*RecommendationResponse, error) {
	if len(req.Seeds) == 0 {
		return &RecommendationResponse{
			Warnings: []string{"No seeds provided"},
		}, nil
	}

	// Set defaults
	if len(req.Models) == 0 {
		req.Models = e.config.DefaultModels
	}
	if req.MergeStrategy == "" {
		req.MergeStrategy = e.config.DefaultMerge
	}
	if req.Limit <= 0 {
		req.Limit = 25
	}

	log.Debug(ctx, "Generating recommendations",
		"seeds", len(req.Seeds),
		"models", req.Models,
		"merge", req.MergeStrategy,
		"limit", req.Limit,
	)

	// Resolve seed embeddings
	seedEmbeddings, warnings, err := e.resolveSeedEmbeddings(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("resolve seed embeddings: %w", err)
	}

	if len(seedEmbeddings) == 0 {
		return &RecommendationResponse{
			Warnings: append(warnings, "No embeddings found for seeds"),
		}, nil
	}

	// Build exclude set
	excludeNames := e.buildExcludeSet(req)

	// Perform multi-model search
	candidates, err := e.searchMultiModel(ctx, seedEmbeddings, req, excludeNames)
	if err != nil {
		return nil, fmt.Errorf("multi-model search: %w", err)
	}

	// Apply negative prompt penalties
	if len(req.NegativeEmbeddings) > 0 || len(req.NegativePrompts) > 0 {
		e.applyNegativePenalties(ctx, candidates, req)
	}

	// Sort by score (descending)
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].Score > candidates[j].Score
	})

	// Limit results
	if len(candidates) > req.Limit {
		candidates = candidates[:req.Limit]
	}

	// Resolve track IDs
	tracks, err := e.resolveCandidates(ctx, candidates)
	if err != nil {
		return nil, fmt.Errorf("resolve candidates: %w", err)
	}

	log.Debug(ctx, "Recommendations generated",
		"candidates", len(candidates),
		"tracks", len(tracks),
	)

	return &RecommendationResponse{
		Tracks:   tracks,
		Warnings: warnings,
	}, nil
}

// candidate represents an internal recommendation candidate.
type candidate struct {
	Name               string
	Score              float64
	Scores             []float64 // Individual scores from each model
	Models             []string  // Models that contributed
	NegativeSimilarity *float64
}

// resolveSeedEmbeddings retrieves embeddings for all seeds.
func (e *Engine) resolveSeedEmbeddings(ctx context.Context, req RecommendationRequest) (map[string]map[string][]float64, []string, error) {
	// Map: model -> seed_name -> embedding
	result := make(map[string]map[string][]float64)
	var warnings []string

	for _, model := range req.Models {
		result[model] = make(map[string][]float64)
	}

	for _, seed := range req.Seeds {
		if len(seed.Embeddings) > 0 {
			for model, emb := range seed.Embeddings {
				if len(emb) == 0 {
					continue
				}
				if _, ok := result[model]; !ok {
					continue
				}
				result[model][fmt.Sprintf("direct_%s_%s", model, seed.TrackID)] = emb
			}
			continue
		}

		// If seed has direct embedding, use it
		if len(seed.Embedding) > 0 {
			// Direct embeddings are used for the primary model
			primaryModel := req.Models[0]
			result[primaryModel][fmt.Sprintf("direct_%s_%s", primaryModel, seed.TrackID)] = seed.Embedding
			continue
		}

		lookupNames := seedLookupNames(seed)
		if len(lookupNames) == 0 {
			continue
		}

		// Otherwise, look up by track ID and alternative names.
		for _, model := range req.Models {
			collection := CollectionForModel(model)
			embeddings, err := e.milvus.GetByNames(ctx, collection, lookupNames)
			if err != nil {
				log.Warn(ctx, "Failed to get seed embedding",
					"trackId", seed.TrackID,
					"lookupNames", lookupNames,
					"model", model,
					"error", err,
				)
				continue
			}

			storeKey := seed.TrackID
			for _, lookup := range lookupNames {
				emb, ok := embeddings[lookup]
				if !ok {
					continue
				}
				if storeKey == "" {
					storeKey = lookup
				}
				result[model][storeKey] = emb
				break
			}
		}
	}

	// Check if we found any embeddings
	totalEmbeddings := 0
	for _, modelEmbs := range result {
		totalEmbeddings += len(modelEmbs)
	}
	if totalEmbeddings == 0 {
		warnings = append(warnings, "No embeddings found for any seeds")
	}

	return result, warnings, nil
}

// buildExcludeSet builds a set of track names to exclude from results.
func (e *Engine) buildExcludeSet(req RecommendationRequest) []string {
	excludeSet := make(map[string]bool)

	// Exclude seed tracks
	for _, seed := range req.Seeds {
		for _, lookup := range seedLookupNames(seed) {
			excludeSet[lookup] = true
		}
	}

	// Exclude explicitly excluded tracks
	for _, id := range req.ExcludeTrackIDs {
		excludeSet[id] = true
	}

	// Exclude disliked tracks
	for _, id := range req.DislikedTrackIDs {
		excludeSet[id] = true
	}

	result := make([]string, 0, len(excludeSet))
	for name := range excludeSet {
		result = append(result, name)
	}
	return result
}

func seedLookupNames(seed SeedTrack) []string {
	seen := make(map[string]struct{}, 1+len(seed.LookupNames))
	result := make([]string, 0, 1+len(seed.LookupNames))
	appendLookup := func(value string) {
		v := strings.TrimSpace(value)
		if v == "" {
			return
		}
		if _, ok := seen[v]; ok {
			return
		}
		seen[v] = struct{}{}
		result = append(result, v)
	}

	appendLookup(seed.TrackID)
	for _, name := range seed.LookupNames {
		appendLookup(name)
	}
	return result
}

// resolveCandidates converts internal candidates to recommendation items.
func (e *Engine) resolveCandidates(ctx context.Context, candidates []candidate) ([]RecommendationItem, error) {
	if len(candidates) == 0 {
		return nil, nil
	}

	// Collect names to resolve
	names := make([]string, len(candidates))
	for i, c := range candidates {
		names[i] = c.Name
	}

	// Resolve track IDs
	var nameToID map[string]string
	if e.resolver != nil {
		var err error
		nameToID, err = e.resolver.ResolveTrackIDs(ctx, names)
		if err != nil {
			log.Warn(ctx, "Failed to resolve track IDs", "error", err)
			nameToID = make(map[string]string)
		}
	} else {
		// Without resolver, use names as IDs
		nameToID = make(map[string]string)
		for _, name := range names {
			nameToID[name] = name
		}
	}

	// Build result
	tracks := make([]RecommendationItem, 0, len(candidates))
	for _, c := range candidates {
		trackID := nameToID[c.Name]
		if trackID == "" {
			trackID = c.Name // Fallback to name
		}

		tracks = append(tracks, RecommendationItem{
			TrackID:            trackID,
			Score:              c.Score,
			Models:             c.Models,
			NegativeSimilarity: c.NegativeSimilarity,
		})
	}

	return tracks, nil
}
