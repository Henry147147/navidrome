package evaluator

import (
	"context"
	"sort"
	"strings"

	"github.com/navidrome/navidrome/recommender/engine"
	"github.com/navidrome/navidrome/recommender/milvus"
	"github.com/navidrome/navidrome/server/subsonic"
)

type legacyClient struct {
	milvus   *milvus.Client
	resolver engine.TrackNameResolver
}

type legacyCandidate struct {
	Name   string
	Score  float64
	Scores []float64
	Models []string
}

func NewLegacyClient(milvusClient *milvus.Client, resolver engine.TrackNameResolver) subsonic.RecommendationClient {
	return &legacyClient{milvus: milvusClient, resolver: resolver}
}

func (c *legacyClient) Recommend(ctx context.Context, _ string, payload subsonic.RecommendationRequest) (*subsonic.RecommendationResponse, error) {
	if len(payload.Seeds) == 0 {
		return &subsonic.RecommendationResponse{}, nil
	}

	models := payload.Models
	if len(models) == 0 {
		models = []string{engine.ModelLyrics, engine.ModelDescription, engine.ModelFlamingo}
	}
	mergeStrategy := strings.TrimSpace(payload.MergeStrategy)
	if mergeStrategy == "" {
		mergeStrategy = "union"
	}
	limit := payload.Limit
	if limit <= 0 {
		limit = 25
	}

	seedEmbeddings, err := c.resolveSeedEmbeddings(ctx, payload.Seeds, models)
	if err != nil {
		return nil, err
	}
	excludeNames := legacyExcludeSet(payload)
	modelResults := make(map[string][]legacyCandidate)

	for _, model := range models {
		vectors := seedEmbeddings[model]
		if len(vectors) == 0 {
			continue
		}
		topK := 75
		if topK < limit*3 {
			topK = limit * 3
		}
		hits, err := c.milvus.SearchMultiple(ctx, engine.CollectionForModel(model), vectors, milvus.SearchOptions{
			TopK:         topK,
			ExcludeNames: excludeNames,
		})
		if err != nil {
			return nil, err
		}
		candidates := make([]legacyCandidate, 0, len(hits))
		for _, hit := range hits {
			score := hit.Distance
			if payload.Diversity > 0 {
				score *= (1 - payload.Diversity)
			}
			candidates = append(candidates, legacyCandidate{
				Name:   hit.Name,
				Score:  score,
				Scores: []float64{score},
				Models: []string{model},
			})
		}
		modelResults[model] = candidates
	}

	var merged []legacyCandidate
	switch mergeStrategy {
	case "intersection":
		merged = legacyMergeIntersection(modelResults)
	case "priority":
		merged = legacyMergePriority(modelResults, payload.ModelPriorities, limit)
	default:
		minAgreement := payload.MinModelAgreement
		if minAgreement <= 0 {
			minAgreement = 1
		}
		merged = legacyMergeUnion(modelResults, minAgreement)
	}

	sort.Slice(merged, func(i, j int) bool {
		if merged[i].Score == merged[j].Score {
			return merged[i].Name < merged[j].Name
		}
		return merged[i].Score > merged[j].Score
	})
	if len(merged) > limit {
		merged = merged[:limit]
	}

	names := make([]string, 0, len(merged))
	for _, candidate := range merged {
		names = append(names, candidate.Name)
	}
	nameToID, err := c.resolveTrackIDs(ctx, names)
	if err != nil {
		return nil, err
	}

	tracks := make([]subsonic.RecommendationItem, 0, len(merged))
	for _, candidate := range merged {
		trackID := nameToID[candidate.Name]
		if trackID == "" {
			trackID = candidate.Name
		}
		tracks = append(tracks, subsonic.RecommendationItem{
			TrackID: trackID,
			Score:   candidate.Score,
			Models:  append([]string(nil), candidate.Models...),
		})
	}

	return &subsonic.RecommendationResponse{Tracks: tracks}, nil
}

func (c *legacyClient) resolveSeedEmbeddings(ctx context.Context, seeds []subsonic.RecommendationSeed, models []string) (map[string][][]float64, error) {
	result := make(map[string][][]float64, len(models))
	for _, model := range models {
		result[model] = make([][]float64, 0, len(seeds))
	}

	for _, seed := range seeds {
		if len(seed.Embeddings) > 0 {
			for _, model := range models {
				if embedding := seed.Embeddings[model]; len(embedding) > 0 {
					result[model] = append(result[model], embedding)
				}
			}
			continue
		}
		if len(seed.Embedding) > 0 && len(models) > 0 {
			result[models[0]] = append(result[models[0]], seed.Embedding)
			continue
		}

		lookupNames := legacySeedLookupNames(seed)
		if len(lookupNames) == 0 {
			continue
		}
		for _, model := range models {
			embeddings, err := c.milvus.GetByNames(ctx, engine.CollectionForModel(model), lookupNames)
			if err != nil {
				return nil, err
			}
			for _, lookup := range lookupNames {
				if embedding, ok := embeddings[lookup]; ok {
					result[model] = append(result[model], embedding)
					break
				}
			}
		}
	}

	return result, nil
}

func (c *legacyClient) resolveTrackIDs(ctx context.Context, names []string) (map[string]string, error) {
	if c.resolver == nil {
		ids := make(map[string]string, len(names))
		for _, name := range names {
			ids[name] = name
		}
		return ids, nil
	}
	return c.resolver.ResolveTrackIDs(ctx, names)
}

func legacyExcludeSet(payload subsonic.RecommendationRequest) []string {
	excludeSet := make(map[string]struct{})
	for _, seed := range payload.Seeds {
		for _, lookup := range legacySeedLookupNames(seed) {
			excludeSet[lookup] = struct{}{}
		}
	}
	for _, id := range payload.ExcludeTrackIDs {
		if id = strings.TrimSpace(id); id != "" {
			excludeSet[id] = struct{}{}
		}
	}
	for _, id := range payload.DislikedTrackIDs {
		if id = strings.TrimSpace(id); id != "" {
			excludeSet[id] = struct{}{}
		}
	}
	values := make([]string, 0, len(excludeSet))
	for value := range excludeSet {
		values = append(values, value)
	}
	return values
}

func legacySeedLookupNames(seed subsonic.RecommendationSeed) []string {
	seen := make(map[string]struct{}, 1+len(seed.LookupNames))
	values := make([]string, 0, 1+len(seed.LookupNames))
	appendValue := func(value string) {
		value = strings.TrimSpace(value)
		if value == "" {
			return
		}
		if _, ok := seen[value]; ok {
			return
		}
		seen[value] = struct{}{}
		values = append(values, value)
	}
	appendValue(seed.TrackID)
	for _, lookup := range seed.LookupNames {
		appendValue(lookup)
	}
	return values
}

func legacyMergeUnion(results map[string][]legacyCandidate, minAgreement int) []legacyCandidate {
	aggregated := make(map[string]*legacyCandidate)
	for model, candidates := range results {
		for _, candidate := range candidates {
			if existing, ok := aggregated[candidate.Name]; ok {
				existing.Scores = append(existing.Scores, candidate.Score)
				existing.Models = append(existing.Models, model)
			} else {
				aggregated[candidate.Name] = &legacyCandidate{
					Name:   candidate.Name,
					Scores: []float64{candidate.Score},
					Models: []string{model},
				}
			}
		}
	}
	merged := make([]legacyCandidate, 0, len(aggregated))
	for _, candidate := range aggregated {
		if len(candidate.Models) < minAgreement {
			continue
		}
		candidate.Score = average(candidate.Scores)
		merged = append(merged, *candidate)
	}
	return merged
}

func legacyMergeIntersection(results map[string][]legacyCandidate) []legacyCandidate {
	if len(results) == 0 {
		return nil
	}

	modelSets := make([]map[string]float64, 0, len(results))
	modelNames := make([]string, 0, len(results))
	for model, candidates := range results {
		current := make(map[string]float64, len(candidates))
		for _, candidate := range candidates {
			current[candidate.Name] = candidate.Score
		}
		modelSets = append(modelSets, current)
		modelNames = append(modelNames, model)
	}

	common := make(map[string]struct{})
	for name := range modelSets[0] {
		inAll := true
		for idx := 1; idx < len(modelSets); idx++ {
			if _, ok := modelSets[idx][name]; !ok {
				inAll = false
				break
			}
		}
		if inAll {
			common[name] = struct{}{}
		}
	}

	merged := make([]legacyCandidate, 0, len(common))
	for name := range common {
		scores := make([]float64, 0, len(results))
		for _, candidates := range results {
			for _, candidate := range candidates {
				if candidate.Name == name {
					scores = append(scores, candidate.Score)
					break
				}
			}
		}
		merged = append(merged, legacyCandidate{
			Name:   name,
			Score:  average(scores),
			Scores: scores,
			Models: append([]string(nil), modelNames...),
		})
	}
	return merged
}

func legacyMergePriority(results map[string][]legacyCandidate, priorities map[string]int, topK int) []legacyCandidate {
	intersection := legacyMergeIntersection(results)
	if len(intersection) >= topK {
		return intersection
	}

	primaryModel := ""
	minPriority := int(^uint(0) >> 1)
	for model := range results {
		priority := priorities[model]
		if priority <= 0 {
			priority = 100
		}
		if priority < minPriority {
			minPriority = priority
			primaryModel = model
		}
	}
	if primaryModel == "" {
		for model := range results {
			primaryModel = model
			break
		}
	}
	if primaryModel == "" {
		return intersection
	}

	merged := make([]legacyCandidate, 0, len(intersection)+len(results[primaryModel]))
	seen := make(map[string]struct{}, len(intersection))
	for _, candidate := range intersection {
		merged = append(merged, candidate)
		seen[candidate.Name] = struct{}{}
	}
	for _, candidate := range results[primaryModel] {
		if _, ok := seen[candidate.Name]; ok {
			continue
		}
		merged = append(merged, candidate)
	}
	return merged
}

func average(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sum := 0.0
	for _, value := range values {
		sum += value
	}
	return sum / float64(len(values))
}

var _ subsonic.RecommendationClient = (*legacyClient)(nil)
