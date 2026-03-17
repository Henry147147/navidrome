package milvus

import (
	"context"
	"fmt"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/navidrome/navidrome/log"
)

// SearchOptions configures similarity search behavior.
type SearchOptions struct {
	TopK         int
	ExcludeNames []string
}

// SearchResult represents a single search hit.
type SearchResult struct {
	Name     string
	Distance float64
}

// Search performs ANN similarity search on a collection.
func (c *Client) Search(ctx context.Context, collection string, vector []float64, opts SearchOptions) ([]SearchResult, error) {
	if len(vector) == 0 {
		return nil, fmt.Errorf("empty search vector")
	}

	if err := c.loadCollection(ctx, collection); err != nil {
		return nil, err
	}

	topK := opts.TopK
	if topK <= 0 {
		topK = 25
	}

	// Build search parameters for HNSW index
	// ef should be >= topK for good recall
	ef := max(64, topK*2)
	sp, err := entity.NewIndexHNSWSearchParam(ef)
	if err != nil {
		return nil, fmt.Errorf("create search params: %w", err)
	}

	// Build exclusion filter
	var filter string
	if len(opts.ExcludeNames) > 0 {
		filter = buildNotInFilter("name", opts.ExcludeNames)
	}

	// Convert to float32 vector
	vectors := []entity.Vector{entity.FloatVector(float64sToFloat32s(vector))}

	log.Debug(ctx, "Searching collection",
		"collection", collection,
		"topK", topK,
		"excludeCount", len(opts.ExcludeNames),
		"vectorDim", len(vector),
	)

	results, err := c.milvusClient.Search(
		ctx,
		collection,
		nil, // partitions
		filter,
		[]string{"name"}, // output fields
		vectors,
		"embedding",
		entity.COSINE,
		topK,
		sp,
	)
	if err != nil {
		return nil, fmt.Errorf("search %s: %w", collection, err)
	}

	var hits []SearchResult
	for _, result := range results {
		for i := 0; i < result.ResultCount; i++ {
			var name string
			for _, field := range result.Fields {
				if field.Name() == "name" {
					nameCol := field.(*entity.ColumnVarChar)
					name, _ = nameCol.ValueByIdx(i)
					break
				}
			}

			hits = append(hits, SearchResult{
				Name:     name,
				Distance: float64(result.Scores[i]),
			})
		}
	}

	log.Debug(ctx, "Search complete", "collection", collection, "hits", len(hits))
	return hits, nil
}

// SearchMultiple performs searches with multiple query vectors and combines results.
func (c *Client) SearchMultiple(ctx context.Context, collection string, vectors [][]float64, opts SearchOptions) ([]SearchResult, error) {
	if len(vectors) == 0 {
		return nil, nil
	}

	if err := c.loadCollection(ctx, collection); err != nil {
		return nil, err
	}

	topK := opts.TopK
	if topK <= 0 {
		topK = 25
	}

	ef := max(64, topK*2)
	sp, err := entity.NewIndexHNSWSearchParam(ef)
	if err != nil {
		return nil, fmt.Errorf("create search params: %w", err)
	}

	var filter string
	if len(opts.ExcludeNames) > 0 {
		filter = buildNotInFilter("name", opts.ExcludeNames)
	}

	// Convert all vectors
	entityVectors := make([]entity.Vector, len(vectors))
	for i, v := range vectors {
		entityVectors[i] = entity.FloatVector(float64sToFloat32s(v))
	}

	results, err := c.milvusClient.Search(
		ctx,
		collection,
		nil,
		filter,
		[]string{"name"},
		entityVectors,
		"embedding",
		entity.COSINE,
		topK,
		sp,
	)
	if err != nil {
		return nil, fmt.Errorf("search %s: %w", collection, err)
	}

	// Combine results from all vectors, tracking best score per name
	scoreMap := make(map[string]float64)
	for _, result := range results {
		for i := 0; i < result.ResultCount; i++ {
			var name string
			for _, field := range result.Fields {
				if field.Name() == "name" {
					nameCol := field.(*entity.ColumnVarChar)
					name, _ = nameCol.ValueByIdx(i)
					break
				}
			}

			score := float64(result.Scores[i])
			if existing, ok := scoreMap[name]; !ok || score > existing {
				scoreMap[name] = score
			}
		}
	}

	// Convert to slice
	hits := make([]SearchResult, 0, len(scoreMap))
	for name, score := range scoreMap {
		hits = append(hits, SearchResult{
			Name:     name,
			Distance: score,
		})
	}

	return hits, nil
}
