package milvus

import (
	"context"
	"fmt"
	"strings"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/navidrome/navidrome/log"
)

// EmbeddingData holds data for Milvus storage.
type EmbeddingData struct {
	Name        string
	Embedding   []float64
	Offset      float64
	ModelID     string
	Description string
}

// DimensionForCollection returns the vector dimension for a collection.
func DimensionForCollection(collection string) int {
	switch collection {
	case CollectionLyrics:
		return DimLyrics
	case CollectionDescription:
		return DimDescription
	case CollectionFlamingo:
		return DimFlamingo
	default:
		return DimLyrics
	}
}

// Upsert inserts or updates embedding data in a collection.
func (c *Client) Upsert(ctx context.Context, collection string, data []EmbeddingData) error {
	if len(data) == 0 {
		return nil
	}

	if err := c.loadCollection(ctx, collection); err != nil {
		return err
	}

	// Prepare column data
	names := make([]string, len(data))
	embeddings := make([][]float32, len(data))
	offsets := make([]float32, len(data))
	modelIDs := make([]string, len(data))
	descriptions := make([]string, len(data))

	dim := DimensionForCollection(collection)

	for i, d := range data {
		names[i] = d.Name
		embeddings[i] = float64sToFloat32s(d.Embedding)
		offsets[i] = float32(d.Offset)
		modelIDs[i] = d.ModelID
		descriptions[i] = d.Description

		// Validate dimension
		if len(embeddings[i]) != dim {
			return fmt.Errorf("embedding dimension mismatch for %s: got %d, expected %d",
				d.Name, len(embeddings[i]), dim)
		}
	}

	// Build columns
	columns := []entity.Column{
		entity.NewColumnVarChar("name", names),
		entity.NewColumnFloatVector("embedding", dim, embeddings),
		entity.NewColumnFloat("offset", offsets),
		entity.NewColumnVarChar("model_id", modelIDs),
	}

	// Add description column for description collection
	if collection == CollectionDescription {
		columns = append(columns, entity.NewColumnVarChar("description", descriptions))
	}

	log.Debug(ctx, "Upserting embeddings", "collection", collection, "count", len(data))

	_, err := c.milvusClient.Upsert(ctx, collection, "", columns...)
	if err != nil {
		return fmt.Errorf("upsert to %s: %w", collection, err)
	}

	return c.flush(ctx, collection)
}

// GetByNames retrieves embeddings for the given track names.
func (c *Client) GetByNames(ctx context.Context, collection string, names []string) (map[string][]float64, error) {
	if len(names) == 0 {
		return map[string][]float64{}, nil
	}

	if err := c.loadCollection(ctx, collection); err != nil {
		return nil, err
	}

	filter := buildInFilter("name", names)

	results, err := c.milvusClient.Query(ctx, collection, nil, filter, []string{"name", "embedding"})
	if err != nil {
		return nil, fmt.Errorf("query %s: %w", collection, err)
	}

	embeddings := make(map[string][]float64)

	// Extract name column
	var nameCol *entity.ColumnVarChar
	var embCol *entity.ColumnFloatVector
	for _, col := range results {
		switch col.Name() {
		case "name":
			nameCol = col.(*entity.ColumnVarChar)
		case "embedding":
			embCol = col.(*entity.ColumnFloatVector)
		}
	}

	if nameCol == nil || embCol == nil {
		return embeddings, nil
	}

	for i := 0; i < nameCol.Len(); i++ {
		name, _ := nameCol.ValueByIdx(i)
		emb := embCol.Data()[i]
		embeddings[name] = float32sToFloat64s(emb)
	}

	return embeddings, nil
}

// Exists checks if names exist in the collection.
func (c *Client) Exists(ctx context.Context, collection string, names []string) (map[string]bool, error) {
	result := make(map[string]bool)
	if len(names) == 0 {
		return result, nil
	}

	if err := c.loadCollection(ctx, collection); err != nil {
		return nil, err
	}

	filter := buildInFilter("name", names)

	results, err := c.milvusClient.Query(ctx, collection, nil, filter, []string{"name"})
	if err != nil {
		return nil, fmt.Errorf("query %s: %w", collection, err)
	}

	// Initialize all as false
	for _, name := range names {
		result[name] = false
	}

	// Extract found names
	for _, col := range results {
		if col.Name() == "name" {
			nameCol := col.(*entity.ColumnVarChar)
			for i := 0; i < nameCol.Len(); i++ {
				name, _ := nameCol.ValueByIdx(i)
				result[name] = true
			}
		}
	}

	return result, nil
}

// Delete removes embeddings by names from a collection.
func (c *Client) Delete(ctx context.Context, collection string, names []string) error {
	if len(names) == 0 {
		return nil
	}

	if err := c.loadCollection(ctx, collection); err != nil {
		return err
	}

	filter := buildInFilter("name", names)

	log.Debug(ctx, "Deleting embeddings", "collection", collection, "count", len(names))

	if err := c.milvusClient.Delete(ctx, collection, "", filter); err != nil {
		return fmt.Errorf("delete from %s: %w", collection, err)
	}

	return c.flush(ctx, collection)
}

// Count returns the number of entities in a collection.
func (c *Client) Count(ctx context.Context, collection string) (int64, error) {
	stats, err := c.milvusClient.GetCollectionStatistics(ctx, collection)
	if err != nil {
		return 0, fmt.Errorf("get collection stats: %w", err)
	}

	countStr, ok := stats["row_count"]
	if !ok {
		return 0, nil
	}

	var count int64
	fmt.Sscanf(countStr, "%d", &count)
	return count, nil
}

// buildInFilter creates a filter expression for matching names.
func buildInFilter(field string, values []string) string {
	if len(values) == 0 {
		return ""
	}

	// Escape and quote values
	quoted := make([]string, len(values))
	for i, v := range values {
		// Escape single quotes
		escaped := strings.ReplaceAll(v, "'", "\\'")
		quoted[i] = fmt.Sprintf("'%s'", escaped)
	}

	return fmt.Sprintf("%s in [%s]", field, strings.Join(quoted, ", "))
}

// buildNotInFilter creates a filter expression for excluding names.
func buildNotInFilter(field string, values []string) string {
	if len(values) == 0 {
		return ""
	}

	quoted := make([]string, len(values))
	for i, v := range values {
		escaped := strings.ReplaceAll(v, "'", "\\'")
		quoted[i] = fmt.Sprintf("'%s'", escaped)
	}

	return fmt.Sprintf("%s not in [%s]", field, strings.Join(quoted, ", "))
}

// float64sToFloat32s converts a slice of float64 to float32.
func float64sToFloat32s(in []float64) []float32 {
	out := make([]float32, len(in))
	for i, v := range in {
		out[i] = float32(v)
	}
	return out
}

// float32sToFloat64s converts a slice of float32 to float64.
func float32sToFloat64s(in []float32) []float64 {
	out := make([]float64, len(in))
	for i, v := range in {
		out[i] = float64(v)
	}
	return out
}
