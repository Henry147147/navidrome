package milvus

import (
	"context"
	"fmt"
	"strconv"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/navidrome/navidrome/log"
)

// ensureCollection creates a collection if it doesn't exist.
func (c *Client) ensureCollection(ctx context.Context, name string, dim int) error {
	exists, err := c.milvusClient.HasCollection(ctx, name)
	if err != nil {
		return fmt.Errorf("check collection %s: %w", name, err)
	}

	if exists {
		collection, err := c.milvusClient.DescribeCollection(ctx, name)
		if err != nil {
			return fmt.Errorf("describe collection %s: %w", name, err)
		}
		existingDim, ok := collectionEmbeddingDim(collection)
		if ok && existingDim != dim {
			log.Warn(ctx, "Collection dimension mismatch, recreating",
				"collection", name,
				"expected", dim,
				"actual", existingDim,
			)
			if err := c.DropCollection(ctx, name); err != nil {
				return fmt.Errorf("drop collection %s: %w", name, err)
			}
		} else {
			log.Debug(ctx, "Collection already exists", "collection", name)
			return nil
		}
	}

	log.Info(ctx, "Creating collection", "collection", name, "dimension", dim)

	schema := c.buildSchema(name, dim)
	if err := c.milvusClient.CreateCollection(ctx, schema, entity.DefaultShardNumber); err != nil {
		return fmt.Errorf("create collection %s: %w", name, err)
	}

	// Create index on the embedding field
	if err := c.createIndex(ctx, name); err != nil {
		return fmt.Errorf("create index for %s: %w", name, err)
	}

	return nil
}

func collectionEmbeddingDim(collection *entity.Collection) (int, bool) {
	if collection == nil || collection.Schema == nil {
		return 0, false
	}
	for _, field := range collection.Schema.Fields {
		if field.Name != "embedding" {
			continue
		}
		if field.TypeParams == nil {
			return 0, false
		}
		dimStr, ok := field.TypeParams[entity.TypeParamDim]
		if !ok {
			return 0, false
		}
		dim, err := strconv.Atoi(dimStr)
		if err != nil {
			return 0, false
		}
		return dim, true
	}
	return 0, false
}

// buildSchema constructs the schema for a collection.
func (c *Client) buildSchema(name string, dim int) *entity.Schema {
	schema := &entity.Schema{
		CollectionName: name,
		AutoID:         false,
		Fields: []*entity.Field{
			{
				Name:       "name",
				DataType:   entity.FieldTypeVarChar,
				PrimaryKey: true,
				TypeParams: map[string]string{
					"max_length": "512",
				},
			},
			{
				Name:     "embedding",
				DataType: entity.FieldTypeFloatVector,
				TypeParams: map[string]string{
					"dim": fmt.Sprintf("%d", dim),
				},
			},
			{
				Name:     "offset",
				DataType: entity.FieldTypeFloat,
			},
			{
				Name:     "model_id",
				DataType: entity.FieldTypeVarChar,
				TypeParams: map[string]string{
					"max_length": "256",
				},
			},
		},
	}

	// Add description field for the description collection
	if name == CollectionDescription {
		schema.Fields = append(schema.Fields, &entity.Field{
			Name:     "description",
			DataType: entity.FieldTypeVarChar,
			TypeParams: map[string]string{
				"max_length": "4096",
			},
		})
	}

	return schema
}

// createIndex creates an HNSW index on the embedding field.
func (c *Client) createIndex(ctx context.Context, name string) error {
	// Use HNSW index for good search performance
	idx, err := entity.NewIndexHNSW(entity.COSINE, 50, 250)
	if err != nil {
		return fmt.Errorf("create HNSW index params: %w", err)
	}

	log.Debug(ctx, "Creating HNSW index", "collection", name)
	if err := c.milvusClient.CreateIndex(ctx, name, "embedding", idx, false); err != nil {
		return fmt.Errorf("create index on %s.embedding: %w", name, err)
	}

	// Create inverted index on name field for filtering
	nameIdx := entity.NewScalarIndexWithType(entity.Inverted)
	if err := c.milvusClient.CreateIndex(ctx, name, "name", nameIdx, false); err != nil {
		// Non-fatal: some Milvus versions don't support scalar indexes
		log.Warn(ctx, "Could not create scalar index on name field", "collection", name, "error", err)
	}

	return nil
}

// DropCollection removes a collection and all its data.
func (c *Client) DropCollection(ctx context.Context, name string) error {
	exists, err := c.milvusClient.HasCollection(ctx, name)
	if err != nil {
		return fmt.Errorf("check collection %s: %w", name, err)
	}

	if !exists {
		return nil
	}

	log.Info(ctx, "Dropping collection", "collection", name)
	if err := c.milvusClient.DropCollection(ctx, name); err != nil {
		return fmt.Errorf("drop collection %s: %w", name, err)
	}

	c.mu.Lock()
	delete(c.loaded, name)
	c.mu.Unlock()

	return nil
}

// CollectionStats returns statistics about a collection.
func (c *Client) CollectionStats(ctx context.Context, name string) (map[string]string, error) {
	stats, err := c.milvusClient.GetCollectionStatistics(ctx, name)
	if err != nil {
		return nil, fmt.Errorf("get stats for %s: %w", name, err)
	}
	return stats, nil
}
