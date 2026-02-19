package milvus

import (
	"strings"
	"testing"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

func TestBuildSchemaFields(t *testing.T) {
	c := &Client{}

	lyricsSchema := c.buildSchema(CollectionLyrics, 123)
	if !schemaHasField(lyricsSchema, "lyrics") {
		t.Fatal("lyrics schema should include lyrics field")
	}
	if schemaHasField(lyricsSchema, "description") {
		t.Fatal("lyrics schema should not include description field")
	}

	descriptionSchema := c.buildSchema(CollectionDescription, 123)
	if !schemaHasField(descriptionSchema, "description") {
		t.Fatal("description schema should include description field")
	}
	if schemaHasField(descriptionSchema, "lyrics") {
		t.Fatal("description schema should not include lyrics field")
	}

	flamingoSchema := c.buildSchema(CollectionFlamingo, 123)
	if schemaHasField(flamingoSchema, "lyrics") {
		t.Fatal("flamingo schema should not include lyrics field")
	}
	if schemaHasField(flamingoSchema, "description") {
		t.Fatal("flamingo schema should not include description field")
	}
}

func schemaHasField(schema *entity.Schema, name string) bool {
	if schema == nil {
		return false
	}
	for _, field := range schema.Fields {
		if field.Name == name {
			return true
		}
	}
	return false
}

func TestCollectionEmbeddingDim(t *testing.T) {
	t.Run("returns dimension when embedding field exists", func(t *testing.T) {
		collection := &entity.Collection{
			Schema: &entity.Schema{
				Fields: []*entity.Field{
					{
						Name: "embedding",
						TypeParams: map[string]string{
							entity.TypeParamDim: "4096",
						},
					},
				},
			},
		}

		dim, ok := collectionEmbeddingDim(collection)
		if !ok || dim != 4096 {
			t.Fatalf("expected dim=4096, ok=true, got dim=%d ok=%v", dim, ok)
		}
	})

	t.Run("returns false when embedding field has invalid dimension", func(t *testing.T) {
		collection := &entity.Collection{
			Schema: &entity.Schema{
				Fields: []*entity.Field{
					{
						Name: "embedding",
						TypeParams: map[string]string{
							entity.TypeParamDim: "not-a-number",
						},
					},
				},
			},
		}

		_, ok := collectionEmbeddingDim(collection)
		if ok {
			t.Fatalf("expected ok=false for invalid dimension")
		}
	})

	t.Run("returns false when collection or schema is nil", func(t *testing.T) {
		if _, ok := collectionEmbeddingDim(nil); ok {
			t.Fatalf("expected ok=false for nil collection")
		}
		if _, ok := collectionEmbeddingDim(&entity.Collection{}); ok {
			t.Fatalf("expected ok=false for nil schema")
		}
	})
}

func TestCollectionHasField(t *testing.T) {
	collection := &entity.Collection{
		Schema: &entity.Schema{
			Fields: []*entity.Field{
				{Name: "name"},
				{Name: "embedding"},
			},
		},
	}

	if !collectionHasField(collection, "name") {
		t.Fatalf("expected name field to exist")
	}
	if collectionHasField(collection, "lyrics") {
		t.Fatalf("did not expect lyrics field to exist")
	}
	if collectionHasField(nil, "name") {
		t.Fatalf("expected false for nil collection")
	}
}

func TestRequiredCollectionTextField(t *testing.T) {
	if got := requiredCollectionTextField(CollectionLyrics); got != "lyrics" {
		t.Fatalf("expected lyrics field, got %q", got)
	}
	if got := requiredCollectionTextField(CollectionDescription); got != "description" {
		t.Fatalf("expected description field, got %q", got)
	}
	if got := requiredCollectionTextField(CollectionFlamingo); got != "" {
		t.Fatalf("expected no required field for flamingo, got %q", got)
	}
}

func TestSchemaMismatchError(t *testing.T) {
	tests := []struct {
		name         string
		dimKnown     bool
		expectedDim  int
		existingDim  int
		missingField bool
		wantErr      string
	}{
		{
			name:         "no mismatch",
			dimKnown:     true,
			expectedDim:  4096,
			existingDim:  4096,
			missingField: false,
			wantErr:      "",
		},
		{
			name:         "dimension mismatch only",
			dimKnown:     true,
			expectedDim:  4096,
			existingDim:  1024,
			missingField: false,
			wantErr:      "expected dim=4096 actual=1024",
		},
		{
			name:         "missing field only",
			dimKnown:     true,
			expectedDim:  4096,
			existingDim:  4096,
			missingField: true,
			wantErr:      "required field missing",
		},
		{
			name:         "dimension mismatch and missing field",
			dimKnown:     true,
			expectedDim:  4096,
			existingDim:  1024,
			missingField: true,
			wantErr:      "expected dim=4096 actual=1024 and required field is missing",
		},
		{
			name:         "unknown dimension with missing field",
			dimKnown:     false,
			expectedDim:  4096,
			existingDim:  0,
			missingField: true,
			wantErr:      "required field missing",
		},
		{
			name:         "unknown dimension without missing field treated as match",
			dimKnown:     false,
			expectedDim:  4096,
			existingDim:  0,
			missingField: false,
			wantErr:      "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := schemaMismatchError(CollectionLyrics, tt.expectedDim, tt.existingDim, tt.dimKnown, tt.missingField)
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("expected nil error, got %v", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("expected error containing %q, got nil", tt.wantErr)
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("expected error to contain %q, got %q", tt.wantErr, err.Error())
			}
		})
	}
}
