package milvus

import (
	"strings"
	"testing"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

func TestBuildSchemaFields(t *testing.T) {
	c := &Client{}

	audioSchema := c.buildSchema(CollectionMuQAudio, 123)
	if schemaHasField(audioSchema, "lyrics") {
		t.Fatal("muq audio schema should not include lyrics field")
	}
	if schemaHasField(audioSchema, "description") {
		t.Fatal("muq audio schema should not include description field")
	}

	sharedSchema := c.buildSchema(CollectionMuQMulan, 123)
	if schemaHasField(sharedSchema, "lyrics") {
		t.Fatal("muq mulan schema should not include lyrics field")
	}
	if schemaHasField(sharedSchema, "description") {
		t.Fatal("muq mulan schema should not include description field")
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
							entity.TypeParamDim: "512",
						},
					},
				},
			},
		}

		dim, ok := collectionEmbeddingDim(collection)
		if !ok || dim != 512 {
			t.Fatalf("expected dim=512, ok=true, got dim=%d ok=%v", dim, ok)
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
	if got := requiredCollectionTextField(CollectionMuQAudio); got != "" {
		t.Fatalf("expected no required field for muq audio, got %q", got)
	}
	if got := requiredCollectionTextField(CollectionMuQMulan); got != "" {
		t.Fatalf("expected no required field for muq mulan, got %q", got)
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
			expectedDim:  512,
			existingDim:  512,
			missingField: false,
			wantErr:      "",
		},
		{
			name:         "dimension mismatch only",
			dimKnown:     true,
			expectedDim:  512,
			existingDim:  1024,
			missingField: false,
			wantErr:      "expected dim=512 actual=1024",
		},
		{
			name:         "missing field only ignored for generic schema",
			dimKnown:     true,
			expectedDim:  512,
			existingDim:  512,
			missingField: true,
			wantErr:      "required field missing",
		},
		{
			name:         "dimension mismatch and missing field",
			dimKnown:     true,
			expectedDim:  512,
			existingDim:  1024,
			missingField: true,
			wantErr:      "expected dim=512 actual=1024 and required field is missing",
		},
		{
			name:         "unknown dimension with missing field",
			dimKnown:     false,
			expectedDim:  512,
			existingDim:  0,
			missingField: true,
			wantErr:      "required field missing",
		},
		{
			name:         "unknown dimension without missing field treated as match",
			dimKnown:     false,
			expectedDim:  512,
			existingDim:  0,
			missingField: false,
			wantErr:      "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := schemaMismatchError(CollectionMuQMulan, tt.expectedDim, tt.existingDim, tt.dimKnown, tt.missingField)
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

func TestShouldAdoptExistingDimension(t *testing.T) {
	c := &Client{}

	if !c.shouldAdoptExistingDimension(CollectionMuQAudio, DimMuQAudio, 2048) {
		t.Fatalf("expected canonical audio default to auto-adopt existing dimension")
	}
	if !c.shouldAdoptExistingDimension(CollectionMuQMulan, DimMuQMulan, 768) {
		t.Fatalf("expected canonical shared default to auto-adopt existing dimension")
	}
	if c.shouldAdoptExistingDimension(CollectionMuQAudio, 2048, 1024) {
		t.Fatalf("did not expect explicit non-default audio dimension to auto-adopt")
	}
}
