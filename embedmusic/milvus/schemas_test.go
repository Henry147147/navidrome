package milvus

import (
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
