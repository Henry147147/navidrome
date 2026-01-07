package main

import (
	"fmt"
	"path/filepath"

	"test/llama/musicembed"
)

var (
	musicFile  = "/home/henry/projects/navidrome/music/Lorde-Pure_Heroine-24BIT-WEB-FLAC-2013-TVRf/04-lorde-ribs.flac"
	musicFile2 = "/home/henry/projects/navidrome/music/Lorde-Pure_Heroine-24BIT-WEB-FLAC-2013-TVRf/05-lorde-buzzcut_season.flac"
)

func main() {
	config := musicembed.DefaultConfig()
	config.LibraryPath, _ = filepath.Abs("./llama-lib")

	fmt.Println("Initializing music embedding client...")
	client, err := musicembed.New(config)
	if err != nil {
		fmt.Printf("Failed to initialize: %v\n", err)
		return
	}
	defer client.Close()

	runTextEmbeddingTest(client)
	runDescriptionTest(client)
	runAudioEmbeddingTest(client)
	runSimilarityComparison(client)
}

func runTextEmbeddingTest(client *musicembed.Client) {
	fmt.Println("\n=== Text Embedding Test ===")

	text1 := "What is the capital of the United States?"
	text2 := "Washington, D.C. is the capital of the United States."
	text3 := "Toronto is a major city in Canada."

	emb1, err := client.EmbedText(text1)
	if err != nil {
		fmt.Printf("Error embedding text 1: %v\n", err)
		return
	}

	emb2, err := client.EmbedText(text2)
	if err != nil {
		fmt.Printf("Error embedding text 2: %v\n", err)
		return
	}

	emb3, err := client.EmbedText(text3)
	if err != nil {
		fmt.Printf("Error embedding text 3: %v\n", err)
		return
	}

	fmt.Printf("Related pair similarity: %.4f\n", musicembed.CosineSimilarity(emb1, emb2))
	fmt.Printf("Unrelated pair similarity: %.4f\n", musicembed.CosineSimilarity(emb1, emb3))
}

func runDescriptionTest(client *musicembed.Client) {
	fmt.Println("\n=== Song Description Test ===")
	fmt.Printf("Processing: %s\n", musicFile)

	description, err := client.GenerateDescription(musicFile)
	if err != nil {
		fmt.Printf("Error generating description: %v\n", err)
		return
	}

	fmt.Println("Description:")
	fmt.Println(description)
}

func runAudioEmbeddingTest(client *musicembed.Client) {
	fmt.Println("\n=== Audio Embedding Test ===")

	fmt.Printf("Processing: %s\n", musicFile)
	emb1, err := client.EmbedAudio(musicFile)
	if err != nil {
		fmt.Printf("Error embedding audio 1: %v\n", err)
		return
	}
	fmt.Printf("Audio 1 embedding size: %d\n", len(emb1))

	fmt.Printf("Processing: %s\n", musicFile2)
	emb2, err := client.EmbedAudio(musicFile2)
	if err != nil {
		fmt.Printf("Error embedding audio 2: %v\n", err)
		return
	}
	fmt.Printf("Audio 2 embedding size: %d\n", len(emb2))

	similarity := musicembed.CosineSimilarity(emb1, emb2)
	fmt.Printf("Audio cosine similarity: %.4f\n", similarity)

	selfSimilarity := musicembed.CosineSimilarity(emb1, emb1)
	fmt.Printf("Self-similarity (sanity check): %.4f\n", selfSimilarity)
}

func runSimilarityComparison(client *musicembed.Client) {
	fmt.Println("\n=== Description Similarity Comparison ===")

	fmt.Printf("Generating description for: %s\n", musicFile)
	desc1, err := client.GenerateDescription(musicFile)
	if err != nil {
		fmt.Printf("Error generating description 1: %v\n", err)
		return
	}

	fmt.Printf("Generating description for: %s\n", musicFile2)
	desc2, err := client.GenerateDescription(musicFile2)
	if err != nil {
		fmt.Printf("Error generating description 2: %v\n", err)
		return
	}

	emb1, err := client.EmbedText(desc1)
	if err != nil {
		fmt.Printf("Error embedding description 1: %v\n", err)
		return
	}

	emb2, err := client.EmbedText(desc2)
	if err != nil {
		fmt.Printf("Error embedding description 2: %v\n", err)
		return
	}

	similarity := musicembed.CosineSimilarity(emb1, emb2)
	fmt.Printf("Text/text cosine similarity: %.4f\n", similarity)
}
