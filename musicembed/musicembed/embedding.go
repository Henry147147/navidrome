package musicembed

import (
	"fmt"
	"strings"

	"github.com/hybridgroup/yzma/pkg/llama"
	"github.com/hybridgroup/yzma/pkg/mtmd"
)

// EmbedText generates an embedding vector from text using the embedding model.
func (c *Client) EmbedText(text string) ([]float32, error) {
	if !strings.HasSuffix(text, "<|endoftext|>") {
		text += "<|endoftext|>"
	}

	vocab := llama.ModelGetVocab(c.embeddingModel)
	tokens := llama.Tokenize(vocab, text, true, true)
	if len(tokens) == 0 {
		return nil, fmt.Errorf("empty tokenization")
	}

	ctxParams := llama.ContextDefaultParams()
	ctxParams.NCtx = uint32(maxInt(c.config.ContextSize, len(tokens)))
	ctxParams.NBatch = uint32(maxInt(c.config.BatchSize, len(tokens)))
	ctxParams.PoolingType = llama.PoolingTypeLast
	ctxParams.Embeddings = 1

	lctx, err := llama.InitFromModel(c.embeddingModel, ctxParams)
	if err != nil {
		return nil, fmt.Errorf("unable to initialize context: %w", err)
	}
	defer llama.Free(lctx)

	batch := llama.BatchGetOne(tokens)
	if status, err := llama.Decode(lctx, batch); err != nil || status != 0 {
		if err != nil {
			return nil, fmt.Errorf("decode failed: %w", err)
		}
		return nil, fmt.Errorf("decode failed with status: %d", status)
	}
	_ = llama.Synchronize(lctx)

	nEmbd := llama.ModelNEmbd(c.embeddingModel)
	vec, err := llama.GetEmbeddingsSeq(lctx, 0, nEmbd)
	if err != nil {
		return nil, fmt.Errorf("unable to get embeddings: %w", err)
	}
	if len(vec) == 0 {
		return nil, fmt.Errorf("empty embeddings")
	}

	if err := validateEmbedding(vec); err != nil {
		return nil, err
	}

	emb := make([]float32, len(vec))
	copy(emb, vec)
	return emb, nil
}

// EmbedAudio generates an embedding vector directly from an audio file using mmproj.
func (c *Client) EmbedAudio(audioFile string) ([]float32, error) {
	bitmap := mtmd.BitmapInitFromFile(c.mmprojCtx, audioFile)
	if bitmap == 0 {
		return nil, fmt.Errorf("failed to load audio file: %s", audioFile)
	}
	defer mtmd.BitmapFree(bitmap)

	input := mtmd.NewInputText("<sound>", true, true)

	chunks := mtmd.InputChunksInit()
	defer mtmd.InputChunksFree(chunks)

	status := mtmd.Tokenize(c.mmprojCtx, chunks, input, []mtmd.Bitmap{bitmap})
	if status != 0 {
		return nil, fmt.Errorf("tokenization failed: %d", status)
	}

	for i := uint32(0); i < mtmd.InputChunksSize(chunks); i++ {
		chunk := mtmd.InputChunksGet(chunks, i)
		if mtmd.InputChunkGetType(chunk) == mtmd.InputChunkTypeAudio {
			if errCode := encodeChunk(c.mmprojCtx, chunk); errCode != 0 {
				return nil, fmt.Errorf("encode failed: %d", errCode)
			}

			nTokens := int(mtmd.InputChunkGetNTokens(chunk))
			nEmbd := int(llama.ModelNEmbd(c.musicModel))
			size := nTokens * nEmbd

			rawEmbeddings := getOutputEmbd(c.mmprojCtx, size)
			if rawEmbeddings == nil {
				return nil, fmt.Errorf("failed to get embeddings")
			}

			pooled := meanPool(rawEmbeddings, nTokens, nEmbd)
			return pooled, nil
		}
	}

	return nil, fmt.Errorf("no audio chunk found")
}
