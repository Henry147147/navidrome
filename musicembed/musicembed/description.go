package musicembed

import (
	"fmt"
	"strings"

	"github.com/hybridgroup/yzma/pkg/llama"
	"github.com/hybridgroup/yzma/pkg/mtmd"
)

// GenerateDescription generates a text description for an audio file.
func (c *Client) GenerateDescription(audioFile string) (string, error) {
	llama.SamplerReset(c.sampler)

	bitmap := mtmd.BitmapInitFromFile(c.mmprojCtx, audioFile)
	if bitmap == 0 {
		return "", fmt.Errorf("failed to load audio file: %s", audioFile)
	}
	defer mtmd.BitmapFree(bitmap)

	prompt := "\n" + c.config.DescriptionPrompt
	messages := []llama.ChatMessage{llama.NewChatMessage("user", "<sound>"+prompt)}
	input := mtmd.NewInputText(applyChatTemplate(true, c.chatTemplate, messages), true, true)

	tokenizedOutput := mtmd.InputChunksInit()
	defer mtmd.InputChunksFree(tokenizedOutput)

	status := mtmd.Tokenize(c.mmprojCtx, tokenizedOutput, input, []mtmd.Bitmap{bitmap})
	if status != 0 {
		return "", fmt.Errorf("tokenization failed: error code=%d", status)
	}

	// Calculate context size requirements
	modelContextParams := llama.ContextDefaultParams()
	var maxChunkTokens uint32
	var totalNPos uint32

	for i := uint32(0); i < mtmd.InputChunksSize(tokenizedOutput); i++ {
		chunk := mtmd.InputChunksGet(tokenizedOutput, i)
		var nTokens uint32
		switch mtmd.InputChunkGetType(chunk) {
		case mtmd.InputChunkTypeText:
			nTokens = mtmd.InputChunkGetNTokens(chunk)
		case mtmd.InputChunkTypeImage, mtmd.InputChunkTypeAudio:
			nTokens = mtmd.ImageTokensGetNTokens(mtmd.InputChunkGetTokensImage(chunk))
		default:
			nTokens = mtmd.InputChunkGetNTokens(chunk)
		}
		if nTokens > maxChunkTokens {
			maxChunkTokens = nTokens
		}
		nPos := mtmd.InputChunkGetNPos(chunk)
		if nPos > 0 {
			totalNPos += uint32(nPos)
		}
	}

	if totalNPos > 0 {
		requiredCtx := totalNPos + uint32(c.config.MaxOutputTokens) + uint32(c.config.GenerationMargin)
		if ctxTrain := uint32(llama.ModelNCtxTrain(c.musicModel)); ctxTrain > 0 && requiredCtx > ctxTrain {
			requiredCtx = ctxTrain
		}
		if modelContextParams.NCtx == 0 || requiredCtx > modelContextParams.NCtx {
			modelContextParams.NCtx = requiredCtx
		}
	}

	if maxChunkTokens > modelContextParams.NBatch {
		modelContextParams.NBatch = maxChunkTokens
	}

	modelCtx, err := llama.InitFromModel(c.musicModel, modelContextParams)
	if err != nil {
		return "", fmt.Errorf("unable to initialize context from model: %w", err)
	}
	defer llama.Free(modelCtx)

	var n llama.Pos
	nBatch := int32(llama.NBatch(modelCtx))
	if status := mtmd.HelperEvalChunks(c.mmprojCtx, modelCtx, tokenizedOutput, 0, 0, nBatch, true, &n); status != 0 {
		return "", fmt.Errorf("HelperEvalChunks failed: error code=%d", status)
	}

	var sb strings.Builder
	var lastToken llama.Token
	var repeatRun int

	for i := 0; i < c.config.MaxOutputTokens; i++ {
		token := llama.SamplerSample(c.sampler, modelCtx, -1)
		llama.SamplerAccept(c.sampler, token)

		if llama.VocabIsEOG(c.vocab, token) {
			break
		}

		if token == lastToken {
			repeatRun++
			if repeatRun >= 20 {
				break
			}
		} else {
			lastToken = token
			repeatRun = 0
		}

		buf := make([]byte, 128)
		l := llama.TokenToPiece(c.vocab, token, buf, 0, true)
		sb.WriteString(string(buf[:l]))

		batch := llama.BatchGetOne([]llama.Token{token})
		batch.Pos = &n
		if status, err := llama.Decode(modelCtx, batch); err != nil || status != 0 {
			if err != nil {
				return sb.String(), fmt.Errorf("decode failed: %w", err)
			}
			return sb.String(), fmt.Errorf("decode failed with status: %d", status)
		}
		n++
	}

	return strings.TrimSpace(sb.String()), nil
}
