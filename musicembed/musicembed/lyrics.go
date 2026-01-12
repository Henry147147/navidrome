package musicembed

import (
	"fmt"
	"strings"

	"github.com/hybridgroup/yzma/pkg/llama"
	"github.com/hybridgroup/yzma/pkg/mtmd"
)

// GenerateLyrics generates lyrics for an audio file using Music Flamingo.
func (c *Client) GenerateLyrics(audioFile string) (string, error) {
	var lyrics string
	err := c.withMusicModel(func() error {
		llama.SamplerReset(c.sampler)

		bitmap := mtmd.BitmapInitFromFile(c.mmprojCtx, audioFile)
		if bitmap == 0 {
			return fmt.Errorf("failed to load audio file: %s", audioFile)
		}
		defer mtmd.BitmapFree(bitmap)

		prompt := "\n" + c.config.LyricsPrompt
		messages := []llama.ChatMessage{llama.NewChatMessage("user", "<sound>"+prompt)}
		input := mtmd.NewInputText(applyChatTemplate(true, c.chatTemplate, messages), true, true)

		tokenizedOutput := mtmd.InputChunksInit()
		defer mtmd.InputChunksFree(tokenizedOutput)

		status := mtmd.Tokenize(c.mmprojCtx, tokenizedOutput, input, []mtmd.Bitmap{bitmap})
		if status != 0 {
			return fmt.Errorf("tokenization failed: error code=%d", status)
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
			return fmt.Errorf("unable to initialize context from model: %w", err)
		}
		defer llama.Free(modelCtx)

		var n llama.Pos
		nBatch := int32(llama.NBatch(modelCtx))
		if status := mtmd.HelperEvalChunks(c.mmprojCtx, modelCtx, tokenizedOutput, 0, 0, nBatch, true, &n); status != 0 {
			return fmt.Errorf("HelperEvalChunks failed: error code=%d", status)
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
					return fmt.Errorf("decode failed: %w", err)
				}
				return fmt.Errorf("decode failed with status: %d", status)
			}
			n++
		}

		lyrics = strings.TrimSpace(sb.String())
		return nil
	})
	if err != nil {
		return "", err
	}
	return lyrics, nil
}

// GenerateLyricsWithCheck first checks if the audio contains lyrics, and only extracts
// lyrics if present. This avoids wasting GPU time on instrumental pieces.
// If lyrics are detected, it continues the conversation to extract them without reprocessing.
func (c *Client) GenerateLyricsWithCheck(audioFile string) (string, error) {
	var lyrics string
	err := c.withMusicModel(func() error {
		llama.SamplerReset(c.sampler)

		bitmap := mtmd.BitmapInitFromFile(c.mmprojCtx, audioFile)
		if bitmap == 0 {
			return fmt.Errorf("failed to load audio file: %s", audioFile)
		}
		defer mtmd.BitmapFree(bitmap)

		// First, ask if the song has lyrics
		checkPrompt := "\n" + c.config.LyricsCheckPrompt
		messages := []llama.ChatMessage{llama.NewChatMessage("user", "<sound>"+checkPrompt)}
		input := mtmd.NewInputText(applyChatTemplate(true, c.chatTemplate, messages), true, true)

		tokenizedOutput := mtmd.InputChunksInit()
		defer mtmd.InputChunksFree(tokenizedOutput)

		status := mtmd.Tokenize(c.mmprojCtx, tokenizedOutput, input, []mtmd.Bitmap{bitmap})
		if status != 0 {
			return fmt.Errorf("tokenization failed: error code=%d", status)
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
			// Reserve enough space for both check and potential lyrics generation
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
			return fmt.Errorf("unable to initialize context from model: %w", err)
		}
		defer llama.Free(modelCtx)

		var n llama.Pos
		nBatch := int32(llama.NBatch(modelCtx))
		if status := mtmd.HelperEvalChunks(c.mmprojCtx, modelCtx, tokenizedOutput, 0, 0, nBatch, true, &n); status != 0 {
			return fmt.Errorf("HelperEvalChunks failed: error code=%d", status)
		}

		// Generate the check response (YES/NO)
		var checkResponse strings.Builder
		maxCheckTokens := 10

		for i := 0; i < maxCheckTokens; i++ {
			token := llama.SamplerSample(c.sampler, modelCtx, -1)
			llama.SamplerAccept(c.sampler, token)

			if llama.VocabIsEOG(c.vocab, token) {
				break
			}

			buf := make([]byte, 128)
			l := llama.TokenToPiece(c.vocab, token, buf, 0, true)
			checkResponse.WriteString(string(buf[:l]))

			batch := llama.BatchGetOne([]llama.Token{token})
			batch.Pos = &n
			if status, err := llama.Decode(modelCtx, batch); err != nil || status != 0 {
				if err != nil {
					return fmt.Errorf("decode failed: %w", err)
				}
				return fmt.Errorf("decode failed with status: %d", status)
			}
			n++
		}

		response := strings.ToUpper(strings.TrimSpace(checkResponse.String()))
		hasLyrics := strings.HasPrefix(response, "YES")

		// If no lyrics, return early with a marker string
		if !hasLyrics {
			lyrics = "No lyrics identified in this piece."
			return nil
		}

		// Lyrics detected! Continue the conversation to extract them
		// Add the follow-up prompt for lyrics extraction
		lyricsPrompt := "\n" + c.config.LyricsPrompt

		// Tokenize the lyrics prompt (without audio this time)
		lyricsInput := mtmd.NewInputText(lyricsPrompt, false, false)
		lyricsTokenizedOutput := mtmd.InputChunksInit()
		defer mtmd.InputChunksFree(lyricsTokenizedOutput)

		status = mtmd.Tokenize(c.mmprojCtx, lyricsTokenizedOutput, lyricsInput, nil)
		if status != 0 {
			return fmt.Errorf("lyrics prompt tokenization failed: error code=%d", status)
		}

		// Evaluate the lyrics prompt tokens
		if status := mtmd.HelperEvalChunks(c.mmprojCtx, modelCtx, lyricsTokenizedOutput, 0, 0, nBatch, false, &n); status != 0 {
			return fmt.Errorf("HelperEvalChunks for lyrics prompt failed: error code=%d", status)
		}

		// Now generate the full lyrics
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
					return fmt.Errorf("decode failed: %w", err)
				}
				return fmt.Errorf("decode failed with status: %d", status)
			}
			n++
		}

		lyrics = strings.TrimSpace(sb.String())
		return nil
	})
	if err != nil {
		return "", err
	}
	return lyrics, nil
}

// HasLyrics checks if an audio file contains lyrics by asking the model.
// Returns true if lyrics are detected, false if instrumental or uncertain.
func (c *Client) HasLyrics(audioFile string) (bool, error) {
	var hasLyrics bool
	err := c.withMusicModel(func() error {
		llama.SamplerReset(c.sampler)

		bitmap := mtmd.BitmapInitFromFile(c.mmprojCtx, audioFile)
		if bitmap == 0 {
			return fmt.Errorf("failed to load audio file: %s", audioFile)
		}
		defer mtmd.BitmapFree(bitmap)

		prompt := "\n" + c.config.LyricsCheckPrompt
		messages := []llama.ChatMessage{llama.NewChatMessage("user", "<sound>"+prompt)}
		input := mtmd.NewInputText(applyChatTemplate(true, c.chatTemplate, messages), true, true)

		tokenizedOutput := mtmd.InputChunksInit()
		defer mtmd.InputChunksFree(tokenizedOutput)

		status := mtmd.Tokenize(c.mmprojCtx, tokenizedOutput, input, []mtmd.Bitmap{bitmap})
		if status != 0 {
			return fmt.Errorf("tokenization failed: error code=%d", status)
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
			requiredCtx := totalNPos + 10 + uint32(c.config.GenerationMargin) // Only need 10 tokens for YES/NO
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
			return fmt.Errorf("unable to initialize context from model: %w", err)
		}
		defer llama.Free(modelCtx)

		var n llama.Pos
		nBatch := int32(llama.NBatch(modelCtx))
		if status := mtmd.HelperEvalChunks(c.mmprojCtx, modelCtx, tokenizedOutput, 0, 0, nBatch, true, &n); status != 0 {
			return fmt.Errorf("HelperEvalChunks failed: error code=%d", status)
		}

		// Generate only a few tokens (enough for YES/NO response)
		var sb strings.Builder
		maxTokens := 10 // Only need a few tokens for YES/NO

		for i := 0; i < maxTokens; i++ {
			token := llama.SamplerSample(c.sampler, modelCtx, -1)
			llama.SamplerAccept(c.sampler, token)

			if llama.VocabIsEOG(c.vocab, token) {
				break
			}

			buf := make([]byte, 128)
			l := llama.TokenToPiece(c.vocab, token, buf, 0, true)
			sb.WriteString(string(buf[:l]))

			batch := llama.BatchGetOne([]llama.Token{token})
			batch.Pos = &n
			if status, err := llama.Decode(modelCtx, batch); err != nil || status != 0 {
				if err != nil {
					return fmt.Errorf("decode failed: %w", err)
				}
				return fmt.Errorf("decode failed with status: %d", status)
			}
			n++
		}

		response := strings.ToUpper(strings.TrimSpace(sb.String()))
		hasLyrics = strings.HasPrefix(response, "YES")
		return nil
	})
	if err != nil {
		return false, err
	}
	return hasLyrics, nil
}
