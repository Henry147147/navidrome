// Package llamacpp provides a local llama.cpp client backed by the yzma bindings.
package llamacpp

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/hybridgroup/yzma/pkg/llama"
	"github.com/hybridgroup/yzma/pkg/mtmd"
	"github.com/navidrome/navidrome/log"
)

const (
	DefaultLibraryPath        = "./llama-lib"
	DefaultTextModelPath      = "./models/qwen-embedder.gguf"
	DefaultAudioModelPath     = "./models/music-flamingo.gguf"
	DefaultAudioProjectorPath = "./models/mmproj-music-flamingo.gguf"

	// Prompts for Music Flamingo audio description
	lyricsPrompt = `Listen to this audio and transcribe any lyrics or vocal content you hear.
If there are no lyrics or vocals, respond with just the word "INSTRUMENTAL".
Output only the lyrics or transcription, nothing else.`

	describePrompt = `Analyze this audio and provide a detailed description including:
- Genre and subgenre
- Mood and atmosphere
- Tempo (slow/medium/fast)
- Key instruments heard
- Vocal characteristics (if any)
- Production style
Format as a single cohesive paragraph.`

	// Maximum tokens to generate for each response
	maxGenerationTokens = 512
)

var ErrNotImplemented = errors.New("llama.cpp inference not yet implemented")

// Config holds local llama.cpp configuration.
type Config struct {
	LibraryPath        string
	TextModelPath      string
	AudioModelPath     string
	AudioProjectorPath string

	ContextSize  uint32
	BatchSize    uint32
	UBatchSize   uint32
	Threads      int
	ThreadsBatch int

	GPULayers int
	MainGPU   int

	Timeout      time.Duration
	MaxRetries   int
	RetryBackoff time.Duration
}

// DefaultConfig returns sensible defaults for local llama.cpp usage.
func DefaultConfig() Config {
	return Config{
		LibraryPath:        DefaultLibraryPath,
		TextModelPath:      DefaultTextModelPath,
		AudioModelPath:     DefaultAudioModelPath,
		AudioProjectorPath: DefaultAudioProjectorPath,
		Timeout:            10 * time.Minute,
		MaxRetries:         3,
		RetryBackoff:       2 * time.Second,
	}
}

// AudioEmbedRequest for audio embedding endpoint.
type AudioEmbedRequest struct {
	AudioPath  string `json:"audio_path"`
	SampleRate int    `json:"sample_rate,omitempty"`
	BatchID    string `json:"batch_id,omitempty"`
}

// AudioEmbedResponse from audio embedding endpoint.
type AudioEmbedResponse struct {
	Embedding []float64 `json:"embedding"`
	ModelID   string    `json:"model_id"`
	Duration  float64   `json:"duration_seconds"`
	Error     string    `json:"error,omitempty"`
}

// AudioDescribeRequest for audio description endpoint.
type AudioDescribeRequest struct {
	AudioPath string `json:"audio_path"`
	Prompt    string `json:"prompt,omitempty"`
}

// AudioDescribeResponse from audio description endpoint.
type AudioDescribeResponse struct {
	Description    string    `json:"description"`
	AudioEmbedding []float64 `json:"audio_embedding"`
	ModelID        string    `json:"model_id"`
	Error          string    `json:"error,omitempty"`
}

// TextEmbedRequest for text embedding endpoint.
type TextEmbedRequest struct {
	Text    string `json:"text"`
	ModelID string `json:"model_id,omitempty"`
}

// TextEmbedResponse from text embedding endpoint.
type TextEmbedResponse struct {
	Embedding []float64 `json:"embedding"`
	ModelID   string    `json:"model_id"`
	Dimension int       `json:"dimension"`
	Error     string    `json:"error,omitempty"`
}

// Backend defines the interface for llama.cpp inference backends.
type Backend interface {
	Init(ctx context.Context) error
	Close() error
	EmbedAudioBatch(ctx context.Context, reqs []AudioEmbedRequest) ([]AudioEmbedResponse, error)
	DescribeAudioBatch(ctx context.Context, reqs []AudioDescribeRequest) ([]AudioDescribeResponse, error)
	EmbedTextBatch(ctx context.Context, reqs []TextEmbedRequest) ([]TextEmbedResponse, error)
	HealthCheck(ctx context.Context) error
}

// Client wraps the llama.cpp backend.
type Client struct {
	config  Config
	backend Backend
	mu      sync.RWMutex
	closed  bool
}

// Option configures a llama.cpp client.
type Option func(*Client)

// WithBackend overrides the default local backend (used by tests).
func WithBackend(backend Backend) Option {
	return func(c *Client) {
		c.backend = backend
	}
}

// NewClient creates a new llama.cpp client.
func NewClient(cfg Config, opts ...Option) (*Client, error) {
	cfg = applyDefaults(cfg)
	client := &Client{
		config: cfg,
	}
	for _, opt := range opts {
		opt(client)
	}
	if client.backend == nil {
		client.backend = newLocalBackend(cfg)
	}

	ctx := context.Background()
	if cfg.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, cfg.Timeout)
		defer cancel()
	}

	if err := client.backend.Init(ctx); err != nil {
		return nil, err
	}

	return client, nil
}

// Close releases backend resources.
func (c *Client) Close() error {
	c.mu.Lock()
	if c.closed {
		c.mu.Unlock()
		return nil
	}
	c.closed = true
	backend := c.backend
	c.mu.Unlock()

	if backend == nil {
		return nil
	}
	return backend.Close()
}

// HealthCheck verifies the backend is ready.
func (c *Client) HealthCheck(ctx context.Context) error {
	c.mu.RLock()
	if c.closed {
		c.mu.RUnlock()
		return fmt.Errorf("llama.cpp client is closed")
	}
	backend := c.backend
	c.mu.RUnlock()

	if backend == nil {
		return fmt.Errorf("llama.cpp backend is not configured")
	}
	return backend.HealthCheck(ctx)
}

// EmbedAudioBatch generates audio embeddings for multiple files.
func (c *Client) EmbedAudioBatch(ctx context.Context, reqs []AudioEmbedRequest) ([]AudioEmbedResponse, error) {
	if len(reqs) == 0 {
		return nil, nil
	}
	c.mu.RLock()
	if c.closed {
		c.mu.RUnlock()
		return nil, fmt.Errorf("llama.cpp client is closed")
	}
	backend := c.backend
	c.mu.RUnlock()

	if backend == nil {
		return nil, fmt.Errorf("llama.cpp backend is not configured")
	}
	return backend.EmbedAudioBatch(ctx, reqs)
}

// DescribeAudioBatch generates text descriptions for multiple audio files.
func (c *Client) DescribeAudioBatch(ctx context.Context, reqs []AudioDescribeRequest) ([]AudioDescribeResponse, error) {
	if len(reqs) == 0 {
		return nil, nil
	}
	c.mu.RLock()
	if c.closed {
		c.mu.RUnlock()
		return nil, fmt.Errorf("llama.cpp client is closed")
	}
	backend := c.backend
	c.mu.RUnlock()

	if backend == nil {
		return nil, fmt.Errorf("llama.cpp backend is not configured")
	}
	return backend.DescribeAudioBatch(ctx, reqs)
}

// EmbedTextBatch generates text embeddings for multiple strings.
func (c *Client) EmbedTextBatch(ctx context.Context, reqs []TextEmbedRequest) ([]TextEmbedResponse, error) {
	if len(reqs) == 0 {
		return nil, nil
	}
	c.mu.RLock()
	if c.closed {
		c.mu.RUnlock()
		return nil, fmt.Errorf("llama.cpp client is closed")
	}
	backend := c.backend
	c.mu.RUnlock()

	if backend == nil {
		return nil, fmt.Errorf("llama.cpp backend is not configured")
	}
	return backend.EmbedTextBatch(ctx, reqs)
}

// EmbedAudio generates an audio embedding for a single file.
func (c *Client) EmbedAudio(ctx context.Context, req AudioEmbedRequest) (*AudioEmbedResponse, error) {
	results, err := c.EmbedAudioBatch(ctx, []AudioEmbedRequest{req})
	if err != nil {
		return nil, err
	}
	if len(results) == 0 {
		return nil, fmt.Errorf("no result from audio embedding")
	}
	return &results[0], nil
}

// DescribeAudio generates a text description for a single audio file.
func (c *Client) DescribeAudio(ctx context.Context, req AudioDescribeRequest) (*AudioDescribeResponse, error) {
	results, err := c.DescribeAudioBatch(ctx, []AudioDescribeRequest{req})
	if err != nil {
		return nil, err
	}
	if len(results) == 0 {
		return nil, fmt.Errorf("no result from audio description")
	}
	return &results[0], nil
}

// EmbedText generates a text embedding for a single string.
func (c *Client) EmbedText(ctx context.Context, req TextEmbedRequest) (*TextEmbedResponse, error) {
	results, err := c.EmbedTextBatch(ctx, []TextEmbedRequest{req})
	if err != nil {
		return nil, err
	}
	if len(results) == 0 {
		return nil, fmt.Errorf("no result from text embedding")
	}
	return &results[0], nil
}

func applyDefaults(cfg Config) Config {
	defaults := DefaultConfig()
	if cfg.LibraryPath == "" {
		cfg.LibraryPath = defaults.LibraryPath
	}
	if cfg.Timeout == 0 {
		cfg.Timeout = defaults.Timeout
	}
	if cfg.MaxRetries == 0 {
		cfg.MaxRetries = defaults.MaxRetries
	}
	if cfg.RetryBackoff == 0 {
		cfg.RetryBackoff = defaults.RetryBackoff
	}
	if cfg.TextModelPath == "" && cfg.AudioModelPath == "" && cfg.AudioProjectorPath == "" {
		cfg.TextModelPath = defaults.TextModelPath
		cfg.AudioModelPath = defaults.AudioModelPath
		cfg.AudioProjectorPath = defaults.AudioProjectorPath
	}
	if cfg.AudioModelPath != "" && cfg.AudioProjectorPath == "" {
		cfg.AudioProjectorPath = defaults.AudioProjectorPath
	}
	return cfg
}

type modelResources struct {
	model    llama.Model
	vocab    llama.Vocab
	ctx      llama.Context
	embedCtx llama.Context
	mtmdCtx  mtmd.Context
}

type localBackend struct {
	cfg    Config
	mu     sync.RWMutex
	ready  bool
	closed bool
	text   *modelResources
	audio  *modelResources
}

func newLocalBackend(cfg Config) *localBackend {
	return &localBackend{cfg: cfg}
}

func (b *localBackend) Init(ctx context.Context) error {
	b.mu.Lock()
	if b.ready {
		b.mu.Unlock()
		return nil
	}
	if b.closed {
		b.mu.Unlock()
		return fmt.Errorf("llama.cpp backend is closed")
	}
	b.mu.Unlock()

	if b.cfg.LibraryPath == "" {
		return fmt.Errorf("llama.cpp library path is required")
	}
	if b.cfg.TextModelPath == "" && b.cfg.AudioModelPath == "" {
		return fmt.Errorf("no llama.cpp model paths configured")
	}

	if err := loadLlamaLibraries(b.cfg.LibraryPath); err != nil {
		return err
	}
	if b.cfg.AudioProjectorPath != "" {
		if err := loadMTMDLibraries(b.cfg.LibraryPath); err != nil {
			return err
		}
	}

	var text *modelResources
	var audio *modelResources
	var err error

	if b.cfg.TextModelPath != "" {
		text, err = b.loadTextModel(ctx)
		if err != nil {
			b.closeResources()
			return err
		}
	}

	if b.cfg.AudioModelPath != "" {
		audio, err = b.loadAudioModel(ctx)
		if err != nil {
			b.closeResources()
			return err
		}
	}

	b.mu.Lock()
	b.text = text
	b.audio = audio
	b.ready = true
	b.mu.Unlock()

	return nil
}

func (b *localBackend) Close() error {
	b.mu.Lock()
	if b.closed {
		b.mu.Unlock()
		return nil
	}
	b.closed = true
	b.mu.Unlock()

	b.closeResources()
	llama.Close()
	return nil
}

func (b *localBackend) HealthCheck(ctx context.Context) error {
	b.mu.RLock()
	defer b.mu.RUnlock()

	if b.closed {
		return fmt.Errorf("llama.cpp backend is closed")
	}
	if !b.ready {
		return fmt.Errorf("llama.cpp backend not initialized")
	}
	if b.text == nil && b.audio == nil {
		return fmt.Errorf("llama.cpp backend has no loaded models")
	}
	if b.audio != nil && b.audio.mtmdCtx != 0 {
		if !mtmd.SupportAudio(b.audio.mtmdCtx) {
			return fmt.Errorf("audio model does not support audio inputs")
		}
	}
	return nil
}

func (b *localBackend) EmbedAudioBatch(ctx context.Context, reqs []AudioEmbedRequest) ([]AudioEmbedResponse, error) {
	if len(reqs) == 0 {
		return nil, nil
	}

	b.mu.RLock()
	audio := b.audio
	b.mu.RUnlock()

	if audio == nil {
		return nil, fmt.Errorf("audio model not loaded")
	}
	if audio.mtmdCtx == 0 {
		return nil, fmt.Errorf("audio projector not loaded")
	}

	log.Debug(ctx, "EmbedAudioBatch processing", "count", len(reqs))
	responses := make([]AudioEmbedResponse, len(reqs))

	for i, req := range reqs {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		start := time.Now()

		// Clear memory state between requests
		mem, err := llama.GetMemory(audio.embedCtx)
		if err == nil {
			_ = llama.MemoryClear(mem, false)
		}

		// Load audio file as bitmap
		bitmap := mtmd.BitmapInitFromFile(audio.mtmdCtx, req.AudioPath)
		if bitmap == 0 {
			responses[i] = AudioEmbedResponse{Error: fmt.Sprintf("failed to load audio: %s", req.AudioPath)}
			continue
		}

		// Create input text with media marker for embedding mode
		marker := mtmd.DefaultMarker()
		inputText := mtmd.NewInputText(marker, true, true)

		// Create chunks container for tokenization
		chunks := mtmd.InputChunksInit()

		// Tokenize with audio bitmap
		result := mtmd.Tokenize(audio.mtmdCtx, chunks, inputText, []mtmd.Bitmap{bitmap})
		if result != 0 {
			mtmd.InputChunksFree(chunks)
			mtmd.BitmapFree(bitmap)
			responses[i] = AudioEmbedResponse{Error: fmt.Sprintf("tokenization failed: %d", result)}
			continue
		}

		// Use batch size from config, fallback to 2048 for audio
		batchSize := int32(b.cfg.BatchSize)
		if batchSize == 0 {
			batchSize = 2048
		}

		// Evaluate chunks through the embedding context
		var newNPast llama.Pos
		evalResult := mtmd.HelperEvalChunks(
			audio.mtmdCtx,
			audio.embedCtx,
			chunks,
			0,         // nPast
			0,         // seqID
			batchSize, // nBatch - must be large enough for audio chunks
			true,      // logitsLast - we want the final state for embeddings
			&newNPast,
		)

		mtmd.InputChunksFree(chunks)
		mtmd.BitmapFree(bitmap)

		if evalResult != 0 {
			responses[i] = AudioEmbedResponse{Error: fmt.Sprintf("evaluation failed: %d", evalResult)}
			continue
		}

		// Get embedding dimension from model (Flamingo produces 3584-dim)
		embeddingDim := int(llama.ModelNEmbd(audio.model))
		if embeddingDim == 0 {
			embeddingDim = 3584 // fallback for flamingo
		}

		// Get embeddings from the embedding context using sequence-based retrieval
		// Use seq_id 0 which matches what we passed to HelperEvalChunks
		embeddings, err := llama.GetEmbeddingsSeq(audio.embedCtx, 0, int32(embeddingDim))
		if err != nil {
			responses[i] = AudioEmbedResponse{Error: fmt.Sprintf("get embeddings failed: %v", err)}
			continue
		}

		// Convert float32 to float64
		embedding := make([]float64, len(embeddings))
		for j, v := range embeddings {
			embedding[j] = float64(v)
		}

		responses[i] = AudioEmbedResponse{
			Embedding: embedding,
			ModelID:   "flamingo",
			Duration:  time.Since(start).Seconds(),
		}
	}

	return responses, nil
}

func (b *localBackend) DescribeAudioBatch(ctx context.Context, reqs []AudioDescribeRequest) ([]AudioDescribeResponse, error) {
	if len(reqs) == 0 {
		return nil, nil
	}

	b.mu.RLock()
	audio := b.audio
	b.mu.RUnlock()

	if audio == nil {
		return nil, fmt.Errorf("audio model not loaded")
	}
	if audio.mtmdCtx == 0 {
		return nil, fmt.Errorf("audio projector not loaded")
	}

	log.Debug(ctx, "DescribeAudioBatch processing", "count", len(reqs))
	responses := make([]AudioDescribeResponse, len(reqs))

	for i, req := range reqs {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		// Load audio file as bitmap
		bitmap := mtmd.BitmapInitFromFile(audio.mtmdCtx, req.AudioPath)
		if bitmap == 0 {
			responses[i] = AudioDescribeResponse{Error: fmt.Sprintf("failed to load audio: %s", req.AudioPath)}
			continue
		}

		// First call: Extract lyrics from the audio
		lyrics, err := b.generateFromAudio(ctx, audio, bitmap, lyricsPrompt)
		if err != nil {
			log.Warn(ctx, "Lyrics extraction failed", err, "path", req.AudioPath)
			lyrics = ""
		}

		// Second call: Generate description of the audio
		description, err := b.generateFromAudio(ctx, audio, bitmap, describePrompt)
		mtmd.BitmapFree(bitmap)

		if err != nil {
			responses[i] = AudioDescribeResponse{Error: fmt.Sprintf("description failed: %v", err)}
			continue
		}

		// Combine lyrics and description
		fullDescription := description
		if lyrics != "" && !strings.EqualFold(strings.TrimSpace(lyrics), "INSTRUMENTAL") {
			fullDescription = fmt.Sprintf("Lyrics:\n%s\n\nDescription:\n%s", strings.TrimSpace(lyrics), description)
		}

		responses[i] = AudioDescribeResponse{
			Description: fullDescription,
			ModelID:     "flamingo",
		}
	}

	return responses, nil
}

// generateFromAudio runs a single generation pass with the audio model.
func (b *localBackend) generateFromAudio(ctx context.Context, audio *modelResources, bitmap mtmd.Bitmap, prompt string) (string, error) {
	// Clear memory state before generation
	mem, err := llama.GetMemory(audio.ctx)
	if err == nil {
		_ = llama.MemoryClear(mem, false)
	}

	// Build the full prompt with the media marker
	marker := mtmd.DefaultMarker()
	fullPrompt := marker + "\n" + prompt

	// Create input text structure
	inputText := mtmd.NewInputText(fullPrompt, true, true)

	// Create chunks container for tokenization
	chunks := mtmd.InputChunksInit()

	// Tokenize with audio bitmap
	result := mtmd.Tokenize(audio.mtmdCtx, chunks, inputText, []mtmd.Bitmap{bitmap})
	if result != 0 {
		mtmd.InputChunksFree(chunks)
		return "", fmt.Errorf("tokenization failed: %d", result)
	}

	// Use batch size from config, fallback to 2048 for audio (needs to handle large audio chunks)
	batchSize := int32(b.cfg.BatchSize)
	if batchSize == 0 {
		batchSize = 2048
	}

	// Evaluate chunks through the generation context
	var nPast llama.Pos
	evalResult := mtmd.HelperEvalChunks(
		audio.mtmdCtx,
		audio.ctx,
		chunks,
		0,         // nPast
		0,         // seqID
		batchSize, // nBatch - must be large enough for audio chunks
		true,      // logitsLast
		&nPast,
	)
	mtmd.InputChunksFree(chunks)

	if evalResult != 0 {
		return "", fmt.Errorf("evaluation failed: %d", evalResult)
	}

	// Generate response tokens
	output := b.generateTokens(ctx, audio, nPast)
	return strings.TrimSpace(output), nil
}

// generateTokens generates text tokens from the current context state.
func (b *localBackend) generateTokens(ctx context.Context, audio *modelResources, nPast llama.Pos) string {
	var result strings.Builder

	// Create sampler chain
	sampler := b.createSampler()
	defer llama.SamplerFree(sampler)

	for i := 0; i < maxGenerationTokens; i++ {
		select {
		case <-ctx.Done():
			return result.String()
		default:
		}

		// Sample next token from the last computed logits position (-1)
		token := llama.SamplerSample(sampler, audio.ctx, -1)

		// Check for end of generation
		if llama.VocabIsEOG(audio.vocab, token) {
			break
		}

		// Accept the token for the sampler state
		llama.SamplerAccept(sampler, token)

		// Convert token to text
		buf := make([]byte, 256)
		n := llama.TokenToPiece(audio.vocab, token, buf, 0, true)
		if n > 0 {
			result.Write(buf[:n])
		}

		// Create batch with the new token and decode
		batch := llama.BatchGetOne([]llama.Token{token})
		if _, err := llama.Decode(audio.ctx, batch); err != nil {
			break
		}
		nPast++
	}

	return result.String()
}

// createSampler creates a sampler chain for text generation.
func (b *localBackend) createSampler() llama.Sampler {
	params := llama.SamplerChainDefaultParams()
	chain := llama.SamplerChainInit(params)

	// Add temperature sampler (0.7 for creative but coherent output)
	llama.SamplerChainAdd(chain, llama.SamplerInitTempExt(0.7, 0.0, 1.0))

	// Add top-p sampler (nucleus sampling)
	llama.SamplerChainAdd(chain, llama.SamplerInitTopP(0.9, 1))

	// Add top-k sampler
	llama.SamplerChainAdd(chain, llama.SamplerInitTopK(40))

	// Add repetition penalty
	llama.SamplerChainAdd(chain, llama.SamplerInitPenalties(64, 1.1, 0.0, 0.0))

	// Add distribution sampler for final token selection
	llama.SamplerChainAdd(chain, llama.SamplerInitDist(0))

	return chain
}

func (b *localBackend) EmbedTextBatch(ctx context.Context, reqs []TextEmbedRequest) ([]TextEmbedResponse, error) {
	if len(reqs) == 0 {
		return nil, nil
	}

	b.mu.RLock()
	text := b.text
	b.mu.RUnlock()

	if text == nil {
		return nil, fmt.Errorf("text model not loaded")
	}

	log.Debug(ctx, "EmbedTextBatch processing", "count", len(reqs))
	responses := make([]TextEmbedResponse, len(reqs))

	for i, req := range reqs {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		// Clear memory state between requests
		mem, err := llama.GetMemory(text.embedCtx)
		if err == nil {
			_ = llama.MemoryClear(mem, false)
		}

		// Tokenize the text
		tokens := llama.Tokenize(text.vocab, req.Text, true, true)
		if len(tokens) == 0 {
			responses[i] = TextEmbedResponse{Error: "tokenization produced no tokens"}
			continue
		}

		// Create batch from tokens
		batch := llama.BatchGetOne(tokens)

		// Decode to compute embeddings
		if _, err := llama.Decode(text.embedCtx, batch); err != nil {
			responses[i] = TextEmbedResponse{Error: fmt.Sprintf("decode failed: %v", err)}
			continue
		}

		// Get embedding dimension from model
		embeddingDim := int(llama.ModelNEmbd(text.model))
		if embeddingDim == 0 {
			embeddingDim = 2560 // fallback for qwen embedding model
		}

		// Get embeddings for sequence 0 (used in BatchGetOne)
		embeddings, err := llama.GetEmbeddingsSeq(text.embedCtx, 0, int32(embeddingDim))
		if err != nil {
			responses[i] = TextEmbedResponse{Error: fmt.Sprintf("get embeddings failed: %v", err)}
			continue
		}

		// Convert float32 to float64
		embedding := make([]float64, len(embeddings))
		for j, v := range embeddings {
			embedding[j] = float64(v)
		}

		responses[i] = TextEmbedResponse{
			Embedding: embedding,
			ModelID:   "qwen",
			Dimension: len(embedding),
		}
	}

	return responses, nil
}

func (b *localBackend) loadTextModel(ctx context.Context) (*modelResources, error) {
	log.Info(ctx, "Loading llama.cpp text model", "path", b.cfg.TextModelPath)
	model, err := llama.ModelLoadFromFile(b.cfg.TextModelPath, b.modelParams())
	if err != nil {
		return nil, fmt.Errorf("load text model: %w", err)
	}

	embedCtx, err := llama.InitFromModel(model, b.contextParams(true))
	if err != nil {
		_ = llama.ModelFree(model)
		return nil, fmt.Errorf("init text embedding context: %w", err)
	}

	return &modelResources{
		model:    model,
		vocab:    llama.ModelGetVocab(model),
		embedCtx: embedCtx,
	}, nil
}

func (b *localBackend) loadAudioModel(ctx context.Context) (*modelResources, error) {
	log.Info(ctx, "Loading llama.cpp audio model", "path", b.cfg.AudioModelPath)
	model, err := llama.ModelLoadFromFile(b.cfg.AudioModelPath, b.modelParams())
	if err != nil {
		return nil, fmt.Errorf("load audio model: %w", err)
	}

	ctxParams := b.contextParams(false)
	lctx, err := llama.InitFromModel(model, ctxParams)
	if err != nil {
		_ = llama.ModelFree(model)
		return nil, fmt.Errorf("init audio context: %w", err)
	}

	embedCtx, err := llama.InitFromModel(model, b.contextParams(true))
	if err != nil {
		_ = llama.Free(lctx)
		_ = llama.ModelFree(model)
		return nil, fmt.Errorf("init audio embedding context: %w", err)
	}

	var mtmdCtx mtmd.Context
	if b.cfg.AudioProjectorPath != "" {
		mtmdCtx, err = mtmd.InitFromFile(b.cfg.AudioProjectorPath, model, b.mtmdParams())
		if err != nil {
			_ = llama.Free(embedCtx)
			_ = llama.Free(lctx)
			_ = llama.ModelFree(model)
			return nil, fmt.Errorf("init audio projector: %w", err)
		}
	}

	return &modelResources{
		model:    model,
		vocab:    llama.ModelGetVocab(model),
		ctx:      lctx,
		embedCtx: embedCtx,
		mtmdCtx:  mtmdCtx,
	}, nil
}

func (b *localBackend) closeResources() {
	b.mu.Lock()
	text := b.text
	audio := b.audio
	b.text = nil
	b.audio = nil
	b.mu.Unlock()

	if text != nil {
		if text.embedCtx != 0 {
			_ = llama.Free(text.embedCtx)
		}
		if text.ctx != 0 {
			_ = llama.Free(text.ctx)
		}
		if text.model != 0 {
			_ = llama.ModelFree(text.model)
		}
	}

	if audio != nil {
		if audio.mtmdCtx != 0 {
			_ = mtmd.Free(audio.mtmdCtx)
		}
		if audio.embedCtx != 0 {
			_ = llama.Free(audio.embedCtx)
		}
		if audio.ctx != 0 {
			_ = llama.Free(audio.ctx)
		}
		if audio.model != 0 {
			_ = llama.ModelFree(audio.model)
		}
	}
}

func (b *localBackend) modelParams() llama.ModelParams {
	params := llama.ModelDefaultParams()
	if b.cfg.GPULayers > 0 {
		params.NGpuLayers = int32(b.cfg.GPULayers)
		params.MainGpu = int32(b.cfg.MainGPU)
	}
	return params
}

func (b *localBackend) contextParams(embeddings bool) llama.ContextParams {
	params := llama.ContextDefaultParams()
	if b.cfg.ContextSize > 0 {
		params.NCtx = b.cfg.ContextSize
	}
	if b.cfg.BatchSize > 0 {
		params.NBatch = b.cfg.BatchSize
	}
	if b.cfg.UBatchSize > 0 {
		params.NUbatch = b.cfg.UBatchSize
	}
	if b.cfg.Threads > 0 {
		params.NThreads = int32(b.cfg.Threads)
	}
	if b.cfg.ThreadsBatch > 0 {
		params.NThreadsBatch = int32(b.cfg.ThreadsBatch)
	}
	if embeddings {
		params.Embeddings = 1
		// Use PoolingTypeLast (3) as required by qwen embedding models
		// The model metadata specifies pooling_type=3 (last token pooling)
		params.PoolingType = llama.PoolingTypeLast
	}
	return params
}

func (b *localBackend) mtmdParams() mtmd.ContextParamsType {
	params := mtmd.ContextParamsDefault()
	if b.cfg.Threads > 0 {
		params.Threads = int32(b.cfg.Threads)
	}
	if b.cfg.GPULayers > 0 {
		params.UseGPU = true
	}
	return params
}

var (
	llamaLoadOnce sync.Once
	llamaLoadErr  error
	llamaLoadPath string

	mtmdLoadOnce sync.Once
	mtmdLoadErr  error
	mtmdLoadPath string
)

func loadLlamaLibraries(path string) error {
	llamaLoadOnce.Do(func() {
		llamaLoadPath = path
		llamaLoadErr = llama.Load(path)
		if llamaLoadErr == nil {
			llama.Init()
		}
	})
	if llamaLoadErr != nil {
		return llamaLoadErr
	}
	if llamaLoadPath != path {
		return fmt.Errorf("llama.cpp already loaded from %s", llamaLoadPath)
	}
	return nil
}

func loadMTMDLibraries(path string) error {
	mtmdLoadOnce.Do(func() {
		mtmdLoadPath = path
		mtmdLoadErr = mtmd.Load(path)
	})
	if mtmdLoadErr != nil {
		return mtmdLoadErr
	}
	if mtmdLoadPath != path {
		return fmt.Errorf("mtmd already loaded from %s", mtmdLoadPath)
	}
	return nil
}
