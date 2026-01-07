// Package llamacpp provides a local llama.cpp client backed by the yzma bindings.
package llamacpp

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/hybridgroup/yzma/pkg/llama"
	"github.com/hybridgroup/yzma/pkg/loader"
	"github.com/hybridgroup/yzma/pkg/mtmd"
	"github.com/navidrome/navidrome/log"
)

var (
	DefaultLibraryPath        = defaultLibraryPath()
	DefaultTextModelPath      = "./musicembed/models/qwen-embedder-4b.gguf"
	DefaultAudioModelPath     = "./musicembed/models/music-flamingo.gguf"
	DefaultAudioProjectorPath = "./musicembed/models/mmproj-music-flamingo.gguf"

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
	maxGenerationTokens = 2048
)

func defaultLibraryPath() string {
	path := os.Getenv("YZMA_LIB")
	if path == "" {
		path = "./musicembed/llama-lib"
	}
	absPath, err := filepath.Abs(path)
	if err != nil {
		return path
	}
	return absPath
}

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
	if cfg.LibraryPath != "" {
		if absPath, err := filepath.Abs(cfg.LibraryPath); err == nil {
			cfg.LibraryPath = absPath
		}
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
	cfg         Config
	mu          sync.RWMutex
	ready       bool
	closed      bool
	text        *modelResources
	audio       *modelResources
	textRefs    int
	audioRefs   int
	textLoadMu  sync.Mutex
	audioLoadMu sync.Mutex
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

	b.mu.Lock()
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
		if b.cfg.TextModelPath == "" && b.cfg.AudioModelPath == "" {
			return fmt.Errorf("llama.cpp backend has no configured models")
		}
		return nil
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

	audio, release, err := b.acquireAudio(ctx)
	if err != nil {
		return nil, err
	}
	defer release()

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

		tokenCount := countChunkTokens(chunks)
		posCount := countChunkPositions(chunks)
		if err := b.ensureAudioEvalCapacity(ctx, audio, tokenCount, posCount, true); err != nil {
			mtmd.InputChunksFree(chunks)
			mtmd.BitmapFree(bitmap)
			responses[i] = AudioEmbedResponse{Error: fmt.Sprintf("ensure batch size failed: %v", err)}
			continue
		}

		batchSize := int32(llama.NBatch(audio.embedCtx))

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

	audio, release, err := b.acquireAudio(ctx)
	if err != nil {
		return nil, err
	}
	defer release()

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
			log.Warn(ctx, "Lyrics extraction failed", "error", err, "path", req.AudioPath)
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

	tokenCount := countChunkTokens(chunks)
	posCount := countChunkPositions(chunks)
	if err := b.ensureAudioEvalCapacity(ctx, audio, tokenCount, posCount, false); err != nil {
		mtmd.InputChunksFree(chunks)
		return "", fmt.Errorf("ensure batch size failed: %w", err)
	}

	batchSize := int32(llama.NBatch(audio.ctx))

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

	text, release, err := b.acquireText(ctx)
	if err != nil {
		return nil, err
	}
	defer release()

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
		if err := b.ensureTextEvalCapacity(ctx, text, uint32(len(tokens))); err != nil {
			responses[i] = TextEmbedResponse{Error: fmt.Sprintf("ensure batch size failed: %v", err)}
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

func (b *localBackend) acquireText(ctx context.Context) (*modelResources, func(), error) {
	if b.cfg.TextModelPath == "" {
		return nil, nil, fmt.Errorf("text model not configured")
	}

	b.mu.Lock()
	if b.closed {
		b.mu.Unlock()
		return nil, nil, fmt.Errorf("llama.cpp backend is closed")
	}
	if b.text != nil {
		log.Debug(ctx, "llama.cpp text model already loaded")
		b.textRefs++
		text := b.text
		b.mu.Unlock()
		return text, b.releaseText, nil
	}
	b.mu.Unlock()

	b.textLoadMu.Lock()
	defer b.textLoadMu.Unlock()

	b.mu.Lock()
	if b.closed {
		b.mu.Unlock()
		return nil, nil, fmt.Errorf("llama.cpp backend is closed")
	}
	if b.text != nil {
		log.Debug(ctx, "llama.cpp text model already loaded")
		b.textRefs++
		text := b.text
		b.mu.Unlock()
		return text, b.releaseText, nil
	}
	b.mu.Unlock()

	log.Info(ctx, "Loading llama.cpp text model (lazy)", "path", b.cfg.TextModelPath)
	text, err := b.loadTextModel(ctx)
	if err != nil {
		return nil, nil, err
	}

	b.mu.Lock()
	if b.closed {
		b.mu.Unlock()
		b.freeTextResources(text)
		return nil, nil, fmt.Errorf("llama.cpp backend is closed")
	}
	if b.text != nil {
		log.Debug(ctx, "llama.cpp text model already loaded")
		b.textRefs++
		existing := b.text
		b.mu.Unlock()
		b.freeTextResources(text)
		return existing, b.releaseText, nil
	}
	b.text = text
	b.textRefs = 1
	b.mu.Unlock()

	log.Info(ctx, "Loaded llama.cpp text model")
	return text, b.releaseText, nil
}

func (b *localBackend) acquireAudio(ctx context.Context) (*modelResources, func(), error) {
	if b.cfg.AudioModelPath == "" {
		return nil, nil, fmt.Errorf("audio model not configured")
	}

	b.mu.Lock()
	if b.closed {
		b.mu.Unlock()
		return nil, nil, fmt.Errorf("llama.cpp backend is closed")
	}
	if b.audio != nil {
		log.Debug(ctx, "llama.cpp audio model already loaded")
		b.audioRefs++
		audio := b.audio
		b.mu.Unlock()
		return audio, b.releaseAudio, nil
	}
	b.mu.Unlock()

	b.audioLoadMu.Lock()
	defer b.audioLoadMu.Unlock()

	b.mu.Lock()
	if b.closed {
		b.mu.Unlock()
		return nil, nil, fmt.Errorf("llama.cpp backend is closed")
	}
	if b.audio != nil {
		log.Debug(ctx, "llama.cpp audio model already loaded")
		b.audioRefs++
		audio := b.audio
		b.mu.Unlock()
		return audio, b.releaseAudio, nil
	}
	b.mu.Unlock()

	log.Info(ctx, "Loading llama.cpp audio model (lazy)", "path", b.cfg.AudioModelPath)
	audio, err := b.loadAudioModel(ctx)
	if err != nil {
		return nil, nil, err
	}

	b.mu.Lock()
	if b.closed {
		b.mu.Unlock()
		b.freeAudioResources(audio)
		return nil, nil, fmt.Errorf("llama.cpp backend is closed")
	}
	if b.audio != nil {
		log.Debug(ctx, "llama.cpp audio model already loaded")
		b.audioRefs++
		existing := b.audio
		b.mu.Unlock()
		b.freeAudioResources(audio)
		return existing, b.releaseAudio, nil
	}
	b.audio = audio
	b.audioRefs = 1
	b.mu.Unlock()

	log.Info(ctx, "Loaded llama.cpp audio model")
	return audio, b.releaseAudio, nil
}

func (b *localBackend) releaseText() {
	b.mu.Lock()
	if b.textRefs > 0 {
		b.textRefs--
	}
	if b.textRefs != 0 || b.text == nil {
		b.mu.Unlock()
		return
	}
	text := b.text
	b.text = nil
	b.mu.Unlock()

	log.Info(context.Background(), "Unloading llama.cpp text model")
	b.freeTextResources(text)
}

func (b *localBackend) releaseAudio() {
	b.mu.Lock()
	if b.audioRefs > 0 {
		b.audioRefs--
	}
	if b.audioRefs != 0 || b.audio == nil {
		b.mu.Unlock()
		return
	}
	audio := b.audio
	b.audio = nil
	b.mu.Unlock()

	log.Info(context.Background(), "Unloading llama.cpp audio model")
	b.freeAudioResources(audio)
}

func (b *localBackend) ensureTextEvalCapacity(ctx context.Context, text *modelResources, requiredTokens uint32) error {
	if requiredTokens == 0 {
		return nil
	}
	if text.embedCtx == 0 {
		return fmt.Errorf("text embedding context not initialized")
	}

	currentBatch := llama.NBatch(text.embedCtx)
	currentCtx := llama.NCtx(text.embedCtx)
	newBatch := nextBatchSize(requiredTokens, currentBatch, b.cfg.BatchSize)
	newCtx := nextContextSize(requiredTokens, currentCtx, b.cfg.ContextSize)
	if currentBatch >= newBatch && currentCtx >= newCtx {
		return nil
	}

	log.Warn(ctx, "Resizing llama.cpp context to fit text tokens",
		"requiredTokens", requiredTokens,
		"batchFrom", currentBatch,
		"batchTo", newBatch,
		"ctxFrom", currentCtx,
		"ctxTo", newCtx,
	)

	b.textLoadMu.Lock()
	defer b.textLoadMu.Unlock()

	currentBatch = llama.NBatch(text.embedCtx)
	currentCtx = llama.NCtx(text.embedCtx)
	if currentBatch >= newBatch && currentCtx >= newCtx {
		return nil
	}

	params := b.contextParams(true)
	params.NBatch = newBatch
	params.NCtx = newCtx
	newCtxHandle, err := llama.InitFromModel(text.model, params)
	if err != nil {
		return fmt.Errorf("init text embedding context: %w", err)
	}

	old := text.embedCtx
	text.embedCtx = newCtxHandle
	if old != 0 {
		_ = llama.Free(old)
	}

	return nil
}

func (b *localBackend) ensureAudioEvalCapacity(ctx context.Context, audio *modelResources, requiredTokens, requiredPositions uint32, embeddings bool) error {
	if requiredTokens == 0 {
		return nil
	}

	var targetCtx llama.Context
	kind := "audio generation"
	requiredCtx := requiredPositions
	if requiredCtx == 0 {
		requiredCtx = requiredTokens
	}
	if embeddings {
		targetCtx = audio.embedCtx
		kind = "audio embedding"
	} else {
		targetCtx = audio.ctx
		requiredCtx += uint32(maxGenerationTokens)
	}
	const audioCtxMargin = 256
	requiredCtx += audioCtxMargin
	if targetCtx == 0 {
		return fmt.Errorf("%s context not initialized", kind)
	}

	currentBatch := llama.NBatch(targetCtx)
	currentCtx := llama.NCtx(targetCtx)
	newBatch := nextBatchSize(requiredTokens, currentBatch, b.cfg.BatchSize)
	newCtx := nextContextSize(requiredCtx, currentCtx, b.cfg.ContextSize)

	if currentBatch >= newBatch && currentCtx >= newCtx {
		return nil
	}

	log.Warn(ctx, "Resizing llama.cpp context to fit audio tokens",
		"kind", kind,
		"requiredTokens", requiredTokens,
		"requiredCtx", requiredCtx,
		"batchFrom", currentBatch,
		"batchTo", newBatch,
		"ctxFrom", currentCtx,
		"ctxTo", newCtx,
	)

	b.audioLoadMu.Lock()
	defer b.audioLoadMu.Unlock()

	// Re-check under lock in case another goroutine already resized.
	if embeddings {
		targetCtx = audio.embedCtx
	} else {
		targetCtx = audio.ctx
	}
	currentBatch = llama.NBatch(targetCtx)
	currentCtx = llama.NCtx(targetCtx)
	if currentBatch >= newBatch && currentCtx >= newCtx {
		return nil
	}

	params := b.contextParams(embeddings)
	params.NBatch = newBatch
	params.NCtx = newCtx
	newEvalCtx, err := llama.InitFromModel(audio.model, params)
	if err != nil {
		return fmt.Errorf("init %s context: %w", kind, err)
	}

	if embeddings {
		old := audio.embedCtx
		audio.embedCtx = newEvalCtx
		if old != 0 {
			_ = llama.Free(old)
		}
	} else {
		old := audio.ctx
		audio.ctx = newEvalCtx
		if old != 0 {
			_ = llama.Free(old)
		}
	}

	return nil
}

func (b *localBackend) loadTextModel(ctx context.Context) (*modelResources, error) {
	log.Info(ctx, "Loading llama.cpp text model", "path", b.cfg.TextModelPath)
	var model llama.Model
	err := withSilentLlamaLogs(func() error {
		var loadErr error
		model, loadErr = llama.ModelLoadFromFile(b.cfg.TextModelPath, b.modelParams())
		return loadErr
	})
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
	var model llama.Model
	err := withSilentLlamaLogs(func() error {
		var loadErr error
		model, loadErr = llama.ModelLoadFromFile(b.cfg.AudioModelPath, b.modelParams())
		return loadErr
	})
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
		err = withSilentLlamaLogs(func() error {
			var initErr error
			mtmdCtx, initErr = mtmd.InitFromFile(b.cfg.AudioProjectorPath, model, b.mtmdParams())
			return initErr
		})
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
	b.textRefs = 0
	b.audioRefs = 0
	b.mu.Unlock()

	if text != nil {
		b.freeTextResources(text)
	}

	if audio != nil {
		b.freeAudioResources(audio)
	}
}

func (b *localBackend) freeTextResources(text *modelResources) {
	if text == nil {
		return
	}
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

func (b *localBackend) freeAudioResources(audio *modelResources) {
	if audio == nil {
		return
	}
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
	mtmdLogReady bool
)

func loadLlamaLibraries(path string) error {
	llamaLoadOnce.Do(func() {
		llamaLoadPath = path
		llamaLoadErr = ensureLibraryPath(path)
		if llamaLoadErr == nil {
			llamaLoadErr = preloadGGMLDeps(path)
		}
		if llamaLoadErr == nil {
			llamaLoadErr = llama.Load(path)
		}
		if llamaLoadErr == nil {
			configureLlamaLogging()
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
		mtmdLoadErr = ensureLibraryPath(path)
		if mtmdLoadErr == nil {
			mtmdLoadErr = mtmd.Load(path)
			if mtmdLoadErr == nil {
				mtmdLogReady = true
				configureLlamaLogging()
			}
		}
	})
	if mtmdLoadErr != nil {
		return mtmdLoadErr
	}
	if mtmdLoadPath != path {
		return fmt.Errorf("mtmd already loaded from %s", mtmdLoadPath)
	}
	return nil
}

func ensureLibraryPath(path string) error {
	if path == "" {
		return nil
	}
	absPath, err := filepath.Abs(path)
	if err != nil {
		log.Warn(context.Background(), "Failed to resolve llama.cpp library path", "path", path, "err", err)
		return err
	}
	var envKey string
	switch runtime.GOOS {
	case "windows":
		envKey = "PATH"
	case "darwin":
		envKey = "DYLD_LIBRARY_PATH"
	default:
		envKey = "LD_LIBRARY_PATH"
	}
	log.Info(context.Background(), "Setting llama.cpp library search path", "env", envKey, "path", absPath)
	return prependEnvPath(envKey, absPath)
}

func prependEnvPath(envKey, path string) error {
	if envKey == "" || path == "" {
		return nil
	}
	current := os.Getenv(envKey)
	parts := strings.Split(current, string(os.PathListSeparator))
	for _, part := range parts {
		if part == path {
			log.Debug(context.Background(), "Library search path already contains path", "env", envKey, "path", path)
			return nil
		}
	}
	if current == "" {
		log.Debug(context.Background(), "Setting library search path", "env", envKey, "path", path)
		return os.Setenv(envKey, path)
	}
	log.Debug(context.Background(), "Prepending library search path", "env", envKey, "path", path)
	return os.Setenv(envKey, path+string(os.PathListSeparator)+current)
}

func countChunkTokens(chunks mtmd.InputChunks) uint32 {
	var total uint32
	size := mtmd.InputChunksSize(chunks)
	for i := uint32(0); i < size; i++ {
		chunk := mtmd.InputChunksGet(chunks, i)
		total += mtmd.InputChunkGetNTokens(chunk)
	}
	return total
}

func countChunkPositions(chunks mtmd.InputChunks) uint32 {
	var total uint32
	size := mtmd.InputChunksSize(chunks)
	for i := uint32(0); i < size; i++ {
		chunk := mtmd.InputChunksGet(chunks, i)
		pos := uint32(mtmd.InputChunkGetNPos(chunk))
		if pos == 0 {
			pos = mtmd.InputChunkGetNTokens(chunk)
		}
		total += pos
	}
	return total
}

func nextBatchSize(required, current, configured uint32) uint32 {
	desired := required
	if configured > desired {
		desired = configured
	}
	if desired <= current {
		return current
	}
	const minBatch = 2048
	if desired < minBatch {
		desired = minBatch
	}
	const step = 256
	if remainder := desired % step; remainder != 0 {
		desired += step - remainder
	}
	return desired
}

func nextContextSize(required, current, configured uint32) uint32 {
	desired := required
	if configured > desired {
		desired = configured
	}
	if desired <= current {
		return current
	}
	const minCtx = 2048
	if desired < minCtx {
		desired = minCtx
	}
	const step = 256
	if remainder := desired % step; remainder != 0 {
		desired += step - remainder
	}
	return desired
}

func withSilentLlamaLogs(fn func() error) error {
	llama.LogSet(llama.LogSilent())
	if mtmdLogReady {
		mtmd.LogSet(llama.LogSilent())
	}
	defer configureLlamaLogging()
	return fn()
}

func configureLlamaLogging() {
	mode := strings.TrimSpace(strings.ToLower(os.Getenv("ND_LLAMA_LOG")))
	switch mode {
	case "1", "true", "yes", "normal", "info", "debug":
		llama.LogSet(llama.LogNormal)
		if mtmdLogReady {
			mtmd.LogSet(llama.LogNormal)
		}
	default:
		llama.LogSet(llama.LogSilent())
		if mtmdLogReady {
			mtmd.LogSet(llama.LogSilent())
		}
	}
}

func preloadGGMLDeps(path string) error {
	deps := []string{
		"ggml-base",
		"ggml-cpu",
		"ggml-cuda",
		"ggml-rpc",
	}
	for _, dep := range deps {
		filename := loader.GetLibraryFilename(path, dep)
		log.Debug(context.Background(), "Checking ggml dependency", "lib", dep, "path", filename)
		if _, err := os.Stat(filename); err != nil {
			if os.IsNotExist(err) {
				log.Debug(context.Background(), "Skipping missing ggml dependency", "lib", dep, "path", filename)
				continue
			}
			log.Warn(context.Background(), "Failed to stat ggml dependency", "lib", dep, "path", filename, "err", err)
			return err
		}
		log.Info(context.Background(), "Preloading ggml dependency", "lib", dep, "path", filename)
		if _, err := loader.LoadLibrary(path, dep); err != nil {
			return fmt.Errorf("preload %s: %w", dep, err)
		}
	}
	return nil
}
