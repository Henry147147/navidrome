// Package llamacpp provides a local llama.cpp client backed by the yzma bindings.
package llamacpp

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/hybridgroup/yzma/pkg/llama"
	"github.com/hybridgroup/yzma/pkg/mtmd"
	"github.com/navidrome/navidrome/log"
)

const (
	DefaultLibraryPath        = "./llama-lib"
	DefaultModelPath          = "./models/music-flamingo.gguf"
	DefaultAudioProjectorPath = "./models/mmproj-music-flamingo.gguf"
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
		TextModelPath:      DefaultModelPath,
		AudioModelPath:     DefaultModelPath,
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
	log.Debug(ctx, "EmbedAudioBatch called (stub)", "count", len(reqs))
	return nil, fmt.Errorf("%w: audio embedding", ErrNotImplemented)
}

func (b *localBackend) DescribeAudioBatch(ctx context.Context, reqs []AudioDescribeRequest) ([]AudioDescribeResponse, error) {
	log.Debug(ctx, "DescribeAudioBatch called (stub)", "count", len(reqs))
	return nil, fmt.Errorf("%w: audio description", ErrNotImplemented)
}

func (b *localBackend) EmbedTextBatch(ctx context.Context, reqs []TextEmbedRequest) ([]TextEmbedResponse, error) {
	log.Debug(ctx, "EmbedTextBatch called (stub)", "count", len(reqs))
	return nil, fmt.Errorf("%w: text embedding", ErrNotImplemented)
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
		params.PoolingType = llama.PoolingTypeMean
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
