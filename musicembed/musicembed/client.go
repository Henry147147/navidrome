package musicembed

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/hybridgroup/yzma/pkg/llama"
	"github.com/hybridgroup/yzma/pkg/loader"
	"github.com/hybridgroup/yzma/pkg/mtmd"
)

var soundMarker = []byte("<sound>\x00")

// Client provides access to music embedding functionality.
type Client struct {
	config Config

	mu          sync.Mutex
	activeModel modelKind
	closed      bool

	// Models
	embeddingModel llama.Model
	musicModel     llama.Model

	// Contexts
	mmprojCtx mtmd.Context

	// Vocab and sampler for description generation
	vocab   llama.Vocab
	sampler llama.Sampler

	// Chat template
	chatTemplate string
}

type modelKind int

const (
	modelNone modelKind = iota
	modelEmbedding
	modelMusic
)

// New creates a new Client and initializes all required libraries and models.
func New(config Config) (*Client, error) {
	absPath, err := filepath.Abs(config.LibraryPath)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve library path: %w", err)
	}

	if err := validateModelFiles(config); err != nil {
		return nil, err
	}

	if err := initLibraries(absPath); err != nil {
		return nil, fmt.Errorf("failed to initialize libraries: %w", err)
	}

	return &Client{config: config}, nil
}

// Close releases all resources held by the client.
func (c *Client) Close() {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return
	}
	c.closed = true
	c.unloadMusicLocked()
	c.unloadEmbeddingLocked()
	c.activeModel = modelNone
}

func (c *Client) withEmbeddingModel(fn func() error) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.ensureEmbeddingModelLocked(); err != nil {
		return err
	}
	return fn()
}

func (c *Client) withMusicModel(fn func() error) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.ensureMusicModelLocked(); err != nil {
		return err
	}
	return fn()
}

func (c *Client) ensureEmbeddingModelLocked() error {
	if c.closed {
		return fmt.Errorf("musicembed client is closed")
	}
	if c.activeModel == modelEmbedding && c.embeddingModel != 0 {
		return nil
	}

	c.unloadMusicLocked()

	embedParams := llama.ModelDefaultParams()
	embedParams.NGpuLayers = int32(c.config.GPULayers)
	embedParams.MainGpu = int32(c.config.MainGPU)

	model, err := llama.ModelLoadFromFile(c.config.EmbeddingModelFile, embedParams)
	if err != nil {
		return fmt.Errorf("failed to load embedding model: %w", err)
	}
	if model == 0 {
		return fmt.Errorf("failed to load embedding model from %s", c.config.EmbeddingModelFile)
	}

	c.embeddingModel = model
	c.activeModel = modelEmbedding
	return nil
}

func (c *Client) ensureMusicModelLocked() error {
	if c.closed {
		return fmt.Errorf("musicembed client is closed")
	}
	if c.activeModel == modelMusic && c.musicModel != 0 && c.mmprojCtx != 0 && c.sampler != 0 {
		return nil
	}

	c.unloadEmbeddingLocked()

	musicParams := llama.ModelDefaultParams()
	musicParams.NGpuLayers = int32(c.config.GPULayers)
	musicParams.MainGpu = int32(c.config.MainGPU)

	model, err := llama.ModelLoadFromFile(c.config.ModelFile, musicParams)
	if err != nil {
		return fmt.Errorf("failed to load music model: %w", err)
	}
	if model == 0 {
		return fmt.Errorf("failed to load music model from %s", c.config.ModelFile)
	}

	vocab := llama.ModelGetVocab(model)
	chatTemplate := llama.ModelChatTemplate(model, "")
	sampler := newMusicSampler(model)
	if sampler == 0 {
		llama.ModelFree(model)
		return fmt.Errorf("failed to create sampler")
	}

	mtmd.LogSet(llama.LogSilent())
	ctxParams := mtmd.ContextParamsDefault()
	ctxParams.UseGPU = c.config.UseGPU
	ctxParams.PrintTimings = false
	ctxParams.Threads = int32(c.config.Threads)
	ctxParams.MediaMarker = &soundMarker[0]
	ctxParams.FlashAttentionType = llama.FlashAttentionTypeAuto
	ctxParams.Warmup = false
	ctxParams.ImageMinTokens = 0
	ctxParams.ImageMaxTokens = 0

	mmprojCtx, err := mtmd.InitFromFile(c.config.MmprojFile, model, ctxParams)
	if err != nil {
		llama.SamplerFree(sampler)
		llama.ModelFree(model)
		return fmt.Errorf("failed to initialize mmproj context: %w", err)
	}

	c.musicModel = model
	c.vocab = vocab
	c.chatTemplate = chatTemplate
	c.sampler = sampler
	c.mmprojCtx = mmprojCtx
	c.activeModel = modelMusic
	return nil
}

func (c *Client) unloadEmbeddingLocked() {
	if c.embeddingModel != 0 {
		llama.ModelFree(c.embeddingModel)
		c.embeddingModel = 0
	}
	if c.activeModel == modelEmbedding {
		c.activeModel = modelNone
	}
}

func (c *Client) unloadMusicLocked() {
	if c.mmprojCtx != 0 {
		mtmd.Free(c.mmprojCtx)
		c.mmprojCtx = 0
	}
	if c.sampler != 0 {
		llama.SamplerFree(c.sampler)
		c.sampler = 0
	}
	if c.musicModel != 0 {
		llama.ModelFree(c.musicModel)
		c.musicModel = 0
	}
	c.vocab = 0
	c.chatTemplate = ""
	if c.activeModel == modelMusic {
		c.activeModel = modelNone
	}
}

// initLibraries loads all required native libraries.
func initLibraries(absPath string) error {
	_ = os.Setenv("YZMA_LIB", absPath)

	if err := ensureLibraryPath(absPath); err != nil {
		return err
	}

	if err := preloadGGMLDeps(absPath); err != nil {
		return fmt.Errorf("failed to load ggml dependencies: %w", err)
	}

	if err := llama.Load(absPath); err != nil {
		return fmt.Errorf("failed to load llama library: %w", err)
	}

	if err := mtmd.Load(absPath); err != nil {
		return fmt.Errorf("failed to load mtmd library: %w", err)
	}

	if err := loadMTMDExtensions(absPath); err != nil {
		return fmt.Errorf("failed to load mtmd extensions: %w", err)
	}

	llama.LogSet(llama.LogSilent())
	llama.Init()

	return nil
}

func validateModelFiles(config Config) error {
	if config.EmbeddingModelFile == "" {
		return fmt.Errorf("embedding model file path is required")
	}
	if config.ModelFile == "" {
		return fmt.Errorf("music model file path is required")
	}
	if config.MmprojFile == "" {
		return fmt.Errorf("mmproj file path is required")
	}
	if _, err := os.Stat(config.EmbeddingModelFile); err != nil {
		return fmt.Errorf("embedding model file not found: %w", err)
	}
	if _, err := os.Stat(config.ModelFile); err != nil {
		return fmt.Errorf("music model file not found: %w", err)
	}
	if _, err := os.Stat(config.MmprojFile); err != nil {
		return fmt.Errorf("mmproj file not found: %w", err)
	}
	return nil
}

func ensureLibraryPath(path string) error {
	if path == "" {
		return nil
	}

	envKey := "LD_LIBRARY_PATH"
	current := os.Getenv(envKey)
	parts := strings.Split(current, string(os.PathListSeparator))
	for _, part := range parts {
		if part == path {
			return nil
		}
	}

	if current == "" {
		return os.Setenv(envKey, path)
	}
	return os.Setenv(envKey, path+string(os.PathListSeparator)+current)
}

func preloadGGMLDeps(path string) error {
	deps := []string{"ggml-base", "ggml-cpu", "ggml-cuda"}

	for _, dep := range deps {
		filename := loader.GetLibraryFilename(path, dep)
		if _, err := os.Stat(filename); err != nil {
			if os.IsNotExist(err) {
				continue
			}
			return err
		}
		if _, err := loader.LoadLibrary(path, dep); err != nil {
			return fmt.Errorf("preload %s: %w", dep, err)
		}
	}
	return nil
}

func newMusicSampler(model llama.Model) llama.Sampler {
	params := llama.DefaultSamplerParams()
	params.Temp = 0.8
	params.TopK = 40
	params.TopP = 0.9
	params.TypP = 0.95
	params.PenaltyLastN = -1
	params.PenaltyRepeat = 1.25
	params.PenaltyFreq = 0.2
	params.PenaltyPresent = 0.2

	samplers := []llama.SamplerType{
		llama.SamplerTypePenalties,
		llama.SamplerTypeTopK,
		llama.SamplerTypeTypicalP,
		llama.SamplerTypeTopP,
		llama.SamplerTypeTemperature,
	}

	return llama.NewSampler(model, samplers, params)
}

func applyChatTemplate(add bool, template string, messages []llama.ChatMessage) string {
	buf := make([]byte, 1024)
	length := llama.ChatApplyTemplate(template, messages, add, buf)
	return string(buf[:length])
}
