package musicembed

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/hybridgroup/yzma/pkg/llama"
	"github.com/hybridgroup/yzma/pkg/loader"
	"github.com/hybridgroup/yzma/pkg/mtmd"
)

var soundMarker = []byte("<sound>\x00")

// Client provides access to music embedding functionality.
type Client struct {
	config Config

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

// New creates a new Client and initializes all required libraries and models.
func New(config Config) (*Client, error) {
	absPath, err := filepath.Abs(config.LibraryPath)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve library path: %w", err)
	}

	if err := initLibraries(absPath); err != nil {
		return nil, fmt.Errorf("failed to initialize libraries: %w", err)
	}

	c := &Client{config: config}

	// Load embedding model
	embedParams := llama.ModelDefaultParams()
	embedParams.NGpuLayers = int32(config.GPULayers)
	embedParams.MainGpu = int32(config.MainGPU)

	c.embeddingModel, err = llama.ModelLoadFromFile(config.EmbeddingModelFile, embedParams)
	if err != nil {
		return nil, fmt.Errorf("failed to load embedding model: %w", err)
	}
	if c.embeddingModel == 0 {
		return nil, fmt.Errorf("failed to load embedding model from %s", config.EmbeddingModelFile)
	}

	// Load music model
	musicParams := llama.ModelDefaultParams()
	musicParams.NGpuLayers = int32(config.GPULayers)
	musicParams.MainGpu = int32(config.MainGPU)

	c.musicModel, err = llama.ModelLoadFromFile(config.ModelFile, musicParams)
	if err != nil {
		c.Close()
		return nil, fmt.Errorf("failed to load music model: %w", err)
	}
	if c.musicModel == 0 {
		c.Close()
		return nil, fmt.Errorf("failed to load music model from %s", config.ModelFile)
	}

	c.vocab = llama.ModelGetVocab(c.musicModel)
	c.chatTemplate = llama.ModelChatTemplate(c.musicModel, "")
	c.sampler = newMusicSampler(c.musicModel)
	if c.sampler == 0 {
		c.Close()
		return nil, fmt.Errorf("failed to create sampler")
	}

	// Initialize mmproj context
	mtmd.LogSet(llama.LogSilent())
	ctxParams := mtmd.ContextParamsDefault()
	ctxParams.UseGPU = config.UseGPU
	ctxParams.PrintTimings = false
	ctxParams.Threads = int32(config.Threads)
	ctxParams.MediaMarker = &soundMarker[0]
	ctxParams.FlashAttentionType = llama.FlashAttentionTypeAuto
	ctxParams.Warmup = false
	ctxParams.ImageMinTokens = 0
	ctxParams.ImageMaxTokens = 0

	c.mmprojCtx, err = mtmd.InitFromFile(config.MmprojFile, c.musicModel, ctxParams)
	if err != nil {
		c.Close()
		return nil, fmt.Errorf("failed to initialize mmproj context: %w", err)
	}

	return c, nil
}

// Close releases all resources held by the client.
func (c *Client) Close() {
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
	if c.embeddingModel != 0 {
		llama.ModelFree(c.embeddingModel)
		c.embeddingModel = 0
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
