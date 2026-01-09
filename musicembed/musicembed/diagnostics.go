package musicembed

import (
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/hybridgroup/yzma/pkg/llama"
	"github.com/hybridgroup/yzma/pkg/loader"
)

var (
	diagOnce sync.Once
)

func diagnosticsEnabled() bool {
	mode := strings.TrimSpace(strings.ToLower(os.Getenv("ND_LLAMA_LOG")))
	switch mode {
	case "1", "true", "yes", "normal", "info", "debug":
		return true
	default:
		return false
	}
}

func diagf(format string, args ...any) {
	if !diagnosticsEnabled() {
		return
	}
	log.Printf(format, args...)
}

func logDiagnostics(cfg Config, libraryPath string) {
	if !diagnosticsEnabled() {
		return
	}

	diagOnce.Do(func() {
		diagf("musicembed: diagnostics enabled")
		diagf("musicembed: libraryPath=%q", libraryPath)
		diagf("musicembed: models embed=%q music=%q mmproj=%q", cfg.EmbeddingModelFile, cfg.ModelFile, cfg.MmprojFile)
		diagf("musicembed: gpu layers=%d main gpu=%d use gpu=%t", cfg.GPULayers, cfg.MainGPU, cfg.UseGPU)
		diagf("musicembed: context size=%d batch size=%d generation margin=%d max output=%d threads=%d",
			cfg.ContextSize, cfg.BatchSize, cfg.GenerationMargin, cfg.MaxOutputTokens, cfg.Threads)

		diagf("musicembed: env CUDA_VISIBLE_DEVICES=%q NVIDIA_VISIBLE_DEVICES=%q YZMA_LIB=%q LD_LIBRARY_PATH=%q",
			os.Getenv("CUDA_VISIBLE_DEVICES"),
			os.Getenv("NVIDIA_VISIBLE_DEVICES"),
			os.Getenv("YZMA_LIB"),
			os.Getenv("LD_LIBRARY_PATH"),
		)

		for _, dep := range []string{"ggml-base", "ggml-cpu", "ggml-cuda", "ggml-rpc"} {
			filename := loader.GetLibraryFilename(libraryPath, dep)
			if _, err := os.Stat(filename); err != nil {
				diagf("musicembed: ggml lib %s missing: %s", dep, filename)
			} else {
				diagf("musicembed: ggml lib %s present: %s", dep, filename)
			}
		}

		diagf("musicembed: llama supports gpu offload=%t max devices=%d", llama.SupportsGpuOffload(), llama.MaxDevices())

		deviceCount := llama.GGMLBackendDeviceCount()
		diagf("musicembed: ggml backend device count=%d", deviceCount)
		for i := uint64(0); i < deviceCount; i++ {
			device := llama.GGMLBackendDeviceGet(i)
			name := llama.GGMLBackendDeviceName(device)
			diagf("musicembed: ggml device %d name=%q", i, name)
		}

		sysInfo := strings.TrimSpace(llama.PrintSystemInfo())
		if sysInfo != "" {
			diagf("musicembed: llama system info:\n%s", sysInfo)
		}

		if abs, err := filepath.Abs(libraryPath); err == nil && abs != libraryPath {
			diagf("musicembed: library path resolved to %q", abs)
		}
	})
}
