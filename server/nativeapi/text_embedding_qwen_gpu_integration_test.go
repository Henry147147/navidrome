//go:build integration

package nativeapi

import (
	"bytes"
	"context"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/navidrome/navidrome/conf"
)

const (
	runQwenGPUEnv       = "RUN_QWEN8B_GPU_TEST"
	defaultQwenHFRepo   = "Qwen/Qwen3-Embedding-8B-GGUF"
	defaultQwenHFFile   = "Qwen3-Embedding-8B-Q4_K_M.gguf"
	defaultQwenPoolType = "last"
)

func TestQwen8BEmbeddingInferenceOnGPU(t *testing.T) {
	if os.Getenv(runQwenGPUEnv) != "1" {
		t.Skipf("set %s=1 to run GPU integration test", runQwenGPUEnv)
	}

	llamaServerBin, err := resolveLlamaServerBinary()
	if err != nil {
		t.Skipf("llama-server binary not found: %v", err)
	}

	port, err := getFreeTCPPort()
	if err != nil {
		t.Fatalf("failed to allocate free port: %v", err)
	}
	baseURL := fmt.Sprintf("http://127.0.0.1:%d", port)

	args := []string{
		"--host", "127.0.0.1",
		"--port", strconv.Itoa(port),
		"--embeddings",
		"--pooling", envOrDefault("QWEN_POOLING", defaultQwenPoolType),
		"--n-gpu-layers", envOrDefault("QWEN_GPU_LAYERS", "all"),
		"--ctx-size", envOrDefault("QWEN_CTX_SIZE", "8192"),
		"--ubatch-size", envOrDefault("QWEN_UBATCH_SIZE", "8192"),
		"--batch-size", envOrDefault("QWEN_BATCH_SIZE", "2048"),
		"--no-webui",
	}
	if modelPath := strings.TrimSpace(os.Getenv("QWEN_MODEL_PATH")); modelPath != "" {
		args = append(args, "--model", modelPath)
	} else {
		args = append(args,
			"--hf-repo", envOrDefault("QWEN_HF_REPO", defaultQwenHFRepo),
			"--hf-file", envOrDefault("QWEN_HF_FILE", defaultQwenHFFile),
		)
	}

	var logs bytes.Buffer
	cmd := exec.Command(llamaServerBin, args...) // #nosec G204 -- binary and args come from controlled test configuration
	cmd.Stdout = &logs
	cmd.Stderr = &logs
	libDir := filepath.Dir(llamaServerBin)
	cmd.Env = append(os.Environ(), "LD_LIBRARY_PATH="+prependToPathList(libDir, os.Getenv("LD_LIBRARY_PATH")))

	if err := cmd.Start(); err != nil {
		t.Fatalf("failed to start llama-server: %v", err)
	}

	waitCh := make(chan error, 1)
	go func() {
		waitCh <- cmd.Wait()
	}()

	t.Cleanup(func() {
		_ = stopProcess(cmd, waitCh)
	})

	if err := waitForEmbeddingServer(baseURL, waitCh, 20*time.Minute); err != nil {
		t.Fatalf("embedding server did not become healthy: %v\nlogs:\n%s", err, logs.String())
	}

	prevBase := conf.Server.Recommendations.BaseURL
	prevTextBase := conf.Server.Recommendations.TextBaseURL
	prevTimeout := conf.Server.Recommendations.Timeout
	conf.Server.Recommendations.BaseURL = baseURL
	conf.Server.Recommendations.TextBaseURL = baseURL
	conf.Server.Recommendations.Timeout = 5 * time.Minute
	t.Cleanup(func() {
		conf.Server.Recommendations.BaseURL = prevBase
		conf.Server.Recommendations.TextBaseURL = prevTextBase
		conf.Server.Recommendations.Timeout = prevTimeout
	})

	var router Router
	vec, err := router.getTextEmbedding(context.Background(), "melancholic synthwave with dreamy vocals", "qwen8b", 4096)
	if err != nil {
		t.Fatalf("failed to embed text: %v\nlogs:\n%s", err, logs.String())
	}
	if len(vec) == 0 {
		t.Fatalf("expected non-empty embedding")
	}
	// Qwen3-Embedding-8B emits 4096-dimensional vectors by default in llama.cpp.
	if len(vec) != 4096 {
		t.Fatalf("expected embedding dimension 4096, got %d", len(vec))
	}

	logText := logs.String()
	if !strings.Contains(strings.ToLower(logText), "cuda") {
		t.Fatalf("expected llama-server logs to show CUDA/GPU usage, logs:\n%s", logText)
	}
}

func resolveLlamaServerBinary() (string, error) {
	candidates := []string{
		strings.TrimSpace(os.Getenv("QWEN_LLAMA_SERVER_BIN")),
		filepath.Join("musicembed", "llama-lib", "llama-server"),
		filepath.Join("musicembed", "llama-lib", "llama-server.exe"),
	}
	for _, candidate := range candidates {
		if candidate == "" {
			continue
		}
		if _, err := os.Stat(candidate); err == nil {
			return candidate, nil
		}
	}
	if p, err := exec.LookPath("llama-server"); err == nil {
		return p, nil
	}
	return "", fmt.Errorf("set QWEN_LLAMA_SERVER_BIN or place llama-server in PATH")
}

func getFreeTCPPort() (int, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	defer l.Close()
	addr, ok := l.Addr().(*net.TCPAddr)
	if !ok {
		return 0, fmt.Errorf("unexpected listener type %T", l.Addr())
	}
	return addr.Port, nil
}

func waitForEmbeddingServer(baseURL string, waitCh <-chan error, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	client := &http.Client{Timeout: 3 * time.Second}
	for time.Now().Before(deadline) {
		select {
		case err := <-waitCh:
			return fmt.Errorf("llama-server exited before becoming ready: %w", err)
		default:
		}

		req, err := http.NewRequest(http.MethodGet, baseURL+"/health", nil)
		if err == nil {
			resp, reqErr := client.Do(req) // #nosec G107 -- test-only localhost URL
			if reqErr == nil {
				_ = resp.Body.Close()
				if resp.StatusCode == http.StatusOK {
					return nil
				}
			}
		}

		time.Sleep(2 * time.Second)
	}

	return fmt.Errorf("timed out after %s waiting for %s/health", timeout, baseURL)
}

func stopProcess(cmd *exec.Cmd, waitCh <-chan error) error {
	if cmd == nil || cmd.Process == nil {
		return nil
	}
	_ = cmd.Process.Signal(os.Interrupt)
	select {
	case err := <-waitCh:
		return err
	case <-time.After(10 * time.Second):
		_ = cmd.Process.Kill()
		select {
		case err := <-waitCh:
			return err
		case <-time.After(5 * time.Second):
			return fmt.Errorf("process did not terminate")
		}
	}
}

func envOrDefault(key string, fallback string) string {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return fallback
	}
	return value
}

func prependToPathList(prefix string, existing string) string {
	if strings.TrimSpace(prefix) == "" {
		return existing
	}
	if strings.TrimSpace(existing) == "" {
		return prefix
	}
	return prefix + ":" + existing
}
