package embedder

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/navidrome/navidrome/log"
	"github.com/navidrome/navidrome/recommender/llamacpp"
	"github.com/navidrome/navidrome/recommender/milvus"
)

// Model identifiers.
const (
	ModelMuQ      = "muq"
	ModelQwen3    = "qwen3"
	ModelFlamingo = "flamingo"
)

// Collection names for Milvus.
const (
	CollectionFlamingoAudio = "flamingo_audio_embedding"
)

// PipelineConfig configures batch processing behavior.
type PipelineConfig struct {
	BatchTimeout       time.Duration // Time to wait before processing partial batch
	BatchSize          int           // Maximum items per batch
	EnableDescriptions bool          // Enable stage 2 (audio -> description)
	EnableTextEmbed    bool          // Enable stage 3 (description -> text embedding)
}

// TrackContext holds intermediate state for a track being processed.
type TrackContext struct {
	// Input
	FilePath  string
	TrackName string
	TrackID   string
	Artist    string
	Title     string
	Album     string

	// Stage 1 results: audio -> audio embedding
	AudioEmbedding []float64
	ModelID        string

	// Stage 2 results: audio -> text description + flamingo audio embedding
	Description       string
	FlamingoEmbedding []float64

	// Stage 3 results: text description -> text embedding
	TextEmbedding []float64

	// Status
	Error error
	Done  chan struct{}
}

// Pipeline orchestrates the three-stage embedding process with batching.
type Pipeline struct {
	config      PipelineConfig
	llamaClient *llamacpp.Client
	milvus      *milvus.Client

	mu         sync.Mutex
	pending    []*TrackContext
	timer      *time.Timer
	processing bool
	closed     bool
	wg         sync.WaitGroup
}

// NewPipeline creates a new Pipeline.
func NewPipeline(cfg PipelineConfig, llama *llamacpp.Client, milvus *milvus.Client) *Pipeline {
	if cfg.BatchTimeout <= 0 {
		cfg.BatchTimeout = 5 * time.Second
	}
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 50
	}

	return &Pipeline{
		config:      cfg,
		llamaClient: llama,
		milvus:      milvus,
	}
}

// Enqueue adds a track to the processing queue.
func (p *Pipeline) Enqueue(ctx context.Context, track *TrackContext) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.closed {
		return fmt.Errorf("pipeline is closed")
	}

	p.pending = append(p.pending, track)

	// Start timer on first item
	if len(p.pending) == 1 && p.timer == nil {
		p.timer = time.AfterFunc(p.config.BatchTimeout, p.triggerBatch)
	}

	// Check threshold
	if len(p.pending) >= p.config.BatchSize {
		p.triggerBatchLocked()
	}

	return nil
}

// Flush forces immediate processing of pending items.
func (p *Pipeline) Flush(ctx context.Context) error {
	p.mu.Lock()
	if len(p.pending) > 0 && !p.processing {
		p.triggerBatchLocked()
	}
	p.mu.Unlock()

	// Wait for current batch to complete
	p.wg.Wait()
	return nil
}

// Close stops the pipeline and releases resources.
func (p *Pipeline) Close() {
	p.mu.Lock()
	p.closed = true
	if p.timer != nil {
		p.timer.Stop()
		p.timer = nil
	}
	// Signal all pending tracks as cancelled
	for _, track := range p.pending {
		track.Error = fmt.Errorf("pipeline closed")
		close(track.Done)
	}
	p.pending = nil
	p.mu.Unlock()

	// Wait for any in-progress batch
	p.wg.Wait()
}

func (p *Pipeline) triggerBatch() {
	p.mu.Lock()
	p.triggerBatchLocked()
	p.mu.Unlock()
}

func (p *Pipeline) triggerBatchLocked() {
	if p.processing || len(p.pending) == 0 || p.closed {
		return
	}

	if p.timer != nil {
		p.timer.Stop()
		p.timer = nil
	}

	p.processing = true
	batch := p.pending
	p.pending = nil

	p.wg.Add(1)
	go p.processBatch(batch)
}

func (p *Pipeline) processBatch(batch []*TrackContext) {
	defer func() {
		p.wg.Done()
		p.mu.Lock()
		p.processing = false
		// Schedule next batch if items accumulated while processing
		if len(p.pending) > 0 && !p.closed {
			p.timer = time.AfterFunc(p.config.BatchTimeout, p.triggerBatch)
		}
		p.mu.Unlock()
	}()

	ctx := context.Background()

	log.Info(ctx, "Processing embedding batch", "count", len(batch))
	start := time.Now()

	// Stage 1: Audio embeddings for ALL tracks
	p.stage1AudioEmbedding(ctx, batch)

	// Stage 2: Descriptions for ALL tracks (if enabled)
	if p.config.EnableDescriptions {
		p.stage2Description(ctx, batch)
	}

	// Stage 3: Text embeddings for ALL descriptions (if enabled)
	if p.config.EnableTextEmbed {
		p.stage3TextEmbedding(ctx, batch)
	}

	// Store results in Milvus
	p.storeResults(ctx, batch)

	log.Info(ctx, "Batch processing complete",
		"count", len(batch),
		"duration", time.Since(start),
	)

	// Signal completion for all tracks
	for _, track := range batch {
		close(track.Done)
	}
}

func (p *Pipeline) stage1AudioEmbedding(ctx context.Context, batch []*TrackContext) {
	// Collect valid tracks
	requests := make([]llamacpp.AudioEmbedRequest, 0, len(batch))
	indices := make([]int, 0, len(batch))

	for i, track := range batch {
		if track.Error != nil {
			continue
		}
		requests = append(requests, llamacpp.AudioEmbedRequest{
			AudioPath: track.FilePath,
		})
		indices = append(indices, i)
	}

	if len(requests) == 0 {
		return
	}

	log.Debug(ctx, "Stage 1: Audio embedding", "count", len(requests))

	// Call llama.cpp batch endpoint
	responses, err := p.llamaClient.EmbedAudioBatch(ctx, requests)
	if err != nil {
		log.Error(ctx, "Stage 1 failed", err)
		for _, idx := range indices {
			batch[idx].Error = fmt.Errorf("stage1 audio embedding: %w", err)
		}
		return
	}

	// Assign results
	for j, resp := range responses {
		if j >= len(indices) {
			break
		}
		idx := indices[j]
		if resp.Error != "" {
			batch[idx].Error = fmt.Errorf("audio embedding: %s", resp.Error)
			continue
		}
		batch[idx].AudioEmbedding = resp.Embedding
		batch[idx].ModelID = resp.ModelID
	}
}

func (p *Pipeline) stage2Description(ctx context.Context, batch []*TrackContext) {
	// Collect tracks that passed stage 1
	requests := make([]llamacpp.AudioDescribeRequest, 0, len(batch))
	indices := make([]int, 0, len(batch))

	for i, track := range batch {
		if track.Error != nil {
			continue
		}
		requests = append(requests, llamacpp.AudioDescribeRequest{
			AudioPath: track.FilePath,
		})
		indices = append(indices, i)
	}

	if len(requests) == 0 {
		return
	}

	log.Debug(ctx, "Stage 2: Audio description", "count", len(requests))

	responses, err := p.llamaClient.DescribeAudioBatch(ctx, requests)
	if err != nil {
		// Non-fatal: continue without descriptions
		log.Warn(ctx, "Stage 2 failed (non-fatal)", err)
		return
	}

	for j, resp := range responses {
		if j >= len(indices) {
			break
		}
		idx := indices[j]
		if resp.Error != "" {
			// Non-fatal for individual items
			log.Warn(ctx, "Audio description failed", "track", batch[idx].TrackName, "error", resp.Error)
			continue
		}
		batch[idx].Description = resp.Description
		batch[idx].FlamingoEmbedding = resp.AudioEmbedding
	}
}

func (p *Pipeline) stage3TextEmbedding(ctx context.Context, batch []*TrackContext) {
	// Collect tracks with descriptions
	requests := make([]llamacpp.TextEmbedRequest, 0, len(batch))
	indices := make([]int, 0, len(batch))

	for i, track := range batch {
		if track.Error != nil || track.Description == "" {
			continue
		}
		requests = append(requests, llamacpp.TextEmbedRequest{
			Text: track.Description,
		})
		indices = append(indices, i)
	}

	if len(requests) == 0 {
		return
	}

	log.Debug(ctx, "Stage 3: Text embedding", "count", len(requests))

	responses, err := p.llamaClient.EmbedTextBatch(ctx, requests)
	if err != nil {
		// Non-fatal: continue without text embeddings
		log.Warn(ctx, "Stage 3 failed (non-fatal)", err)
		return
	}

	for j, resp := range responses {
		if j >= len(indices) {
			break
		}
		idx := indices[j]
		if resp.Error != "" {
			log.Warn(ctx, "Text embedding failed", "track", batch[idx].TrackName, "error", resp.Error)
			continue
		}
		batch[idx].TextEmbedding = resp.Embedding
	}
}

// EmbeddingData for Milvus storage.
type EmbeddingData struct {
	Name        string
	Embedding   []float64
	Offset      float64
	ModelID     string
	Description string
}

func (p *Pipeline) storeResults(ctx context.Context, batch []*TrackContext) {
	// Group by collection type and batch insert
	var audioData, descData, flamingoData []EmbeddingData

	for _, track := range batch {
		if track.Error != nil {
			continue
		}

		// Store audio embedding
		if len(track.AudioEmbedding) > 0 {
			audioData = append(audioData, EmbeddingData{
				Name:      track.TrackName,
				Embedding: track.AudioEmbedding,
				ModelID:   track.ModelID,
			})
		}

		// Store text embedding (with description)
		if len(track.TextEmbedding) > 0 {
			descData = append(descData, EmbeddingData{
				Name:        track.TrackName,
				Embedding:   track.TextEmbedding,
				Description: track.Description,
				ModelID:     ModelQwen3,
			})
		}

		// Store flamingo audio embedding
		if len(track.FlamingoEmbedding) > 0 {
			flamingoData = append(flamingoData, EmbeddingData{
				Name:      track.TrackName,
				Embedding: track.FlamingoEmbedding,
				ModelID:   ModelFlamingo,
			})
		}
	}

	// Batch upsert to each collection
	if len(audioData) > 0 {
		milvusData := toMilvusData(audioData)
		if err := p.milvus.Upsert(ctx, CollectionEmbedding, milvusData); err != nil {
			log.Error(ctx, "Failed to store audio embeddings", err, "count", len(audioData))
		} else {
			log.Debug(ctx, "Stored audio embeddings", "count", len(audioData))
		}
	}

	if len(descData) > 0 {
		milvusData := toMilvusData(descData)
		if err := p.milvus.Upsert(ctx, CollectionDescriptionEmbedding, milvusData); err != nil {
			log.Error(ctx, "Failed to store description embeddings", err, "count", len(descData))
		} else {
			log.Debug(ctx, "Stored description embeddings", "count", len(descData))
		}
	}

	if len(flamingoData) > 0 {
		milvusData := toMilvusData(flamingoData)
		if err := p.milvus.Upsert(ctx, CollectionFlamingoAudio, milvusData); err != nil {
			log.Error(ctx, "Failed to store flamingo embeddings", err, "count", len(flamingoData))
		} else {
			log.Debug(ctx, "Stored flamingo embeddings", "count", len(flamingoData))
		}
	}
}

// toMilvusData converts local EmbeddingData to milvus.EmbeddingData.
func toMilvusData(data []EmbeddingData) []milvus.EmbeddingData {
	result := make([]milvus.EmbeddingData, len(data))
	for i, d := range data {
		result[i] = milvus.EmbeddingData{
			Name:        d.Name,
			Embedding:   d.Embedding,
			Offset:      d.Offset,
			ModelID:     d.ModelID,
			Description: d.Description,
		}
	}
	return result
}
