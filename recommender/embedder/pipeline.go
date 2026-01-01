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
	ModelLyrics      = "lyrics"
	ModelDescription = "description"
	ModelFlamingo    = "flamingo"
)

// PipelineConfig configures batch processing behavior.
type PipelineConfig struct {
	BatchTimeout      time.Duration // Time to wait before processing partial batch
	BatchSize         int           // Maximum items per batch
	EnableLyrics      bool          // Enable lyrics embedding
	EnableDescription bool          // Enable audio description embedding
	EnableFlamingo    bool          // Enable flamingo audio embedding
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
	Lyrics    string // Lyrics text if available

	// Stage 1 results: lyrics -> lyrics embedding
	LyricsEmbedding []float64

	// Stage 2 results: audio -> description -> description embedding
	Description          string
	DescriptionEmbedding []float64

	// Stage 3 results: audio -> flamingo embedding
	FlamingoEmbedding []float64

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

	// Stage 1: Lyrics embeddings (if enabled and lyrics available)
	if p.config.EnableLyrics {
		p.stageLyricsEmbedding(ctx, batch)
	}

	// Stage 2: Audio description -> description embedding (if enabled)
	if p.config.EnableDescription {
		p.stageDescriptionEmbedding(ctx, batch)
	}

	// Stage 3: Flamingo audio embedding (if enabled)
	if p.config.EnableFlamingo {
		p.stageFlamingoEmbedding(ctx, batch)
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

func (p *Pipeline) stageLyricsEmbedding(ctx context.Context, batch []*TrackContext) {
	// Collect tracks with lyrics
	requests := make([]llamacpp.TextEmbedRequest, 0, len(batch))
	indices := make([]int, 0, len(batch))

	for i, track := range batch {
		if track.Error != nil || track.Lyrics == "" {
			continue
		}
		requests = append(requests, llamacpp.TextEmbedRequest{
			Text: track.Lyrics,
		})
		indices = append(indices, i)
	}

	if len(requests) == 0 {
		return
	}

	log.Debug(ctx, "Stage: Lyrics embedding", "count", len(requests))

	responses, err := p.llamaClient.EmbedTextBatch(ctx, requests)
	if err != nil {
		// Non-fatal: continue without lyrics embeddings
		log.Warn(ctx, "Lyrics embedding failed (non-fatal)", err)
		return
	}

	for j, resp := range responses {
		if j >= len(indices) {
			break
		}
		idx := indices[j]
		if resp.Error != "" {
			log.Warn(ctx, "Lyrics embedding failed", "track", batch[idx].TrackName, "error", resp.Error)
			continue
		}
		batch[idx].LyricsEmbedding = resp.Embedding
	}
}

func (p *Pipeline) stageDescriptionEmbedding(ctx context.Context, batch []*TrackContext) {
	// First, get audio descriptions
	descRequests := make([]llamacpp.AudioDescribeRequest, 0, len(batch))
	descIndices := make([]int, 0, len(batch))

	for i, track := range batch {
		if track.Error != nil {
			continue
		}
		descRequests = append(descRequests, llamacpp.AudioDescribeRequest{
			AudioPath: track.FilePath,
		})
		descIndices = append(descIndices, i)
	}

	if len(descRequests) == 0 {
		return
	}

	log.Debug(ctx, "Stage: Audio description", "count", len(descRequests))

	descResponses, err := p.llamaClient.DescribeAudioBatch(ctx, descRequests)
	if err != nil {
		log.Warn(ctx, "Audio description failed (non-fatal)", err)
		return
	}

	// Assign descriptions
	for j, resp := range descResponses {
		if j >= len(descIndices) {
			break
		}
		idx := descIndices[j]
		if resp.Error != "" {
			log.Warn(ctx, "Audio description failed", "track", batch[idx].TrackName, "error", resp.Error)
			continue
		}
		batch[idx].Description = resp.Description
	}

	// Now embed the descriptions
	embedRequests := make([]llamacpp.TextEmbedRequest, 0, len(batch))
	embedIndices := make([]int, 0, len(batch))

	for i, track := range batch {
		if track.Error != nil || track.Description == "" {
			continue
		}
		embedRequests = append(embedRequests, llamacpp.TextEmbedRequest{
			Text: track.Description,
		})
		embedIndices = append(embedIndices, i)
	}

	if len(embedRequests) == 0 {
		return
	}

	log.Debug(ctx, "Stage: Description text embedding", "count", len(embedRequests))

	embedResponses, err := p.llamaClient.EmbedTextBatch(ctx, embedRequests)
	if err != nil {
		log.Warn(ctx, "Description embedding failed (non-fatal)", err)
		return
	}

	for j, resp := range embedResponses {
		if j >= len(embedIndices) {
			break
		}
		idx := embedIndices[j]
		if resp.Error != "" {
			log.Warn(ctx, "Description embedding failed", "track", batch[idx].TrackName, "error", resp.Error)
			continue
		}
		batch[idx].DescriptionEmbedding = resp.Embedding
	}
}

func (p *Pipeline) stageFlamingoEmbedding(ctx context.Context, batch []*TrackContext) {
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

	log.Debug(ctx, "Stage: Flamingo audio embedding", "count", len(requests))

	responses, err := p.llamaClient.EmbedAudioBatch(ctx, requests)
	if err != nil {
		log.Warn(ctx, "Flamingo embedding failed (non-fatal)", err)
		return
	}

	for j, resp := range responses {
		if j >= len(indices) {
			break
		}
		idx := indices[j]
		if resp.Error != "" {
			log.Warn(ctx, "Flamingo embedding failed", "track", batch[idx].TrackName, "error", resp.Error)
			continue
		}
		batch[idx].FlamingoEmbedding = resp.Embedding
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
	var lyricsData, descData, flamingoData []EmbeddingData

	for _, track := range batch {
		if track.Error != nil {
			continue
		}

		// Store lyrics embedding
		if len(track.LyricsEmbedding) > 0 {
			lyricsData = append(lyricsData, EmbeddingData{
				Name:      track.TrackName,
				Embedding: track.LyricsEmbedding,
				ModelID:   ModelLyrics,
			})
		}

		// Store description embedding (with description text)
		if len(track.DescriptionEmbedding) > 0 {
			descData = append(descData, EmbeddingData{
				Name:        track.TrackName,
				Embedding:   track.DescriptionEmbedding,
				Description: track.Description,
				ModelID:     ModelDescription,
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
	if len(lyricsData) > 0 {
		milvusData := toMilvusData(lyricsData)
		if err := p.milvus.Upsert(ctx, milvus.CollectionLyrics, milvusData); err != nil {
			log.Error(ctx, "Failed to store lyrics embeddings", err, "count", len(lyricsData))
		} else {
			log.Debug(ctx, "Stored lyrics embeddings", "count", len(lyricsData))
		}
	}

	if len(descData) > 0 {
		milvusData := toMilvusData(descData)
		if err := p.milvus.Upsert(ctx, milvus.CollectionDescription, milvusData); err != nil {
			log.Error(ctx, "Failed to store description embeddings", err, "count", len(descData))
		} else {
			log.Debug(ctx, "Stored description embeddings", "count", len(descData))
		}
	}

	if len(flamingoData) > 0 {
		milvusData := toMilvusData(flamingoData)
		if err := p.milvus.Upsert(ctx, milvus.CollectionFlamingo, milvusData); err != nil {
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
