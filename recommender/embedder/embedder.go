// Package embedder provides the three-stage embedding pipeline for audio files.
package embedder

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/navidrome/navidrome/log"
	"github.com/navidrome/navidrome/recommender/milvus"
)

// Config holds embedder configuration.
type Config struct {
	BatchTimeout      time.Duration
	BatchSize         int
	EnableLyrics      bool // Enable lyrics text embedding
	EnableDescription bool // Enable audio description embedding
	EnableFlamingo    bool // Enable flamingo audio embedding
}

// EmbedRequest contains all information needed to embed a track.
type EmbedRequest struct {
	FilePath  string
	TrackName string
	TrackID   string
	Artist    string
	Title     string
	Album     string
	Lyrics    string
}

// EmbedResult contains the results of embedding a track.
type EmbedResult struct {
	TrackName            string
	LyricsEmbedding      []float64
	DescriptionEmbedding []float64
	FlamingoEmbedding    []float64
	Description          string
}

// StatusRequest contains information for checking embedding status.
type StatusRequest struct {
	TrackID        string
	Artist         string
	Title          string
	Album          string
	AlternateNames []string
}

// StatusResult contains embedding status information.
type StatusResult struct {
	Embedded          bool
	HasDescription    bool
	HasAudioEmbedding bool
	CanonicalName     string
}

// MusicClient describes the music embedding client behavior used by the embedder.
type MusicClient interface {
	EmbedText(text string) ([]float32, error)
	EmbedAudio(path string) ([]float32, error)
	GenerateDescription(path string) (string, error)
	Close()
}

// VectorStore describes the vector database operations used by the embedder.
type VectorStore interface {
	Upsert(ctx context.Context, collection string, data []milvus.EmbeddingData) error
	Exists(ctx context.Context, collection string, names []string) (map[string]bool, error)
}

// Embedder implements the embedding pipeline using musicembed and Milvus.
type Embedder struct {
	config      Config
	musicClient MusicClient
	vectorStore VectorStore
	mu          sync.RWMutex
	closed      bool
	batchCh     chan batchItem
	stopCh      chan struct{}
}

// New creates a new Embedder with the given configuration and clients.
func New(cfg Config, musicClient MusicClient, store VectorStore) *Embedder {
	e := &Embedder{
		config:      cfg,
		musicClient: musicClient,
		vectorStore: store,
	}
	e.startBatcher()
	return e
}

// EmbedAudio processes an audio file through all enabled stages.
func (e *Embedder) EmbedAudio(ctx context.Context, req EmbedRequest) (*EmbedResult, error) {
	if !e.batchingEnabled() {
		return e.processSingle(ctx, req)
	}

	response := make(chan batchResult, 1)
	item := batchItem{ctx: ctx, req: req, resp: response}
	if err := e.enqueueBatch(item); err != nil {
		return nil, err
	}

	select {
	case result := <-response:
		return result.result, result.err
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// EmbedText generates text embeddings directly (for query-time use).
func (e *Embedder) EmbedText(ctx context.Context, text string) ([]float64, error) {
	e.mu.RLock()
	if e.closed {
		e.mu.RUnlock()
		return nil, fmt.Errorf("embedder is closed")
	}
	musicClient := e.musicClient
	e.mu.RUnlock()

	if musicClient == nil {
		return nil, fmt.Errorf("music embedder is not configured")
	}
	if text == "" {
		return nil, fmt.Errorf("text is required")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	embedding, err := musicClient.EmbedText(text)
	if err != nil {
		return nil, fmt.Errorf("embed text: %w", err)
	}

	return float32sToFloat64s(embedding), nil
}

// CheckStatus checks if embeddings already exist for a track.
func (e *Embedder) CheckStatus(ctx context.Context, req StatusRequest) (*StatusResult, error) {
	e.mu.RLock()
	if e.closed {
		e.mu.RUnlock()
		return nil, fmt.Errorf("embedder is closed")
	}
	store := e.vectorStore
	e.mu.RUnlock()

	if store == nil {
		return nil, fmt.Errorf("vector store is not configured")
	}

	// Build list of possible names to check
	names := buildPossibleNames(req)
	if len(names) == 0 {
		return &StatusResult{}, nil
	}

	// Check existence in each collection
	lyricsExists, err := store.Exists(ctx, milvus.CollectionLyrics, names)
	if err != nil {
		return nil, fmt.Errorf("check lyrics embeddings: %w", err)
	}

	descExists, err := store.Exists(ctx, milvus.CollectionDescription, names)
	if err != nil {
		return nil, fmt.Errorf("check description embeddings: %w", err)
	}

	flamingoExists, err := store.Exists(ctx, milvus.CollectionFlamingo, names)
	if err != nil {
		return nil, fmt.Errorf("check flamingo embeddings: %w", err)
	}

	// Find the canonical name (first match found)
	var canonicalNameResult string
	var hasLyrics, hasDesc, hasFlamingo bool
	for _, name := range names {
		if lyricsExists[name] {
			canonicalNameResult = name
			hasLyrics = true
			break
		}
	}
	for _, name := range names {
		if descExists[name] {
			if canonicalNameResult == "" {
				canonicalNameResult = name
			}
			hasDesc = true
			break
		}
	}
	for _, name := range names {
		if flamingoExists[name] {
			if canonicalNameResult == "" {
				canonicalNameResult = name
			}
			hasFlamingo = true
			break
		}
	}

	return &StatusResult{
		Embedded:          hasLyrics || hasDesc || hasFlamingo,
		HasAudioEmbedding: hasFlamingo,
		HasDescription:    hasDesc,
		CanonicalName:     canonicalNameResult,
	}, nil
}

// FlushBatch forces processing of any pending batch items.
func (e *Embedder) FlushBatch(ctx context.Context) error {
	e.mu.RLock()
	if e.closed {
		e.mu.RUnlock()
		return fmt.Errorf("embedder is closed")
	}
	e.mu.RUnlock()
	return nil
}

// Close releases resources.
func (e *Embedder) Close() error {
	e.mu.Lock()
	if e.closed {
		e.mu.Unlock()
		return nil
	}
	e.closed = true
	musicClient := e.musicClient
	stopCh := e.stopCh
	e.mu.Unlock()

	if stopCh != nil {
		close(stopCh)
	}
	if musicClient != nil {
		musicClient.Close()
	}
	return nil
}

type batchItem struct {
	ctx  context.Context
	req  EmbedRequest
	resp chan batchResult
}

type batchResult struct {
	result *EmbedResult
	err    error
}

func (e *Embedder) batchingEnabled() bool {
	return e.config.BatchSize > 1
}

func (e *Embedder) startBatcher() {
	if !e.batchingEnabled() {
		return
	}
	e.batchCh = make(chan batchItem, e.config.BatchSize)
	e.stopCh = make(chan struct{})
	go e.runBatcher()
}

func (e *Embedder) enqueueBatch(item batchItem) error {
	e.mu.RLock()
	if e.closed {
		e.mu.RUnlock()
		return fmt.Errorf("embedder is closed")
	}
	batchCh := e.batchCh
	stopCh := e.stopCh
	e.mu.RUnlock()

	if batchCh == nil {
		return e.processBatch([]batchItem{item})
	}

	select {
	case batchCh <- item:
		return nil
	case <-stopCh:
		return fmt.Errorf("embedder is closed")
	}
}

func (e *Embedder) runBatcher() {
	var batch []batchItem

	for {
		select {
		case item := <-e.batchCh:
			if item.ctx != nil {
				if err := item.ctx.Err(); err != nil {
					item.resp <- batchResult{err: err}
					continue
				}
			}
			batch = append(batch, item)
		drain:
			for len(batch) < e.config.BatchSize {
				select {
				case next := <-e.batchCh:
					batch = append(batch, next)
				default:
					break drain
				}
			}
			if len(batch) > 0 {
				_ = e.processBatch(batch)
				batch = nil
			}
		case <-e.stopCh:
			if len(batch) > 0 {
				_ = e.processBatch(batch)
			}
			return
		}
	}
}

func (e *Embedder) processSingle(ctx context.Context, req EmbedRequest) (*EmbedResult, error) {
	response := make(chan batchResult, 1)
	item := batchItem{ctx: ctx, req: req, resp: response}
	if err := e.processBatch([]batchItem{item}); err != nil {
		return nil, err
	}
	result := <-response
	return result.result, result.err
}

func (e *Embedder) processBatch(items []batchItem) error {
	if len(items) == 0 {
		return nil
	}

	e.mu.RLock()
	if e.closed {
		e.mu.RUnlock()
		for _, item := range items {
			item.resp <- batchResult{err: fmt.Errorf("embedder is closed")}
		}
		return fmt.Errorf("embedder is closed")
	}
	musicClient := e.musicClient
	store := e.vectorStore
	e.mu.RUnlock()

	if musicClient == nil {
		err := fmt.Errorf("music embedder is not configured")
		for _, item := range items {
			item.resp <- batchResult{err: err}
		}
		return err
	}
	if store == nil {
		err := fmt.Errorf("vector store is not configured")
		for _, item := range items {
			item.resp <- batchResult{err: err}
		}
		return err
	}

	type workItem struct {
		item      batchItem
		trackName string
		result    *EmbedResult
		err       error
	}
	works := make([]*workItem, 0, len(items))

	for _, item := range items {
		if item.req.FilePath == "" {
			item.resp <- batchResult{err: fmt.Errorf("file path is required")}
			continue
		}
		if err := item.ctx.Err(); err != nil {
			item.resp <- batchResult{err: err}
			continue
		}

		trackName := item.req.TrackName
		if trackName == "" {
			trackName = canonicalName(item.req.Artist, item.req.Title)
		}
		if trackName == "" {
			trackName = item.req.TrackID
		}

		log.Debug(item.ctx, "Embedding audio track",
			"file", item.req.FilePath,
			"name", trackName,
			"trackId", item.req.TrackID,
		)

		works = append(works, &workItem{
			item:      item,
			trackName: trackName,
			result:    &EmbedResult{TrackName: trackName},
		})
	}

	for _, work := range works {
		if err := work.item.ctx.Err(); err != nil {
			work.err = err
			continue
		}

		if e.config.EnableFlamingo {
			embedding, err := musicClient.EmbedAudio(work.item.req.FilePath)
			if err != nil {
				log.Warn(work.item.ctx, "Audio embedding failed (non-fatal)", err)
			} else {
				work.result.FlamingoEmbedding = float32sToFloat64s(embedding)
			}
		}

		if e.config.EnableDescription {
			description, err := musicClient.GenerateDescription(work.item.req.FilePath)
			if err != nil {
				log.Warn(work.item.ctx, "Audio description failed (non-fatal)", err)
			} else {
				work.result.Description = description
			}
		}
	}

	for _, work := range works {
		if work.err != nil {
			continue
		}
		if err := work.item.ctx.Err(); err != nil {
			work.err = err
			continue
		}

		if e.config.EnableLyrics && work.item.req.Lyrics != "" {
			embedding, err := musicClient.EmbedText(work.item.req.Lyrics)
			if err != nil {
				log.Warn(work.item.ctx, "Lyrics embedding failed (non-fatal)", err)
			} else {
				work.result.LyricsEmbedding = float32sToFloat64s(embedding)
			}
		}

		if work.result.Description != "" {
			embedding, err := musicClient.EmbedText(work.result.Description)
			if err != nil {
				log.Warn(work.item.ctx, "Description embedding failed (non-fatal)", err)
			} else {
				work.result.DescriptionEmbedding = float32sToFloat64s(embedding)
			}
		}
	}

	for _, work := range works {
		if work.err == nil {
			e.storeResults(work.item.ctx, store, work.trackName, work.result)
		}
		work.item.resp <- batchResult{result: work.result, err: work.err}
	}

	return nil
}

func (e *Embedder) storeResults(ctx context.Context, store VectorStore, trackName string, result *EmbedResult) {
	if trackName == "" {
		log.Warn(ctx, "Skipping embedding storage: missing track name")
		return
	}

	if len(result.LyricsEmbedding) > 0 {
		e.storeEmbedding(ctx, store, milvus.CollectionLyrics, trackName, result.LyricsEmbedding, "", ModelLyrics)
	}
	if len(result.DescriptionEmbedding) > 0 {
		e.storeEmbedding(ctx, store, milvus.CollectionDescription, trackName, result.DescriptionEmbedding, result.Description, ModelDescription)
	}
	if len(result.FlamingoEmbedding) > 0 {
		e.storeEmbedding(ctx, store, milvus.CollectionFlamingo, trackName, result.FlamingoEmbedding, "", ModelFlamingo)
	}
}

func (e *Embedder) storeEmbedding(ctx context.Context, store VectorStore, collection, name string, embedding []float64, description, modelID string) {
	data := []milvus.EmbeddingData{{
		Name:        name,
		Embedding:   embedding,
		ModelID:     modelID,
		Description: description,
	}}

	if err := store.Upsert(ctx, collection, data); err != nil {
		log.Error(ctx, "Failed to store embeddings", "error", err, "collection", collection, "track", name)
		return
	}
	log.Debug(ctx, "Stored embeddings", "collection", collection, "track", name)
}

// buildPossibleNames generates all possible track names from a status request.
func buildPossibleNames(req StatusRequest) []string {
	names := make([]string, 0, 3+len(req.AlternateNames))

	// Canonical name from artist + title
	if canonical := canonicalName(req.Artist, req.Title); canonical != "" {
		names = append(names, canonical)
	}

	// Track ID as name
	if req.TrackID != "" {
		names = append(names, req.TrackID)
	}

	// Alternate names (e.g., filename)
	names = append(names, req.AlternateNames...)

	return names
}

// canonicalName generates a canonical name from artist and title.
func canonicalName(artist, title string) string {
	if artist == "" && title == "" {
		return ""
	}
	if artist == "" {
		return title
	}
	if title == "" {
		return artist
	}
	return artist + " - " + title
}
