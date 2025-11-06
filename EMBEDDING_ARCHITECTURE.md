# Navidrome Embedding System Architecture

## Overview
The Navidrome embedding system is a multi-component architecture for generating, managing, and searching music embeddings across three different embedding models. It includes batch re-embedding capabilities with progress tracking, GPU memory management, and a REST API for recommendations.

---

## Question 1: Where is the embedding functionality implemented?

### Key Files:
1. **Python Services** (Primary Embedding Logic):
   - `/home/henry/projects/navidrome/python_services/embedding_models.py` (1,484 lines)
     - Abstract base class: `BaseEmbeddingModel`
     - Three concrete implementations: `MuQEmbeddingModel`, `MertModel`, `MusicLatentSpaceModel`
     - Audio loading, segmentation, and enrichment logic

   - `/home/henry/projects/navidrome/python_services/batch_embedding_job.py` (352 lines)
     - `BatchEmbeddingJob` class for orchestrating large-scale re-embedding
     - Progress tracking with ETA calculation
     - Graceful cancellation support
     - Checkpoint intervals for safety

   - `/home/henry/projects/navidrome/python_services/python_embed_server.py` (335 lines)
     - Unix socket server for embedding requests
     - Integrates with Milvus for persistence
     - Handles CUE file splitting for multi-track songs
     - Feature pipeline for similarity search and deduplication

   - `/home/henry/projects/navidrome/python_services/text_embedding_service.py` (301 lines)
     - REST API (FastAPI) for text-to-embedding projection
     - Supports both trained models and stub embedders
     - Provides `/embed_text`, `/models`, and `/health` endpoints

   - `/home/henry/projects/navidrome/python_services/recommender_api.py` (550 lines)
     - FastAPI service for playlist recommendations
     - Batch job endpoints: `/batch/start`, `/batch/progress`, `/batch/cancel`
     - Multi-model similarity search
     - Negative prompt penalties

2. **Go Server Integration**:
   - `/home/henry/projects/navidrome/server/nativeapi/sockets.go` (62 lines)
     - Unix socket client for embedding requests
     - JSON-based request/response protocol
   
   - `/home/henry/projects/navidrome/server/nativeapi/recommendations.go` (1,691 lines)
     - HTTP handlers for batch endpoints
     - Proxies batch requests to Python recommender API
     - Admin-only authorization checks

3. **UI Components**:
   - `/home/henry/projects/navidrome/ui/src/settings/BatchEmbeddingPanel.jsx` (437 lines)
     - React component for batch re-embedding UI
     - Real-time progress polling
     - Model selection and clearExisting toggle
     - ETA calculation and status display

---

## Question 2: What are the three models being used for embedding?

### Model Specifications:

#### 1. **MuQ (MuQ-MuLan)** - Default Model
- **Class**: `MuQEmbeddingModel`
- **Output Dimension**: 1536 dimensions (1536 × 3 = **4,608** after enrichment)
- **Model ID**: `OpenMuQ/MuQ-MuLan-large`
- **Processing**:
  - Audio chunking: 120-second windows with 15-second hop
  - Sample rate: 24,000 Hz
  - Data type: float32
- **Milvus Collection**: `embedding`
- **Index**: COSINE similarity with HNSW (Hierarchical Navigable Small World)
- **Text Support**: Yes - native text embedding
- **Use Case**: Default choice - balanced performance and dimensionality

#### 2. **MERT (MERT-v1-330M)** - High Dimensionality
- **Class**: `MertModel`
- **Output Dimension**: 25,600 base (25 layers × 1024 dims)
- **Enriched Dimension**: 76,800 dimensions (25,600 × 3 after enrichment)
- **Model ID**: `m-a-p/MERT-v1-330M`
- **Processing**:
  - Audio chunking: 30-second windows with 5-second hop
  - Sample rate: 24,000 Hz
  - Data type: float32
  - Uses Wav2Vec2FeatureExtractor preprocessor
- **Milvus Collection**: `mert_embedding`
- **Index**: COSINE similarity with HNSW
- **Text Support**: No - returns zero vector (placeholder)
- **Use Case**: Detailed feature extraction - highest dimensionality, most descriptive

#### 3. **Music2Latent (EncoderDecoder)** - Compact Representation
- **Class**: `MusicLatentSpaceModel`
- **Output Dimension**: 192 base (2 complex channels × 64 + magnitude)
- **Enriched Dimension**: 576 dimensions (192 × 3 after enrichment)
- **Model**: `EncoderDecoder` from music2latent package
- **Processing**:
  - Full file encoding (no chunking in original implementation)
  - Sample rate: 44,100 Hz (customizable)
  - Creates latent space representation with real/imaginary/magnitude components
- **Milvus Collection**: `latent_embedding`
- **Index**: COSINE similarity with HNSW
- **Text Support**: No - returns zero vector (placeholder)
- **Use Case**: Compact representation - smallest footprint, fast computation

### Embedding Enrichment
All models apply the `enrich_embedding()` function:
```python
def enrich_embedding(embedding: torch.Tensor) -> torch.Tensor:
    """
    embedding: [D, T] float tensor (D dimensions, T time steps)
    returns: [3*D] normalized vector
      [mean, robust_sigma_iqr, dmean]
    """
```

This creates three features per dimension:
1. **Mean**: Average over time dimension
2. **Robust Spread (IQR)**: Interquartile range → sigma conversion
3. **Delta Mean**: First differences (rate of change)

---

## Question 3: How is model loading/GPU memory management currently handled?

### Architecture: Lazy Loading with Auto-Unload

#### 1. **BaseEmbeddingModel** (Abstract Base Class)
Location: `/home/henry/projects/navidrome/python_services/embedding_models.py` lines 44-188

**Lazy Loading Pattern**:
```python
class BaseEmbeddingModel(ABC):
    def __init__(self, timeout_seconds: int = 360, ...):
        self._model: Optional[torch.nn.Module] = None
        self._last_used = datetime.now(timezone.utc)
        self._lock = Lock()  # Thread-safe access
        self._stop_event = Event()
        self._unloader = Thread(
            target=self._auto_unload_loop,
            name=f"{self.__class__.__name__}UnloadThread",
            daemon=True,
        )
        self._unloader.start()  # Background unload thread
```

**Timeout-Based Auto-Unload**:
- Default timeout: 360 seconds (6 minutes)
- Minimum timeout: 30 seconds
- Background thread checks every 5 seconds
- Unloads model if idle for timeout duration

```python
def _auto_unload_loop(self) -> None:
    """Background loop that releases model after period of inactivity"""
    check_interval = 5
    while not self._stop_event.wait(check_interval):
        with self._lock:
            if self._model is None:
                continue
            idle = datetime.now(timezone.utc) - self._last_used
            if idle >= timedelta(seconds=self._timeout):
                self.logger.info(
                    "Model idle for %s seconds, unloading %s",
                    int(idle.total_seconds()),
                    self.__class__.__name__,
                )
                try:
                    self._release_model(self._model)
                finally:
                    self._model = None
                self._empty_cuda_cache()
```

#### 2. **Model Loading**
```python
def ensure_model_loaded(self) -> Any:
    """Load the model into memory if not already available"""
    with self._lock:
        if self._model is None:
            self.logger.info("Loading embedding model %s", self.__class__.__name__)
            self._model = self._load_model()
        self._last_used = datetime.now(timezone.utc)
        return self._model
```

**Context Manager Pattern**:
```python
@contextmanager
def model_session(self) -> Generator[torch.nn.Module, Any, Any]:
    """Context manager that ensures model is loaded and updates usage tracking"""
    model = self.ensure_model_loaded()
    try:
        yield model
    finally:
        with self._lock:
            self._last_used = datetime.now(timezone.utc)
```

Usage in embedding:
```python
def embed_music(self, music_file: str, music_name: str) -> dict:
    with self.model_session() as model:
        # Use model - automatically tracked as "last used"
        outputs = model(wavs=chunk_tensor)
```

#### 3. **GPU Memory Management**
```python
def _empty_cuda_cache(self) -> None:
    """Clear GPU cache after model unload"""
    try:
        torch.cuda.empty_cache()
    except Exception:  # pragma: no cover
        pass

def unload_model(self) -> None:
    """Manually unload model and release GPU/CPU resources"""
    with self._lock:
        if self._model is None:
            return
        self.logger.info("Manually unloading model %s", self.__class__.__name__)
        try:
            self._release_model(self._model)
        finally:
            self._model = None
    self._empty_cuda_cache()

def shutdown(self) -> None:
    """Stop background thread and release loaded model"""
    self._stop_event.set()
    self._unloader.join(timeout=1)
    with self._lock:
        if self._model is not None:
            self._release_model(self._model)
            self._model = None
    self._empty_cuda_cache()
```

#### 4. **Batch Processing Memory Optimization**

**Single-Track Audio Embedding**:
```python
def embed_audio_tensor(self, waveform, sample_rate, apply_enrichment=True):
    # Audio chunking strategy
    chunk_size = int(self.window_seconds * self.sample_rate)  # e.g., 2,880,000 samples
    hop_size = int(self.hop_seconds * self.sample_rate)       # e.g., 360,000 samples
    
    # All chunks processed in single batch
    chunk_matrix = np.stack(all_chunks, axis=0)
    chunk_tensor = torch.from_numpy(chunk_matrix).to(self.device).to(self.storage_dtype)
    
    with self.model_session() as model:
        with torch.inference_mode():  # Disable gradients
            outputs = model(wavs=chunk_tensor)  # [num_chunks, D]
```

**Batch Audio Embedding** (Multiple Tracks):
```python
def embed_audio_tensor_batch(self, waveforms, sample_rates, apply_enrichment=True):
    # Group chunks by track
    all_chunks = []
    chunk_to_track = []  # Maps chunk index to track index
    
    for track_idx, (waveform, sample_rate) in enumerate(zip(waveforms, sample_rates)):
        # Chunk each track
        for start_sample in starts:
            all_chunks.append(chunk)
            chunk_to_track.append(track_idx)
    
    # Single batched inference for all tracks
    chunk_matrix = np.stack(all_chunks, axis=0)
    chunk_tensor = torch.from_numpy(chunk_matrix).to(self.device).to(self.storage_dtype)
    
    with self.model_session() as model:
        with torch.inference_mode():
            outputs = model(wavs=chunk_tensor)  # [total_chunks, D]
    
    # Group results back by track
    for chunk_idx, track_idx in enumerate(chunk_to_track):
        track_embeddings[track_idx].append(outputs[chunk_idx])
```

#### 5. **Device Management**
```python
class MuQEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, 
                 device: str = "cuda",          # GPU or CPU
                 storage_dtype: torch.dtype = torch.float32):
        self.device = device
        self.storage_dtype = storage_dtype
    
    def _load_model(self) -> torch.nn.Module:
        model = MuQMuLan.from_pretrained(self.model_id)
        model = model.to(self.device).to(self.storage_dtype).eval()
        return model
```

---

## Question 4: Is there an API endpoint for triggering embedding? How does the embedding process work?

### API Endpoints

#### 1. **Socket Server** (Primary File Upload)
- **Path**: `/tmp/navidrome_embed.sock` (Unix socket)
- **Type**: Streaming JSON protocol
- **Handler**: `EmbedSocketServer.handle_connection()` in `python_embed_server.py`

**Request Format**:
```json
{
  "music_file": "/path/to/audio.flac",
  "name": "Track Name",
  "cue_file": "/path/to/optional.cue",
  "settings": {
    "renameEnabled": false,
    "renamingPrompt": "",
    "openAiEndpoint": "",
    "openAiModel": "",
    "useMetadata": true,
    "similaritySearchEnabled": false,
    "dedupThreshold": 0.85,
    "reasoningLevel": "default"
  }
}
```

**Response Format**:
```json
{
  "status": "ok",
  "duplicates": [...],
  "renamedFile": "new_name.flac",
  "allDuplicates": false,
  "splitFiles": [...]  // if CUE splitting occurred
}
```

**Error Response**:
```json
{
  "status": "error",
  "message": "error description"
}
```

#### 2. **Batch Re-embedding Endpoints** (HTTP)

**Start Batch Job**:
- **Endpoint**: `POST /batch/start`
- **Base URL**: Configured via `NAVIDROME_RECOMMENDER_PORT` (default: 9002)
- **Auth**: Admin-only
- **Request**:
```json
{
  "models": ["muq", "mert", "latent"],
  "clearExisting": true
}
```

**Response**:
```json
{
  "status": "started",
  "job_id": 12345678
}
```

**Get Progress**:
- **Endpoint**: `GET /batch/progress`
- **Auth**: Admin-only
- **Response**:
```json
{
  "total_tracks": 5000,
  "processed_tracks": 1250,
  "failed_tracks": 2,
  "current_track": "Artist - Song Title",
  "status": "running",
  "progress_percent": 25.0,
  "estimated_completion": 1731000000.0  // Unix timestamp
}
```

**Cancel Job**:
- **Endpoint**: `POST /batch/cancel`
- **Auth**: Admin-only
- **Response**:
```json
{
  "status": "cancelling"
}
```

#### 3. **Text Embedding Service** (Recommendations Input)
- **Endpoint**: `POST /embed_text`
- **Base URL**: Configured via `TEXT_EMBEDDING_PORT` (default: 9003)

**Request**:
```json
{
  "text": "upbeat dance music",
  "model": "muq"  // "muq", "mert", or "latent"
}
```

**Response**:
```json
{
  "embedding": [0.001, -0.002, ..., 0.003],
  "model": "muq",
  "dimension": 1536,
  "is_stub": false
}
```

### Complete Embedding Workflow

#### Single File Upload (Socket Server Path):

```
1. User uploads audio file via Navidrome UI/API
   ↓
2. Go server (nativeapi/recommendations.go) connects to socket
   - Creates embedRequest with musicFile, name, cuePath, settings
   ↓
3. EmbedSocketServer.handle_connection() receives request
   ↓
4. _process_embedding_request():
   
   a) If CUE file provided:
      - split_flac_with_cue() creates temporary split files
      - Embed each split track separately
      - Combine results into single payload
      - Delete temporary files
   
   b) If no CUE file:
      - Embed entire file as single segment
   ↓
5. Model-specific embed_music() called:
   
   For each embedding model:
   
   a) prepare_music(music_file, music_name)
      - Creates TrackSegment(index=1, title=name, start=0, end=None)
   
   b) For each segment:
      - _load_audio_segment(music_file, offset, duration)
        - Uses librosa.load() or torchaudio.load()
        - Converts to mono, resamples to model's sample rate
      
      - Chunk audio: [0, chunk_size), [hop_size, chunk_size+hop_size), ...
      
      - with model_session() as model:
          - Ensures model loaded (lazy loading)
          - Sets torch.inference_mode()
          - Runs inference: outputs = model(wavs=chunk_tensor)
          - Updates _last_used timestamp
      
      - enrich_embedding(outputs.T) -> [D] vector
        - Computes: mean, IQR-sigma, delta-mean over time
        - L2 normalizes result
      
      - Returns SegmentEmbedding(index, title, offset_seconds, duration_seconds, embedding)
   ↓
6. add_embedding_to_db() called:
   
   a) load_from_json() converts segment embeddings to SongEmbedding objects
   
   b) scan_for_dups(songs, settings)
      - Uses MilvusSimilaritySearcher
      - Searches existing embeddings for duplicates
      - Returns duplicate list
   
   c) feature_pipeline.rename() (if enabled)
      - Can rename file based on metadata or LLM
   
   d) Upsert to Milvus:
      - milvus_client.upsert("embedding", songs_payload)
      - milvus_client.flush("embedding")
   ↓
7. Response sent back via socket:
   {
     "status": "ok",
     "duplicates": [...],
     "renamedFile": "new_name.flac",
     "allDuplicates": false
   }
```

#### Batch Re-embedding Workflow:

```
1. Admin initiates batch job via UI (BatchEmbeddingPanel.jsx)
   - Selects models: ["muq", "mert", "latent"]
   - clearExisting: true (optional)
   ↓
2. POST /batch/start to Recommender API
   ↓
3. RecommendationEngine.handleBatchStart() -> proxyToPython()
   - Routes to: {recommender_base_url}/batch/start
   ↓
4. BatchEmbeddingJob.run() executes in background thread:
   
   a) _initialize_models(models_to_use)
      - Creates MuQEmbeddingModel, MertModel, MusicLatentSpaceModel instances
   
   b) get_all_tracks() from Navidrome SQLite database:
      - SELECT id, path, artist, title, album FROM media_file
      - Populates BatchJobProgress.total_tracks
   
   c) If clear_existing=true:
      - clear_embeddings(models_to_use)
        - client.drop_collection("embedding")
        - client.drop_collection("mert_embedding")
        - client.drop_collection("latent_embedding")
      - _recreate_schemas()
        - Creates collections with proper dimensions and indexes
   
   d) For each track (with tqdm progress bar):
      - if _cancelled: break
      - _process_track(track, models):
         - Resolve audio file path
         - Normalize track name (artist - title format)
         
         - For each model in models_to_use:
           - model.embed_music(audio_path, canonical_name)
           - Returns dict with segments
           - For each segment:
             - client.insert(collection_name, data)
               - name: segment title
               - embedding: [D] vector
               - offset: offset_seconds
               - model_id: model identifier
      
      - Every 10 tracks:
        - Calculate elapsed time
        - Compute rate = elapsed / tracks_processed
        - remaining = total - processed
        - estimated_completion = now + (rate * remaining)
        - Update BatchJobProgress.estimated_completion
      
      - On error:
        - Append to failed_tracks list
        - Log error and continue
   
   e) Finalize progress:
      - status = "completed" | "completed_with_errors" | "cancelled" | "failed"
      - Log summary statistics
   ↓
5. UI polls GET /batch/progress every 1000ms
   ↓
6. Response includes real-time status:
   {
     "total_tracks": 5000,
     "processed_tracks": 1250,
     "failed_tracks": 2,
     "current_track": "Artist - Song",
     "status": "running",
     "progress_percent": 25.0,
     "estimated_completion": 1731000000.0
   }
   ↓
7. UI updates with progress bar, current track, ETA
   - Shows icons and status messages
   - Allows cancellation via POST /batch/cancel
```

---

## Question 5: Is there existing progress tracking or status reporting?

### Progress Tracking Infrastructure

#### 1. **BatchJobProgress** Dataclass
Location: `/home/henry/projects/navidrome/python_services/batch_embedding_job.py` lines 22-33

```python
@dataclass
class BatchJobProgress:
    """Tracks progress of batch embedding job"""
    
    total_tracks: int                      # Total tracks to process
    processed_tracks: int                  # Successfully processed
    failed_tracks: int                     # Failed to process
    current_track: Optional[str]           # e.g., "Artist - Song Title"
    status: str                            # "running", "completed", "failed", "cancelled"
    started_at: float                      # Unix timestamp
    estimated_completion: Optional[float]  # Unix timestamp (ETA)
```

#### 2. **Batch Job Status States**
```
initialized -> running -> {completed | completed_with_errors | failed | cancelled}
```

**Status Transitions**:
- `initialized`: Job object created but not started
- `running`: Processing tracks (set when run() begins)
- `completed`: All tracks processed successfully
- `completed_with_errors`: Some tracks failed, but job finished
- `failed`: Critical error during processing
- `cancelled`: User-initiated cancellation

#### 3. **Real-Time Progress Updates**

**During Processing** (`batch_embedding_job.py` lines 206-227):
```python
for idx, track in enumerate(tqdm(tracks, desc="Embedding tracks")):
    if self._cancelled:
        self.progress.status = "cancelled"
        break
    
    self.progress.current_track = f"{track['artist']} - {track['title']}"
    
    try:
        self._process_track(track, models_to_use)
        self.progress.processed_tracks += 1
    except Exception as e:
        failed_tracks.append((track["id"], str(e)))
        self.progress.failed_tracks += 1
    
    # Update estimated completion every 10 tracks
    if idx > 0 and idx % 10 == 0:
        elapsed = time.time() - self.progress.started_at
        rate = elapsed / (idx + 1)  # seconds per track
        remaining = self.progress.total_tracks - (idx + 1)
        self.progress.estimated_completion = time.time() + (rate * remaining)
```

**ETA Calculation**:
```python
# Example: 5000 tracks, 250 processed, 100 seconds elapsed
elapsed = 100 seconds
rate = 100 / 250 = 0.4 seconds/track
remaining = 5000 - 250 = 4750 tracks
eta_seconds = 4750 * 0.4 = 1900 seconds = 31.67 minutes
estimated_completion = now + 1900
```

#### 4. **API Progress Endpoint**

**Handler** (`recommender_api.py` lines 502-522):
```python
@app.get("/batch/progress")
def get_batch_progress() -> Dict:
    """Get current batch job progress"""
    job = get_current_job()
    if not job:
        return {"status": "no_job"}
    
    progress = job.get_progress()  # Returns copy via dataclasses.replace()
    return {
        "total_tracks": progress.total_tracks,
        "processed_tracks": progress.processed_tracks,
        "failed_tracks": progress.failed_tracks,
        "current_track": progress.current_track,
        "status": progress.status,
        "progress_percent": (
            progress.processed_tracks / progress.total_tracks * 100
            if progress.total_tracks > 0
            else 0
        ),
        "estimated_completion": progress.estimated_completion,
    }
```

#### 5. **UI Progress Display**

**BatchEmbeddingPanel.jsx** (lines 95-154):
```jsx
// Poll for progress when job is running
useEffect(() => {
  if (!isRunning) return
  
  const pollInterval = setInterval(async () => {
    const { data } = await dataProvider.getBatchEmbeddingProgress()
    setProgress(data)
    
    // Check if job completed
    if (data.status === 'completed' ||
        data.status === 'cancelled' ||
        data.status === 'completed_with_errors' ||
        data.status === 'failed') {
      setIsRunning(false)
      clearInterval(pollInterval)
      // Show success/error notification
    }
  }, 1000)  // Poll every second
  
  return () => clearInterval(pollInterval)
}, [isRunning, ...])
```

**UI Components Displayed**:
- Progress bar: `{processed} / {total}` with percentage
- Current track: "Artist - Song Title" being processed
- Failed count: Number of tracks with errors
- Estimated completion: Formatted time (HH:MM:SS)
- Status icon: Check/warning/error circle based on status
- Cancel button: Gracefully stops job

#### 6. **Cancellation Mechanism**

**Cancellation Flag** (`batch_embedding_job.py` lines 310-313):
```python
def cancel(self) -> None:
    """Cancel the running job"""
    self._cancelled = True
    self.logger.info("Job cancellation requested")
```

**Check During Loop** (`batch_embedding_job.py` lines 207-210):
```python
for idx, track in enumerate(tqdm(tracks)):
    if self._cancelled:
        self.progress.status = "cancelled"
        self.logger.info("Job cancelled by user")
        break
    # Continue processing...
```

**API Endpoint** (`recommender_api.py` lines 524-532):
```python
@app.post("/batch/cancel")
def cancel_batch_job() -> Dict:
    """Cancel running batch job"""
    job = get_current_job()
    if not job:
        raise HTTPException(404, "No job running")
    
    job.cancel()
    return {"status": "cancelling"}
```

#### 7. **Logging**

**Module Loggers**:
```python
LOGGER = logging.getLogger("navidrome.embedding_models")    # embedding_models.py
logger = logging.getLogger(__name__)                         # batch_embedding_job.py
LOGGER = logging.getLogger("navidrome.embed_server")        # python_embed_server.py
LOGGER = logging.getLogger("navidrome.recommender")         # recommender_api.py
```

**Log Output Examples**:
```
INFO: Initializing MuQ model...
INFO: Loading embedding model MuQEmbeddingModel
INFO: Initializing MERT model...
INFO: Initializing Latent Space model...
INFO: Found 5000 tracks in database
INFO: Dropping collection: embedding
INFO: Creating schema for muq...
INFO: Embedded Track 1/5000 (20% complete, ETA 02:15:30)
WARNING: Failed to process track 42: Audio file not found
INFO: Job completed in 1234.5s: 4998/5000 tracks
INFO: Failed tracks: 2
```

---

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Navidrome Frontend                        │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              BatchEmbeddingPanel.jsx (React)               │ │
│  │  - Model Selection (MuQ, MERT, Latent Space)             │ │
│  │  - Clear Existing Embeddings Toggle                      │ │
│  │  - Progress Bar with Real-time Updates                   │ │
│  │  - ETA Display                                           │ │
│  │  - Cancel Button                                         │ │
│  └────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────┬──────────────────────────────┘
                                   │
                    HTTP (Admin Auth Required)
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
        ┌───────────▼──────────────┐  ┌──────────▼────────────────┐
        │  Go Server (nativeapi)   │  │  Recommender API (FastAPI)│
        │                          │  │  (Python - Port 9002)     │
        │  sockets.go              │  │                           │
        │  recommendations.go      │  │ • POST /batch/start       │
        │                          │  │ • GET /batch/progress     │
        │ • handleBatchStart()     │  │ • POST /batch/cancel      │
        │ • handleBatchProgress()  │  │ • POST /playlist/{mode}   │
        │ • handleBatchCancel()    │  │                           │
        │                          │  │ RecommendationEngine      │
        └───────────────┬──────────┘  │  • recommend()            │
                        │              │ BatchEmbeddingJob        │
      Unix Socket       │              │  • run()                 │
      /tmp/navi…        │              │  • cancel()              │
      embed.sock        │              │  • get_progress()        │
           ▲            │              └────────────┬─────────────┘
           │            └─────────────────────────┐ │
           │                                       │ │
        ┌──┴─────────────────────────────────────┐│ │
        │  File Upload Request                   ││ │
        └──┬────────────────────────────────────┐││ │
           │                                     │││ │
        ┌──▼──────────────────────────────────┐ │││ │
        │  EmbedSocketServer (Python)        │ │││ │
        │  (Unix Socket Server)              │ │││ │
        │                                     │ │││ │
        │ • handle_connection()              │ │││ │
        │ • _process_embedding_request()     │ │││ │
        │ • add_embedding_to_db()            │ │││ │
        │                                     │ │││ │
        └──┬──────────────────────────────────┘ │││ │
           │                                     │││ │
           ├─────────────────────────────────────┘││ │
           │                                       ││ │
        ┌──▼────────────────────────────────────┐ │ │
        │   Embedding Models (Python)          │ │ │
        │                                       │ │ │
        │ embedding_models.py                  │ │ │
        │                                       │ │ │
        │ BaseEmbeddingModel (Abstract)         │ │ │
        │  • ensure_model_loaded()              │ │ │
        │  • model_session() context manager    │ │ │
        │  • _auto_unload_loop()                │ │ │
        │  • _empty_cuda_cache()                │ │ │
        │                                       │ │ │
        │ Concrete Models:                      │ │ │
        │  MuQEmbeddingModel (1536-dim)         │ │ │
        │   └─ chunk: 120s, hop: 15s            │ │ │
        │  MertModel (76,800-dim)               │ │ │
        │   └─ chunk: 30s, hop: 5s              │ │ │
        │  MusicLatentSpaceModel (576-dim)      │ │ │
        │   └─ full file encoding               │ │ │
        │                                       │ │ │
        │ Text Embedding Service (FastAPI)      │ │ │
        │  • /embed_text endpoint               │ │ │
        │  • Trained models or stubs            │ │ │
        │                                       │ │ │
        └───────────┬─────────────────────────┬─┘ │ │
                    │                         │   │ │
                    │   GPU Memory Mgmt       │   │ │
                    │   (Lazy Load/Unload)    │   │ │
                    │                         │   │ │
        ┌───────────▼────────┬────────────────┘   │ │
        │                    │                     │ │
     ┌──▼────────────────┐ ┌──▼───────────────────┘ │
     │  PyTorch Models   │ │                        │
     │  (GPU/CPU)        │ │   Milvus Database     │
     │                   │ │   (Vector Storage)    │
     │ • MuQ-MuLan      │ │                        │
     │ • MERT-v1-330M   │ │ Collections:          │
     │ • Music2Latent   │ │  • embedding          │
     │                   │ │  • mert_embedding    │
     │ Device: cuda/cpu  │ │  • latent_embedding  │
     │                   │ │                        │
     │ Storage: float32  │ │ Indexes:             │
     │                   │ │  • INVERTED (names)  │
     │ Enrichment:       │ │  • HNSW (vectors)    │
     │  mean            │ │   └─ metric: COSINE  │
     │  IQR             │ │                        │
     │  delta_mean      │ │ Operations:          │
     │  L2 normalized   │ │  • insert            │
     │                   │ │  • upsert            │
     └───────────────────┘ │  • search            │
                           │  • drop_collection  │
                           │                      │
                           └──────────────────────┘
```

---

## Key Integration Points

### 1. **File Upload → Embedding → Milvus**
```
Go Server (file received)
    ↓
Socket Client connects to /tmp/navidrome_embed.sock
    ↓
EmbedSocketServer.handle_connection()
    ↓
For each model in [MuQ, MERT, Latent]:
    - model.embed_music() with lazy loading
    - Returns SegmentEmbedding objects
    ↓
add_embedding_to_db()
    ↓
Milvus upsert/flush
```

### 2. **Batch Re-embedding → Background Job → Progress API**
```
UI Initiates Job (POST /batch/start)
    ↓
Go Server → Recommender API
    ↓
BatchEmbeddingJob.run() in background thread
    ↓
Every 1000ms: UI polls (GET /batch/progress)
    ↓
Response includes: processed_tracks, total_tracks, eta, status
    ↓
Job Complete or User Cancel (POST /batch/cancel)
```

### 3. **Text Query → Embedding → Recommendations**
```
User Text Query ("upbeat dance music")
    ↓
Text Embedding Service (/embed_text)
    ↓
TextToAudioEmbedder.embed_text()
    ↓
Returns [D] vector in audio space
    ↓
Recommender API.search_similar_embeddings()
    ↓
Milvus HNSW search (COSINE)
    ↓
Return ranked tracks
```

---

## Configuration & Environment Variables

```bash
# Python Services
TEXT_EMBEDDING_PORT=9003                    # Text embedding service
NAVIDROME_RECOMMENDER_PORT=9002             # Recommender API
TEXT_EMBEDDING_DEVICE=cuda                  # Device for embedders
TEXT_EMBEDDING_CHECKPOINT_DIR=checkpoints   # Trained models directory

# Batch Job Configuration
NAVIDROME_DB_PATH=navidrome.db              # SQLite database path
NAVIDROME_MUSIC_ROOT=/music                 # Audio files root directory
NAVIDROME_MILVUS_URI=http://localhost:19530 # Milvus server

# Logging
LOG_DIR=./logs                              # Log files directory
PID_DIR=./.pids                             # PID files directory
```

---

## Implementation Status Summary

### Already Implemented
- [x] Three embedding models (MuQ, MERT, Music2Latent)
- [x] Lazy loading with auto-unload
- [x] GPU memory management
- [x] Unix socket server for file uploads
- [x] Batch re-embedding job infrastructure
- [x] Progress tracking with ETA
- [x] API endpoints (/batch/start, /batch/progress, /batch/cancel)
- [x] Cancellation mechanism
- [x] UI component (BatchEmbeddingPanel.jsx)
- [x] Milvus integration with three collections
- [x] Text embedding service
- [x] Recommendation engine

### Ready for "Reembed All" Feature
The infrastructure is fully in place. To implement a "reembed all" feature, you would:

1. **API Level**: Already exists via `/batch/start` endpoint
2. **UI Level**: Implement UI trigger in admin settings (partially done - BatchEmbeddingPanel exists)
3. **Backend**: Use existing BatchEmbeddingJob.run() with all models
4. **Progress**: Poll existing `/batch/progress` endpoint
5. **Database**: Query from Navidrome's SQLite via get_all_tracks()

