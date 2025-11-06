# Reembed All Feature - Quick Reference

## Summary
The infrastructure for a "reembed all" feature is **fully implemented and ready to use**. This guide provides quick reference points for understanding and extending the system.

## File Locations

### Core Embedding System
| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| Embedding Models | `python_services/embedding_models.py` | 1,484 | MuQ, MERT, Music2Latent implementations |
| Batch Job | `python_services/batch_embedding_job.py` | 352 | Orchestrates batch re-embedding |
| Socket Server | `python_services/python_embed_server.py` | 335 | File upload handler |
| Text Embeddings | `python_services/text_embedding_service.py` | 301 | Text-to-embedding REST API |
| Recommender API | `python_services/recommender_api.py` | 550 | Batch job endpoints |

### Go Integration
| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| Socket Client | `server/nativeapi/sockets.go` | 62 | Unix socket communication |
| HTTP Handlers | `server/nativeapi/recommendations.go` | 1,691 | Batch endpoint handlers |

### UI
| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| Batch Panel | `ui/src/settings/BatchEmbeddingPanel.jsx` | 437 | React component for batch control |

## Three Embedding Models

### 1. MuQ-MuLan (Default)
- **Dimension**: 1,536 → 4,608 (enriched)
- **Chunking**: 120s window, 15s hop
- **Text Support**: Yes ✓
- **Speed**: Fast
- **Use**: Default choice, balanced
- **Collection**: `embedding`

### 2. MERT-v1-330M (High Detail)
- **Dimension**: 25,600 → 76,800 (enriched)
- **Chunking**: 30s window, 5s hop
- **Text Support**: No (placeholder)
- **Speed**: Slowest
- **Use**: Detailed feature extraction
- **Collection**: `mert_embedding`

### 3. Music2Latent (Compact)
- **Dimension**: 192 → 576 (enriched)
- **Chunking**: No chunking
- **Text Support**: No (placeholder)
- **Speed**: Fastest
- **Use**: Compact representation
- **Collection**: `latent_embedding`

## GPU Memory Management

### Automatic Unload
- **Timeout**: 360 seconds (6 minutes) default
- **Check Interval**: 5 seconds
- **Mechanism**: Background daemon thread
- **Thread Safe**: Uses `Lock()` for synchronization

### Key Methods
```python
# Load model (lazy)
model = embedding_model.ensure_model_loaded()

# Use model (context manager)
with embedding_model.model_session() as model:
    outputs = model(wavs=input_tensor)

# Manual unload
embedding_model.unload_model()

# Clear GPU cache
embedding_model._empty_cuda_cache()
```

## API Endpoints

### Batch Job Control
```bash
# Start batch job
POST /batch/start
{
  "models": ["muq", "mert", "latent"],
  "clearExisting": true
}

# Check progress (poll every 1000ms)
GET /batch/progress

# Cancel running job
POST /batch/cancel
```

### Progress Response
```json
{
  "total_tracks": 5000,
  "processed_tracks": 1250,
  "failed_tracks": 2,
  "current_track": "Artist - Song Title",
  "status": "running",
  "progress_percent": 25.0,
  "estimated_completion": 1731000000.0
}
```

### Status Values
- `running` - Job in progress
- `completed` - Success
- `completed_with_errors` - Partial success
- `failed` - Critical error
- `cancelled` - User cancelled

## Batch Job Workflow

```
Admin UI → POST /batch/start
           ↓
Go Server → Recommender API
           ↓
BatchEmbeddingJob.run() [background thread]
           ↓
For each track:
  - Load model (lazy)
  - Embed with all models
  - Insert to Milvus
  - Update progress every 10 tracks
           ↓
UI polls GET /batch/progress every 1s
           ↓
Job Complete → status = "completed"
```

## Key Classes & Methods

### BaseEmbeddingModel
```python
class BaseEmbeddingModel(ABC):
    def ensure_model_loaded() -> torch.nn.Module
    def model_session() -> contextmanager
    def unload_model() -> None
    def _auto_unload_loop() -> None  # Background thread
    def _empty_cuda_cache() -> None
```

### BatchEmbeddingJob
```python
class BatchEmbeddingJob:
    def run(models_to_use, clear_existing) -> Dict
    def cancel() -> None
    def get_progress() -> BatchJobProgress
    def _process_track(track, models) -> None
    def clear_embeddings(models_to_use) -> None
    def get_all_tracks() -> List[Dict]
```

### BatchJobProgress
```python
@dataclass
class BatchJobProgress:
    total_tracks: int
    processed_tracks: int
    failed_tracks: int
    current_track: Optional[str]
    status: str  # running, completed, etc.
    started_at: float
    estimated_completion: Optional[float]
```

## Performance Characteristics

### Batch Processing
- **ETA Calculation**: Based on rate of first 10 tracks
- **Checkpoint Interval**: Every 100 tracks
- **Failed Track Handling**: Logged but doesn't stop job
- **Memory Optimization**: Model auto-unload after 6 minutes idle

### Processing Times (Approximate)
- **MuQ**: 1-2s per track
- **MERT**: 5-10s per track
- **Music2Latent**: 1-3s per track
- **Total for 5000 tracks**: 8-17 hours (all three models)

## Enrichment Function

All models apply enrichment before storage:
```python
def enrich_embedding(embedding: torch.Tensor) -> torch.Tensor:
    """[D, T] → [3*D] normalized vector"""
    # Returns 3 statistics per dimension:
    # 1. mean (average over time)
    # 2. robust_sigma (IQR-based spread)
    # 3. dmean (rate of change)
    # All L2-normalized
```

## Milvus Configuration

### Collections
| Name | Dimension | Index Type | Metric |
|------|-----------|-----------|--------|
| `embedding` | 1,536 | HNSW | COSINE |
| `mert_embedding` | 76,800 | HNSW | COSINE |
| `latent_embedding` | 576 | HNSW | COSINE |

### Index Parameters
```python
# HNSW (Hierarchical Navigable Small World)
M: 50                # Number of bi-directional links
efConstruction: 250  # Search width during build
```

## Environment Variables

```bash
# Python Services
NAVIDROME_RECOMMENDER_PORT=9002
TEXT_EMBEDDING_PORT=9003
NAVIDROME_MILVUS_URI=http://localhost:19530

# Batch Job
NAVIDROME_DB_PATH=navidrome.db
NAVIDROME_MUSIC_ROOT=/music

# Logging
LOG_DIR=./logs
PID_DIR=./.pids
```

## Common Issues & Solutions

### Models Won't Unload
- Check: Is timeout_seconds too high?
- Solution: Reduce timeout or manually call `unload_model()`

### GPU Out of Memory
- Check: Are multiple models loaded simultaneously?
- Solution: Models auto-unload, but can manually unload between operations

### Batch Job Slow
- Check: Are all three models selected? (MERT is slowest)
- Solution: Select only needed models for faster completion

### Progress Stuck
- Check: Is the background thread running?
- Solution: Restart Python services

## Testing the Feature

```bash
# 1. Check service health
curl http://localhost:9002/healthz

# 2. Start batch job
curl -X POST http://localhost:9002/batch/start \
  -H "Content-Type: application/json" \
  -d '{
    "models": ["muq"],
    "clearExisting": true
  }'

# 3. Poll progress
curl http://localhost:9002/batch/progress

# 4. Cancel if needed
curl -X POST http://localhost:9002/batch/cancel
```

## Data Flow Diagram

```
Navidrome DB
(SQLite)
    ↓
BatchEmbeddingJob.get_all_tracks()
    ↓
For each track:
  Model 1 (MuQ)     → Milvus (embedding)
  Model 2 (MERT)    → Milvus (mert_embedding)
  Model 3 (Latent)  → Milvus (latent_embedding)
    ↓
UI polls progress every 1s
    ↓
UI displays: bar, ETA, current track, failures
    ↓
Job complete → User gets summary
```

## Extension Points

### Add New Model
1. Create `NewEmbeddingModel(BaseEmbeddingModel)`
2. Implement: `_load_model()`, `embed_music()`, `embed_string()`, Milvus schemas
3. Register in `batch_embedding_job.py`: `_initialize_models()`
4. Add UI toggle in `BatchEmbeddingPanel.jsx`

### Customize Progress Polling
- Modify `BatchEmbeddingPanel.jsx` line 151 (current: 1000ms)
- Change `estimated_completion` format in `formatETA()` (line 227)

### Add Cancellation Save State
- Currently just stops processing
- Could save progress to DB and resume later
- Would require: Database table + recovery logic

## References

- Embedding Models: `/python_services/embedding_models.py`
- Batch Job: `/python_services/batch_embedding_job.py`
- API: `/python_services/recommender_api.py`
- Full Documentation: `/EMBEDDING_ARCHITECTURE.md`
