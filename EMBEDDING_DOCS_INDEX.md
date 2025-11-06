# Navidrome Embedding System - Documentation Index

This directory contains comprehensive documentation about the Navidrome embedding system and its "reembed all" infrastructure.

## Quick Start

If you're new to this system, start here:
1. **[EXPLORATION_SUMMARY.md](./EXPLORATION_SUMMARY.md)** - Executive summary and answers to key questions (7 min read)
2. **[REEMBED_QUICK_REFERENCE.md](./REEMBED_QUICK_REFERENCE.md)** - Quick lookup tables and API reference (5 min read)

## Comprehensive Documentation

- **[EMBEDDING_ARCHITECTURE.md](./EMBEDDING_ARCHITECTURE.md)** - Complete system architecture (30 min read)
  - Detailed implementation locations
  - Three embedding models specifications
  - GPU memory management implementation
  - Complete API endpoint documentation
  - Full workflow diagrams
  - Progress tracking details

## Documentation Map

```
EMBEDDING_DOCS_INDEX.md (this file)
├── Quick Start
│   ├── EXPLORATION_SUMMARY.md (overview + executive summary)
│   └── REEMBED_QUICK_REFERENCE.md (quick lookups)
│
└── Comprehensive
    └── EMBEDDING_ARCHITECTURE.md (full details)
```

## Key Files in Codebase

### Python Services (Core Implementation)
- `python_services/embedding_models.py` (1,484 lines)
- `python_services/batch_embedding_job.py` (352 lines)
- `python_services/python_embed_server.py` (335 lines)
- `python_services/text_embedding_service.py` (301 lines)
- `python_services/recommender_api.py` (550 lines)

### Go Server (Integration)
- `server/nativeapi/sockets.go` (62 lines)
- `server/nativeapi/recommendations.go` (1,691 lines)

### React UI (Frontend)
- `ui/src/settings/BatchEmbeddingPanel.jsx` (437 lines)

## Five Key Questions Answered

### 1. Where is embedding functionality implemented?
See: **EXPLORATION_SUMMARY.md** > "Question 1" or **EMBEDDING_ARCHITECTURE.md** > "Question 1"

### 2. What are the three models?
See: **REEMBED_QUICK_REFERENCE.md** > "Three Embedding Models" or **EMBEDDING_ARCHITECTURE.md** > "Question 2"

Summary:
- **MuQ-MuLan**: 1,536-dim, fast, text-enabled
- **MERT-v1-330M**: 76,800-dim, detailed, slowest
- **Music2Latent**: 576-dim, compact, fastest

### 3. How is GPU memory managed?
See: **EMBEDDING_ARCHITECTURE.md** > "Question 3" for full implementation details

Summary:
- Lazy loading (load on first use)
- Auto-unload (360s timeout, 5s check interval)
- Thread-safe with context managers
- `torch.cuda.empty_cache()` after unload

### 4. Are there API endpoints?
See: **REEMBED_QUICK_REFERENCE.md** > "API Endpoints" or **EMBEDDING_ARCHITECTURE.md** > "Question 4"

Key endpoints:
- `POST /batch/start` - Start batch job
- `GET /batch/progress` - Get progress
- `POST /batch/cancel` - Cancel job

### 5. Is there progress tracking?
See: **EMBEDDING_ARCHITECTURE.md** > "Question 5"

Tracking includes:
- Total/processed/failed counts
- Current track
- Status (running, completed, etc.)
- Estimated completion time
- Real-time UI polling

## Architecture Overview

```
User ↓
  Admin UI (React)
    BatchEmbeddingPanel.jsx ↓
  HTTP (Admin Auth)
    Go Server
      recommendations.go ↓
  Python Recommender API
    recommender_api.py
    BatchEmbeddingJob ↓
  Background Thread
    For Each Track:
      Model 1 (MuQ) → Milvus
      Model 2 (MERT) → Milvus
      Model 3 (Latent) → Milvus
      Update Progress ↓
  Progress Polling (1000ms)
    UI Updates:
      - Progress Bar
      - Current Track
      - ETA
      - Failures
```

## Three Embedding Models Summary

| Model | Dimension | Speed | Text | Collection |
|-------|-----------|-------|------|------------|
| MuQ | 1,536 (4,608*) | Fast | Yes | embedding |
| MERT | 25,600 (76,800*) | Slowest | No | mert_embedding |
| Latent | 192 (576*) | Fastest | No | latent_embedding |

*After enrichment (mean, IQR, delta-mean)

## API Quick Reference

### Batch Control
```bash
POST /batch/start
{
  "models": ["muq", "mert", "latent"],
  "clearExisting": true
}

GET /batch/progress

POST /batch/cancel
```

### Response Format
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

## Configuration

Environment variables (from `python_services/SERVICES_README.md`):
```
NAVIDROME_RECOMMENDER_PORT=9002
TEXT_EMBEDDING_PORT=9003
NAVIDROME_MILVUS_URI=http://localhost:19530
NAVIDROME_DB_PATH=navidrome.db
NAVIDROME_MUSIC_ROOT=/music
LOG_DIR=./logs
PID_DIR=./.pids
```

## Implementation Status

### Fully Implemented
- [x] Three embedding models
- [x] Lazy loading + auto-unload
- [x] GPU memory management
- [x] Batch re-embedding
- [x] Progress tracking with ETA
- [x] API endpoints
- [x] React UI component
- [x] Milvus integration
- [x] Text embedding service
- [x] Cancellation support

### Status: Production-Ready
The "reembed all" feature is fully functional and integrated across all layers of the application.

## Performance

For a 5,000 track library:
- **MuQ only**: 2-3 hours
- **MERT only**: 7-14 hours
- **Latent only**: 1.5-4 hours
- **All three**: 8-17 hours

## GPU Memory Management

### Auto-Unload Behavior
- Timeout: 360 seconds (6 minutes)
- Check interval: 5 seconds
- Thread-safe: Uses `threading.Lock()`
- Manual control: `unload_model()`, `shutdown()`

### Key Methods
```python
ensure_model_loaded()      # Lazy load
model_session()            # Context manager
unload_model()             # Explicit unload
_empty_cuda_cache()        # Clear GPU memory
```

## Use Cases

### Single File Upload
- User uploads file → Socket server → Embedding → Milvus
- Includes duplicate detection
- Optional: CUE file splitting
- Optional: LLM-based renaming

### Batch Re-embedding
- Admin initiates → Background job
- Processes all tracks with selected models
- Real-time progress tracking
- Graceful cancellation
- Comprehensive error logging

### Text Queries
- Text input → Text embedding service → Audio space
- Similarity search in Milvus
- Multi-model support
- Used for recommendations

## Extension Points

### Add New Embedding Model
1. Extend `BaseEmbeddingModel`
2. Implement: `_load_model()`, `embed_music()`, `embed_string()`, Milvus schemas
3. Register in `BatchEmbeddingJob._initialize_models()`
4. Add UI checkbox in `BatchEmbeddingPanel.jsx`

### Customize Progress Polling
- Change interval: `BatchEmbeddingPanel.jsx` line 151 (default: 1000ms)
- Format ETA: `formatETA()` function (line 227)

### Add Checkpoint/Resume
- Save progress to database after N tracks
- Load and resume on restart
- Requires: Database table + recovery logic

## Testing

All endpoints can be tested with curl:
```bash
# Health check
curl http://localhost:9002/healthz

# Start batch job
curl -X POST http://localhost:9002/batch/start \
  -H "Content-Type: application/json" \
  -d '{"models": ["muq"], "clearExisting": true}'

# Check progress
curl http://localhost:9002/batch/progress

# Cancel job
curl -X POST http://localhost:9002/batch/cancel
```

## Troubleshooting

See **REEMBED_QUICK_REFERENCE.md** > "Common Issues & Solutions"

Common issues:
- Models won't unload
- GPU out of memory
- Batch job slow
- Progress stuck

## Related Documentation

- `python_services/SERVICES_README.md` - Service startup and configuration
- `python_services/pytest.ini` - Test configuration
- Go server documentation in main Navidrome repo

## Contact & Support

For questions about this documentation:
- Refer to the architecture document for deep dives
- Check quick reference for common questions
- Review actual source code for implementation details

---

**Documentation Version**: 1.0
**Last Updated**: November 6, 2025
**Coverage**: Complete system architecture and implementation
**Files Examined**: 9 major files, ~6,500 lines of code
