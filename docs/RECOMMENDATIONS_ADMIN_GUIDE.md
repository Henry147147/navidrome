# Navidrome Recommender System - Administrator Guide

## Overview

This guide covers setup, configuration, monitoring, and maintenance of Navidrome's recommender system for administrators.

## Architecture

```
┌──────────────┐
│   Frontend   │ React UI
└──────┬───────┘
       │
┌──────▼────────────────────────────────────────┐
│  Go Backend (Navidrome)                       │
│  ┌─────────────────────────────────────────┐  │
│  │ Native API (/api/recommendations/*)     │  │
│  │ - Builds seeds from user activity       │  │
│  │ - Manages playlists & exclusions        │  │
│  │ - Applies dislike penalties             │  │
│  └─────────────┬───────────────────────────┘  │
│                │ HTTP POST                     │
│  ┌─────────────▼───────────────────────────┐  │
│  │ Subsonic API (legacy endpoints)         │  │
│  └─────────────────────────────────────────┘  │
└───────────────────┬───────────────────────────┘
                    │
           ┌────────▼─────────────────┐
           │  RecommendationClient    │
           │  HTTP to :9002           │
           └────────┬─────────────────┘
                    │
┌───────────────────▼────────────────────────────┐
│  Python Services                               │
│  ┌──────────────────────────────────────────┐  │
│  │ Recommender API (:9002)                  │  │
│  │ - MultiModelSimilaritySearcher           │  │
│  │ - Negative prompt penalties              │  │
│  │ - Diversity scoring                      │  │
│  └───┬──────────────────────────────────────┘  │
│      │                                          │
│  ┌───▼──────────────────────────────────────┐  │
│  │ Text Embedding Service (:9003)           │  │
│  │ - Text→Audio projection                  │  │
│  │ - Stub fallback                          │  │
│  └──────────────────────────────────────────┘  │
│                                                 │
│  ┌──────────────────────────────────────────┐  │
│  │ Embedding Socket Server (Unix socket)    │  │
│  │ - Audio file embedding                   │  │
│  │ - CUE sheet support                      │  │
│  └──────────────────────────────────────────┘  │
└────────────┬────────────────────────────────────┘
             │
┌────────────▼────────────┐    ┌────────────────┐
│  Milvus Vector DB       │    │  SQLite        │
│  - 3 collections        │    │  - media_file  │
│  - HNSW indexes         │    │  - user_props  │
└─────────────────────────┘    └────────────────┘
```

## Installation & Setup

### Prerequisites

- **Navidrome**: Latest version with recommender support
- **Python 3.9+**: For ML services
- **Milvus 2.x**: Vector database
- **GPU (optional)**: For faster embedding generation
- **Disk Space**: ~500MB per 1000 tracks for embeddings

### Step 1: Install Python Dependencies

```bash
cd python_services
pip install -r requirements.txt
```

**Required packages:**
- fastapi
- uvicorn
- torch
- transformers
- pymilvus
- numpy
- pydantic

### Step 2: Start Milvus

**Docker (recommended):**
```bash
docker-compose -f docker-compose-milvus.yml up -d
```

**Standalone:**
```bash
milvus-server
```

**Verify Milvus is running:**
```bash
curl http://localhost:19530/healthz
```

### Step 3: Configure Environment Variables

Create `.env` or export environment variables:

```bash
# Milvus Connection
NAVIDROME_MILVUS_URI="http://localhost:19530"

# Navidrome Database
NAVIDROME_DB_PATH="/path/to/navidrome.db"
NAVIDROME_MUSIC_ROOT="/path/to/music"

# Python Services
NAVIDROME_RECOMMENDER_PORT=9002
TEXT_EMBEDDING_PORT=9003
NAVIDROME_RECOMMENDER_DEBUG=false

# Text Embedding (optional)
TEXT_EMBEDDING_CHECKPOINT_DIR="/path/to/checkpoints"
TEXT_EMBEDDING_USE_STUBS=false  # true for development
TEXT_EMBEDDING_DEVICE=cuda  # or cpu

# Batch Processing
BATCH_JOB_WORKERS=4
```

### Step 4: Start Python Services

**Using service scripts (recommended):**
```bash
cd python_services
./start_services.sh
```

**Manual startup:**
```bash
# Terminal 1: Recommender API
python3 recommender_api.py

# Terminal 2: Text Embedding Service
python3 text_embedding_service.py

# Terminal 3: Embedding Socket Server
python3 python_embed_server.py
```

**Verify services:**
```bash
curl http://localhost:9002/healthz
curl http://localhost:9003/health
```

### Step 5: Configure Navidrome

Edit `navidrome.toml`:

```toml
[Recommendations]
BaseURL = "http://127.0.0.1:9002"
Timeout = "5s"
DefaultLimit = 25
Diversity = 0.15
```

### Step 6: Initial Embedding

After starting services, embed your music library:

1. Log in as admin user
2. Navigate to Settings → Recommendations
3. Click "Start Re-embedding"
4. Select models (start with just MuQ for testing)
5. Enable "Clear existing embeddings"
6. Click "Start Job"

**Time estimates:**
- Small library (< 1,000 tracks): 30-60 minutes
- Medium library (1,000-10,000 tracks): 2-8 hours
- Large library (> 10,000 tracks): 8-24 hours

*Times vary based on CPU/GPU, model selection, and file formats.*

## Configuration

### Navidrome Configuration

```toml
[Recommendations]
# Python service URL
BaseURL = "http://127.0.0.1:9002"

# Request timeout
Timeout = "5s"

# Default playlist length
DefaultLimit = 25

# Base diversity (0.0-1.0)
Diversity = 0.15
```

### Python Service Configuration

**recommender_api.py:**
- Port: 9002
- Milvus URI: from environment
- Debug logging: via `NAVIDROME_RECOMMENDER_DEBUG`

**text_embedding_service.py:**
- Port: 9003
- Checkpoint directory: from environment
- Stub mode: via `TEXT_EMBEDDING_USE_STUBS`
- Device: cuda/cpu via `TEXT_EMBEDDING_DEVICE`

### Milvus Configuration

**Collections:**
- `embedding` (muq): 1536 dimensions
- `mert_embedding` (mert): 76,800 dimensions
- `latent_embedding` (latent): 576 dimensions

**Index parameters:**
- Type: HNSW (Hierarchical Navigable Small World)
- M: 50 (connectivity)
- efConstruction: 250 (build quality)

**Adjust for performance:**
```python
# Higher M = better recall, more memory
M = 50  # default: 50, range: 8-64

# Higher efConstruction = better index, slower build
efConstruction = 250  # default: 250, range: 100-500
```

## Monitoring

### Service Health Checks

```bash
# Recommender API
curl http://localhost:9002/healthz

# Text Embedding Service
curl http://localhost:9003/health

# Embedding Socket Server
ls -la /tmp/navidrome_embed.sock
```

### Logs

**View service logs:**
```bash
cd python_services
tail -f logs/recommender_api.log
tail -f logs/text_embedding.log
tail -f logs/embedding_server.log
```

**Log rotation:**
Logs automatically rotate at 10MB with 5 backups.

### Performance Metrics

Monitor these metrics:

1. **Response times:**
   - Text recommendations: < 2s
   - Multi-model search: < 3s
   - Single-model search: < 1s

2. **Batch job throughput:**
   - Target: > 500 tracks/hour
   - Varies by model and hardware

3. **Memory usage:**
   - MuQ model: ~2GB RAM
   - MERT model: ~8GB RAM
   - Latent model: ~1GB RAM

4. **Milvus queries:**
   - Search latency: < 100ms
   - Insert latency: < 10ms

### Milvus Monitoring

```bash
# Check collection stats
curl -X POST "http://localhost:19530/v1/vector/collections/embedding/stats"

# Check index status
curl "http://localhost:19530/v1/vector/collections/embedding/indexes"
```

## Maintenance

### Regular Tasks

#### Weekly:
- Check service logs for errors
- Monitor disk space (Milvus data)
- Review batch job failures (if any)

#### Monthly:
- Backup Milvus data
- Analyze recommendation quality feedback
- Review resource usage trends

#### As Needed:
- Re-embed library after adding many tracks
- Update text embedding models
- Tune hyperparameters based on user feedback

### Batch Re-embedding

**When to re-embed:**
- After major library additions (> 10% new tracks)
- After model updates
- If recommendation quality degrades
- To switch embedding models

**How to trigger:**
1. Admin UI: Settings → Batch Re-embedding
2. API: `POST /api/recommendations/batch/start`
3. Python: Call batch_embedding_job.py directly

**Best practices:**
- Schedule during low-usage periods
- Start with one model, add others later
- Monitor progress and errors
- Keep old embeddings until new ones complete

### Troubleshooting

#### Recommendations not working
**Symptoms:** No results returned, errors in UI

**Diagnosis:**
```bash
# Check services
curl http://localhost:9002/healthz

# Check Milvus
curl http://localhost:19530/healthz

# Check Navidrome logs
tail -f /var/log/navidrome.log
```

**Solutions:**
- Restart Python services
- Restart Milvus
- Check network connectivity
- Verify embeddings exist in Milvus

#### Slow recommendations
**Symptoms:** > 5s response time

**Diagnosis:**
```bash
# Check Milvus query performance
# Look for SLOW_QUERY in logs

# Check CPU/Memory usage
htop

# Check Milvus index status
```

**Solutions:**
- Reduce search top_k
- Optimize Milvus index parameters
- Add more CPU/RAM
- Use GPU for embeddings

#### Text recommendations fail
**Symptoms:** "Text embedding service unavailable"

**Diagnosis:**
```bash
# Check text service
curl http://localhost:9003/health

# Check model files
ls -la checkpoints/
```

**Solutions:**
- Verify text service is running
- Check checkpoint files exist
- Enable stub mode for testing
- Review text_embedding.log

#### Batch job stalls
**Symptoms:** Progress stops, no updates

**Diagnosis:**
```bash
# Check batch progress
curl http://localhost:9002/batch/progress

# Check for errors
tail -f logs/recommender_api.log | grep ERROR
```

**Solutions:**
- Cancel and restart job
- Check disk space
- Verify Milvus is responsive
- Review failed tracks list

### Backup & Recovery

#### Backup Milvus Data

```bash
# Stop Milvus
docker stop milvus-standalone

# Backup data directory
tar -czf milvus-backup-$(date +%Y%m%d).tar.gz /path/to/milvus/data

# Restart Milvus
docker start milvus-standalone
```

#### Restore from Backup

```bash
# Stop Milvus
docker stop milvus-standalone

# Restore data
tar -xzf milvus-backup-YYYYMMDD.tar.gz -C /path/to/milvus/

# Restart Milvus
docker start milvus-standalone

# Verify collections
curl http://localhost:19530/v1/vector/collections
```

#### Backup Navidrome Database

```bash
# SQLite backup (includes user settings)
sqlite3 navidrome.db ".backup navidrome-backup-$(date +%Y%m%d).db"
```

## Advanced Configuration

### GPU Acceleration

**Enable GPU for embeddings:**
```bash
export TEXT_EMBEDDING_DEVICE=cuda
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
```

**Multi-GPU:**
```python
# In embedding_models.py, modify model initialization:
device = f"cuda:{gpu_id}"
```

### Custom Text Embedding Models

**Training your own models:**
See `music-text-embedding-training/` directory for training pipeline.

**Deploying custom models:**
1. Train model and save checkpoint
2. Copy checkpoint to `checkpoints/{model}_best_r1.pt`
3. Restart text embedding service
4. Verify model loads: `curl http://localhost:9003/models`

### Tuning Hyperparameters

**Diversity parameter:**
```python
# In recommender_api.py
DEFAULT_DIVERSITY = 0.15  # Increase for more variety
```

**Similarity threshold:**
```python
# Minimum similarity to consider a match
MIN_SIMILARITY = 0.3  # Increase to be more selective
```

**Top-k multiplier:**
```python
# Fetch N times more candidates for filtering
TOP_K_MULTIPLIER = 3  # Increase for better filtering
```

### Scaling Recommendations

**For large deployments:**

1. **Horizontal Scaling:**
   - Run multiple Python service instances
   - Load balance with nginx/HAProxy
   - Use shared Milvus cluster

2. **Milvus Cluster:**
   - Deploy Milvus in cluster mode
   - Use separate query and index nodes
   - Scale based on load

3. **Caching:**
   - Cache frequent queries
   - Use Redis for session data
   - Cache user settings

4. **Optimization:**
   - Use smaller embedding models
   - Reduce top_k search size
   - Batch similar requests

## Security

### Access Control

- Batch re-embedding: Admin only
- Recommendation generation: Authenticated users
- Settings modification: Per-user

### API Security

- All endpoints behind Navidrome auth
- Rate limiting via Navidrome
- No public endpoints exposed

### Data Privacy

- Embeddings stay on your server
- No external API calls
- User data never leaves your instance

## Performance Tuning

### For Small Libraries (< 1,000 tracks)
```toml
DefaultLimit = 20
Diversity = 0.2
```

Use: Single model (MuQ), CPU sufficient

### For Medium Libraries (1,000-10,000 tracks)
```toml
DefaultLimit = 25
Diversity = 0.15
```

Use: MuQ + MERT, GPU recommended for batch jobs

### For Large Libraries (> 10,000 tracks)
```toml
DefaultLimit = 30
Diversity = 0.10
```

Use: All models, GPU required, Milvus cluster recommended

## Troubleshooting Reference

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| No recommendations | Embeddings not generated | Run batch re-embedding |
| Slow queries | Milvus not indexed | Check index status, rebuild |
| OOM errors | Not enough RAM for model | Use smaller model or add RAM |
| Text mode fails | Service not running | Start text_embedding_service.py |
| Batch job fails | Disk space or Milvus issue | Check disk, verify Milvus up |
| Stale recommendations | Need re-embedding | Re-embed with newer models |

## API Endpoints (for monitoring/automation)

```bash
# Get recommendation settings
GET /api/recommendations/settings

# Update recommendation settings
PUT /api/recommendations/settings

# Generate recommendations
POST /api/recommendations/{mode}
# modes: recent, favorites, all, discovery, custom, text

# Start batch job (admin only)
POST /api/recommendations/batch/start
{
  "models": ["muq", "mert", "latent"],
  "clearExisting": true
}

# Get batch progress (admin only)
GET /api/recommendations/batch/progress

# Cancel batch job (admin only)
POST /api/recommendations/batch/cancel
```

## FAQ for Administrators

**Q: How much storage do embeddings require?**
A: Approximately 500MB per 1,000 tracks (varies by model count).

**Q: Can I run this without GPU?**
A: Yes, but batch embedding will be slower. Inference (recommendations) works fine on CPU.

**Q: How do I update the models?**
A: Replace checkpoint files, restart services, optionally re-embed library.

**Q: What if Milvus goes down?**
A: Recommendations will fail. Users can still play music normally. Restart Milvus to restore.

**Q: Can I migrate Milvus data to another server?**
A: Yes, backup and restore the data directory, update URI in config.

**Q: How do I debug a failing batch job?**
A: Check `logs/recommender_api.log` for errors. Look at `failed_tracks` in progress response.

## Support

For issues:
1. Check logs: `python_services/logs/`
2. Verify service health: `curl http://localhost:9002/healthz`
3. Review GitHub issues: https://github.com/navidrome/navidrome/issues
4. Discord community: https://discord.gg/xh7j7yF

## Conclusion

The Navidrome recommender system requires some setup but provides powerful music discovery features. Follow this guide for smooth deployment and operation. Monitor logs, back up data regularly, and tune settings based on your users' feedback.

---

**Version:** 2.0
**Last Updated:** 2025-11-05
**See Also:** [User Guide](RECOMMENDATIONS_USER_GUIDE.md) | [Architecture Documentation](RECOMMENDATIONS_ARCHITECTURE.md)
