# Navidrome Python Backend (Unified)

This directory now runs **one** FastAPI service that handles text embeddings, playlist recommendations, and audio embedding uploads.

## Endpoints
- `POST /embed_text` – text → embedding (muq or qwen3)
- `GET /models` – available text models
- `GET /health` – unified health report
- `POST /playlist/{mode}` – recommendation engine
- `POST /batch/start|/batch/progress|/batch/cancel` – batch re-embedding controls
- `POST /embed/audio` – audio embedding/upload (replaces the old Unix socket server)

Default base URL: `http://127.0.0.1:9002` (configurable via `NAVIDROME_SERVICE_PORT`, `NAVIDROME_RECOMMENDER_PORT`, or `TEXT_EMBEDDING_PORT`).

## Quick Start

```bash
./start_services.sh
```

Logs: `logs/navidrome_service.log`

Check status:

```bash
./status_services.sh
```

Stop the service:

```bash
./stop_services.sh
```

## Configuration

- `NAVIDROME_SERVICE_PORT` – preferred port (falls back to `NAVIDROME_RECOMMENDER_PORT` or `TEXT_EMBEDDING_PORT`, default 9002)
- `LOG_DIR` / `PID_DIR` – log and PID locations (default `./logs` and `./.pids`)
- Text embedding options: see `text_embedding_service.py`
- Recommendation options: see `recommender_api.py`
- Audio embedding options: see `python_embed_server.py`

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Troubleshooting

- Check the log file: `tail -f logs/navidrome_service.log`
- Ensure the port is free: `netstat -ln | grep 9002`
- Confirm Python deps: `pip install -r requirements.txt`

