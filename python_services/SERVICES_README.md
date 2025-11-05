# Navidrome Python Backend Services

This directory contains Python backend services for Navidrome's music recommendation and embedding features.

## Services

The backend consists of three main services:

1. **Text Embedding Service** (port 9003)
   - Provides text-to-embedding conversion for music recommendations
   - REST API service built with FastAPI
   - Endpoint: `http://localhost:9003`

2. **Recommender API** (port 9002)
   - Handles music recommendation requests
   - REST API service built with FastAPI
   - Endpoint: `http://localhost:9002`

3. **Embedding Socket Server** (Unix socket)
   - Processes audio embeddings via Unix socket
   - Socket path: `/tmp/navidrome_embed.sock`

## Quick Start

### Starting All Services

```bash
./start_services.sh
```

This will start all three services in the background. Logs will be written to the `logs/` directory.

### Checking Status

```bash
./status_services.sh
```

Shows the current status of all services, including:
- Running status (PID, runtime)
- Port/socket information
- Log file details

### Stopping Services

```bash
./stop_services.sh
```

Gracefully stops all running services.

## Advanced Usage

### Start Individual Services

Start only specific services:

```bash
# Start only text embedding service
./start_services.sh --text-only

# Start only recommender API
./start_services.sh --recommender-only

# Start only socket server
./start_services.sh --socket-only

# Start HTTP services without socket server
./start_services.sh --no-socket
```

### Stop Individual Services

```bash
# Stop only text embedding service
./stop_services.sh --text

# Stop only recommender API
./stop_services.sh --recommender

# Stop only socket server
./stop_services.sh --socket
```

### Custom Ports

Override default ports using environment variables:

```bash
# Use custom ports
TEXT_EMBEDDING_PORT=8003 NAVIDROME_RECOMMENDER_PORT=8002 ./start_services.sh
```

### View Logs

```bash
# View all logs in real-time
tail -f logs/*.log

# View specific service log
tail -f logs/recommender_api.log

# View recent errors
grep -i error logs/*.log
```

## Configuration

### Environment Variables

- `TEXT_EMBEDDING_PORT` - Port for text embedding service (default: 9003)
- `NAVIDROME_RECOMMENDER_PORT` - Port for recommender API (default: 9002)
- `LOG_DIR` - Directory for log files (default: `./logs`)
- `PID_DIR` - Directory for PID files (default: `./.pids`)

### Service-Specific Configuration

Each service can be configured through additional environment variables. Refer to the individual service files for details:

- `text_embedding_service.py` - Text embedding configuration
- `recommender_api.py` - Recommender configuration
- `python_embed_server.py` - Socket server configuration

## Requirements

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Required packages:
- fastapi
- uvicorn
- pymilvus
- torch
- torchaudio
- And others listed in `requirements.txt`

## Troubleshooting

### Service Won't Start

1. Check if the port is already in use:
   ```bash
   netstat -ln | grep 9002
   netstat -ln | grep 9003
   ```

2. Check the log files:
   ```bash
   cat logs/recommender_api.log
   cat logs/text_embedding_service.log
   ```

3. Ensure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

### Service Crashes

1. Check the logs for error messages:
   ```bash
   tail -50 logs/*.log | grep -i error
   ```

2. Ensure sufficient disk space and memory

3. Check for stale PID or socket files:
   ```bash
   rm -rf .pids/*
   rm -f /tmp/navidrome_embed.sock
   ```

### Socket Permission Issues

If you encounter socket permission errors:

```bash
sudo rm -f /tmp/navidrome_embed.sock
./start_services.sh
```

## Development

### Running Services Manually

For development, you can run services directly:

```bash
# Text embedding service
python3 text_embedding_service.py

# Recommender API
python3 recommender_api.py

# Socket server
python3 python_embed_server.py
```

### Running Tests

```bash
# Run all tests
python3 -m pytest

# Run with verbose output
python3 -m pytest -v

# Run specific test file
python3 -m pytest tests/test_recommender_api.py
```

## Integration with Navidrome

These services are designed to work with the main Navidrome Go application. Ensure:

1. Navidrome is configured to use these services
2. The ports/socket paths match Navidrome's configuration
3. Services are running before starting Navidrome

## Support

For issues related to:
- **Service scripts**: Check logs in `./logs/`
- **Navidrome integration**: Refer to main Navidrome documentation
- **Python dependencies**: Check `requirements.txt` and ensure all packages are installed
