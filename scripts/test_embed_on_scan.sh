#!/bin/bash
set -e

MUSIC_FOLDER="${MUSIC_FOLDER:-/mnt/z/music}"
PORT="${PORT:-4500}"
PYTHON_PORT="${PYTHON_PORT:-9002}"
MILVUS_DB="${MILVUS_DB:-./test_milvus.db}"
LOG_DIR="${LOG_DIR:-./test_logs}"
WAIT_TIME="${WAIT_TIME:-90}"

mkdir -p "$LOG_DIR"

echo "========================================"
echo "Embed-on-Scan Integration Test"
echo "========================================"
echo "Music folder: $MUSIC_FOLDER"
echo "Navidrome port: $PORT"
echo "Python service port: $PYTHON_PORT"
echo "Milvus DB: $MILVUS_DB"
echo "Logs: $LOG_DIR"
echo "Wait time: ${WAIT_TIME}s"
echo "========================================"
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "Cleaning up processes..."
    if [ -n "$PYTHON_PID" ]; then
        kill $PYTHON_PID 2>/dev/null || true
    fi
    if [ -n "$NAVIDROME_PID" ]; then
        kill $NAVIDROME_PID 2>/dev/null || true
    fi
    wait 2>/dev/null || true
    echo "Cleanup complete"
}
trap cleanup EXIT INT TERM

# Start Python service
echo "[1/6] Starting Python embedding service..."
cd python_services
python ./navidrome_service.py --milvus-db-path "$MILVUS_DB" -v > "../$LOG_DIR/python.log" 2>&1 &
PYTHON_PID=$!
cd ..
echo "      Python service started (PID: $PYTHON_PID)"

# Wait for Python service to be ready
echo "[2/6] Waiting for Python service to be ready..."
for i in {1..30}; do
    if curl -s "http://localhost:$PYTHON_PORT/health" > /dev/null 2>&1; then
        echo "      Python service ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "      ERROR: Python service failed to start within 30 seconds"
        echo "      Check logs at: $LOG_DIR/python.log"
        tail -20 "$LOG_DIR/python.log"
        exit 1
    fi
    sleep 1
done

# Build Navidrome
echo "[3/6] Building Navidrome..."
make build > "$LOG_DIR/build.log" 2>&1
echo "      Build complete"

# Start Navidrome
echo "[4/6] Starting Navidrome..."
./navidrome --port "$PORT" --musicfolder "$MUSIC_FOLDER" --address "127.0.0.1" > "$LOG_DIR/navidrome.log" 2>&1 &
NAVIDROME_PID=$!
echo "      Navidrome started (PID: $NAVIDROME_PID)"

# Wait for Navidrome to be ready
echo "[5/6] Waiting for Navidrome to be ready..."
for i in {1..30}; do
    if curl -s "http://localhost:$PORT/ping" > /dev/null 2>&1; then
        echo "      Navidrome ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "      ERROR: Navidrome failed to start within 30 seconds"
        echo "      Check logs at: $LOG_DIR/navidrome.log"
        tail -20 "$LOG_DIR/navidrome.log"
        exit 1
    fi
    sleep 1
done

# Trigger scan
echo "[6/6] Triggering full scan..."
SCAN_URL="http://localhost:$PORT/rest/startScan?u=henry&t=74ad489cc64ef7d06b638b23714ce524&s=d94529&f=json&v=1.8.0&c=NavidromeUI&fullScan=true"
SCAN_RESPONSE=$(curl -s "$SCAN_URL" || echo "")
if [ -z "$SCAN_RESPONSE" ]; then
    echo "      WARNING: Scan request may have failed"
else
    echo "      Scan triggered successfully"
fi

echo ""
echo "========================================"
echo "Monitoring embedding progress for ${WAIT_TIME} seconds..."
echo "========================================"
echo ""

# Monitor progress
START_TIME=$(date +%s)
LAST_PROGRESS=""
while true; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))

    if [ $ELAPSED -ge $WAIT_TIME ]; then
        break
    fi

    # Check for latest progress in logs
    PROGRESS=$(grep "Embedding progress" "$LOG_DIR/navidrome.log" 2>/dev/null | tail -n 1 || echo "")
    if [ -n "$PROGRESS" ] && [ "$PROGRESS" != "$LAST_PROGRESS" ]; then
        echo "[${ELAPSED}s] $PROGRESS"
        LAST_PROGRESS="$PROGRESS"
    fi

    # Check for errors
    RECENT_ERROR=$(grep -i "error.*embed" "$LOG_DIR/navidrome.log" 2>/dev/null | tail -n 1 || echo "")
    if [ -n "$RECENT_ERROR" ]; then
        echo "[${ELAPSED}s] ERROR: $RECENT_ERROR"
    fi

    sleep 5
done

echo ""
echo "========================================"
echo "Test Complete - Summary"
echo "========================================"
echo ""

# Summary statistics
echo "Navidrome Statistics:"
echo "---------------------"
SCHEDULED=$(grep -c "Scheduling embeddings" "$LOG_DIR/navidrome.log" 2>/dev/null || echo "0")
echo "  Embedding runs scheduled: $SCHEDULED"

EMBEDDED=$(grep -c "Embedded track in background" "$LOG_DIR/navidrome.log" 2>/dev/null || echo "0")
echo "  Tracks embedded: $EMBEDDED"

SKIPPED=$(grep -c "Embedding already present, skipping" "$LOG_DIR/navidrome.log" 2>/dev/null || echo "0")
echo "  Tracks skipped (already embedded): $SKIPPED"

FAILED=$(grep -c "Embedding failed" "$LOG_DIR/navidrome.log" 2>/dev/null || echo "0")
echo "  Embedding failures: $FAILED"

HEALTH_CHECKS=$(grep -c "health check" "$LOG_DIR/navidrome.log" 2>/dev/null || echo "0")
echo "  Health checks performed: $HEALTH_CHECKS"

echo ""
echo "Latest Progress:"
echo "----------------"
grep "Embedding progress" "$LOG_DIR/navidrome.log" 2>/dev/null | tail -n 1 || echo "  No progress logged"

echo ""
echo "Python Service Statistics:"
echo "-------------------------"
REQUESTS=$(grep -c "Received embed request" "$LOG_DIR/python.log" 2>/dev/null || echo "0")
echo "  Embed requests received: $REQUESTS"

COMPLETED=$(grep -c "Uploading embedding for" "$LOG_DIR/python.log" 2>/dev/null || echo "0")
echo "  Embeddings completed: $COMPLETED"

PY_ERRORS=$(grep -c "Exception\|Error" "$LOG_DIR/python.log" 2>/dev/null || echo "0")
echo "  Errors/Exceptions: $PY_ERRORS"

echo ""
echo "Log Files:"
echo "----------"
echo "  Navidrome: $LOG_DIR/navidrome.log"
echo "  Python:    $LOG_DIR/python.log"
echo "  Build:     $LOG_DIR/build.log"
echo ""

# Check for common issues
if grep -q "connect to socket.*failed" "$LOG_DIR/navidrome.log" 2>/dev/null; then
    echo "WARNING: Socket connection failures detected!"
fi

if grep -q "health check failed" "$LOG_DIR/navidrome.log" 2>/dev/null; then
    echo "WARNING: Health check failures detected!"
fi

if grep -q "CUDA out of memory" "$LOG_DIR/python.log" 2>/dev/null; then
    echo "WARNING: GPU out of memory detected!"
fi

echo "========================================"
