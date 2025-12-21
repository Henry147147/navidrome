#!/bin/bash

LOG_FILE="${1:-./test_logs/navidrome.log}"
PYTHON_LOG="${2:-./test_logs/python.log}"

if [ ! -f "$LOG_FILE" ]; then
    echo "Error: Navidrome log file not found: $LOG_FILE"
    exit 1
fi

if [ ! -f "$PYTHON_LOG" ]; then
    echo "Error: Python log file not found: $PYTHON_LOG"
    exit 1
fi

echo "=========================================="
echo "Embedding Log Analysis"
echo "=========================================="
echo "Navidrome log: $LOG_FILE"
echo "Python log: $PYTHON_LOG"
echo ""

# Check for errors
echo "ERRORS:"
echo "-------"
ERROR_COUNT=$(grep -c -i "error.*embed" "$LOG_FILE" || echo "0")
if [ "$ERROR_COUNT" -gt 0 ]; then
    echo "  Found $ERROR_COUNT embedding-related errors"
    echo "  Recent errors:"
    grep -i "error.*embed" "$LOG_FILE" | tail -n 5 | sed 's/^/    /'
else
    echo "  No errors found"
fi

# Check for panics
echo ""
echo "PANICS:"
echo "-------"
PANIC_COUNT=$(grep -c -i "panic" "$LOG_FILE" || echo "0")
if [ "$PANIC_COUNT" -gt 0 ]; then
    echo "  WARNING: Found $PANIC_COUNT panics!"
    grep -i "panic" "$LOG_FILE" | sed 's/^/    /'
else
    echo "  No panics found"
fi

# Check socket connectivity
echo ""
echo "SOCKET ISSUES:"
echo "--------------"
SOCKET_ERRORS=$(grep -c "connect to socket.*failed\|connect to socket.*error" "$LOG_FILE" || echo "0")
if [ "$SOCKET_ERRORS" -gt 0 ]; then
    echo "  WARNING: Found $SOCKET_ERRORS socket connection errors"
    grep "connect to socket" "$LOG_FILE" | grep -i "error\|failed" | tail -n 3 | sed 's/^/    /'
else
    echo "  No socket connection errors"
fi

# Check health checks
echo ""
echo "HEALTH CHECKS:"
echo "--------------"
HEALTH_CHECKS=$(grep -c "health check" "$LOG_FILE" || echo "0")
HEALTH_FAILURES=$(grep -c "health check failed" "$LOG_FILE" || echo "0")
echo "  Health checks performed: $HEALTH_CHECKS"
echo "  Health check failures: $HEALTH_FAILURES"
if [ "$HEALTH_FAILURES" -gt 0 ]; then
    echo "  Recent failures:"
    grep "health check failed" "$LOG_FILE" | tail -n 3 | sed 's/^/    /'
fi

# Check progress
echo ""
echo "PROGRESS LOGS:"
echo "--------------"
PROGRESS_COUNT=$(grep -c "Embedding progress" "$LOG_FILE" || echo "0")
if [ "$PROGRESS_COUNT" -gt 0 ]; then
    echo "  Progress updates: $PROGRESS_COUNT"
    echo "  Latest progress:"
    grep "Embedding progress" "$LOG_FILE" | tail -n 5 | sed 's/^/    /'
else
    echo "  No progress logs found"
fi

# Check completion statistics
echo ""
echo "COMPLETION STATS:"
echo "-----------------"
SCHEDULED=$(grep -c "Scheduling embeddings" "$LOG_FILE" || echo "0")
echo "  Embedding runs scheduled: $SCHEDULED"

EMBEDDED=$(grep -c "Embedded track in background" "$LOG_FILE" || echo "0")
echo "  Tracks successfully embedded: $EMBEDDED"

SKIPPED=$(grep -c "Embedding already present, skipping" "$LOG_FILE" || echo "0")
echo "  Tracks skipped (already embedded): $SKIPPED"

FAILED=$(grep -c "Embedding failed after retries" "$LOG_FILE" || echo "0")
echo "  Tracks failed after retries: $FAILED"

RETRIES=$(grep -c "Retrying embedding" "$LOG_FILE" || echo "0")
echo "  Retry attempts: $RETRIES"

# Check Python service
echo ""
echo "PYTHON SERVICE:"
echo "---------------"
REQUESTS=$(grep -c "Received.*request via socket" "$PYTHON_LOG" || echo "0")
echo "  Total requests received: $REQUESTS"

STATUS_REQUESTS=$(grep -c "Received status request" "$PYTHON_LOG" || echo "0")
echo "  Status check requests: $STATUS_REQUESTS"

EMBED_REQUESTS=$(grep -c "Received embed request" "$PYTHON_LOG" || echo "0")
echo "  Embed requests: $EMBED_REQUESTS"

HEALTH_REQUESTS=$(grep -c "Received health request" "$PYTHON_LOG" || echo "0")
echo "  Health check requests: $HEALTH_REQUESTS"

COMPLETED=$(grep -c "Uploading embedding for\|Upserting embeddings to Milvus" "$PYTHON_LOG" || echo "0")
echo "  Embeddings completed: $COMPLETED"

PY_ERRORS=$(grep -c "ERROR\|Exception" "$PYTHON_LOG" || echo "0")
echo "  Errors/Exceptions: $PY_ERRORS"

# Check for common issues
echo ""
echo "COMMON ISSUES:"
echo "--------------"
ISSUES_FOUND=0

if grep -q "CUDA out of memory" "$PYTHON_LOG" 2>/dev/null; then
    echo "  ERROR: GPU OOM detected - consider reducing batch size or enabling CPU offload"
    ISSUES_FOUND=1
fi

if grep -q "connect to socket.*failed" "$LOG_FILE" 2>/dev/null; then
    echo "  ERROR: Socket connection failures - Python service may not be running"
    ISSUES_FOUND=1
fi

if grep -q "health check failed" "$LOG_FILE" 2>/dev/null; then
    echo "  ERROR: Health check failures - service may be unhealthy"
    ISSUES_FOUND=1
fi

if [ "$PROGRESS_COUNT" -eq 0 ] && [ "$EMBEDDED" -eq 0 ] && [ "$SCHEDULED" -gt 0 ]; then
    echo "  WARNING: No progress logs found - embeddings may not have started"
    ISSUES_FOUND=1
fi

if [ "$FAILED" -gt "$(($EMBEDDED / 10))" ] 2>/dev/null && [ "$FAILED" -gt 5 ]; then
    echo "  WARNING: High failure rate (failed: $FAILED, succeeded: $EMBEDDED)"
    ISSUES_FOUND=1
fi

MODEL_LOAD_ERRORS=$(grep -c "MuQMuLanConfig\|missing.*required.*arguments\|trust_remote_code" "$PYTHON_LOG" || echo "0")
if [ "$MODEL_LOAD_ERRORS" -gt 0 ]; then
    echo "  ERROR: Model loading failures detected ($MODEL_LOAD_ERRORS occurrences)"
    ISSUES_FOUND=1
fi

if [ "$ISSUES_FOUND" -eq 0 ]; then
    echo "  No critical issues detected"
fi

echo ""
echo "=========================================="
