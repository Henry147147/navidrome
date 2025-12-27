"""
Manages request batching for the embedding server to minimize GPU model swaps.

Instead of processing each request immediately (causing GPU model swaps per request),
this module accumulates requests and processes them in batches. This allows the
batch processor to use a model-first strategy: process all tracks with MuQ, then
all with Flamingo, then all with Qwen3 - reducing GPU swaps from O(N*3) to O(3).
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from queue import Queue
from typing import Callable, Dict, List, Optional


@dataclass
class PendingRequest:
    """Represents a queued embedding request waiting for batch processing."""

    request_id: str
    payload: dict
    result_queue: Queue  # Single-item queue for result delivery
    enqueue_time: float = field(default_factory=time.monotonic)


@dataclass
class BatchResult:
    """Result for a single request within a batch."""

    request_id: str
    success: bool
    payload: dict  # Response payload
    error: Optional[str] = None


class BatchQueueManager:
    """
    Accumulates embedding requests and processes them in model-optimized batches.

    Trigger conditions (any one):
    - Timeout reached (default: 5 seconds since first request)
    - Count threshold reached (default: 50 requests)
    - Explicit flush signal received

    Each request's connection is held open until its batch completes.
    The caller blocks on the returned Queue until the result is ready.
    """

    def __init__(
        self,
        *,
        batch_timeout_seconds: float = 5.0,
        batch_size_threshold: int = 50,
        process_batch_fn: Callable[[List[dict]], Dict[str, BatchResult]],
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the batch queue manager.

        Args:
            batch_timeout_seconds: Max time to wait after first request before processing
            batch_size_threshold: Max requests to accumulate before forcing batch
            process_batch_fn: Callback that processes a list of payloads and returns
                              results keyed by request_id
            logger: Optional logger instance
        """
        self.batch_timeout = batch_timeout_seconds
        self.batch_size_threshold = batch_size_threshold
        self.process_batch_fn = process_batch_fn
        self.logger = logger or logging.getLogger(__name__)

        self._lock = threading.Lock()
        self._pending: List[PendingRequest] = []
        self._batch_start_time: Optional[float] = None
        self._timer: Optional[threading.Timer] = None
        self._processing = False
        self._shutdown = False

    def enqueue(self, request_id: str, payload: dict) -> Queue:
        """
        Add a request to the batch queue.

        Returns a Queue that will receive the BatchResult when processing completes.
        Caller should block on queue.get() to wait for result.

        Args:
            request_id: Unique identifier for this request
            payload: The embedding request payload

        Returns:
            Queue that will receive a single BatchResult when ready
        """
        result_queue: Queue = Queue(maxsize=1)

        with self._lock:
            if self._shutdown:
                # Return error immediately if shutting down
                result_queue.put(
                    BatchResult(
                        request_id=request_id,
                        success=False,
                        payload={"status": "error", "message": "Server shutting down"},
                        error="Server shutting down",
                    )
                )
                return result_queue

            request = PendingRequest(
                request_id=request_id,
                payload=payload,
                result_queue=result_queue,
            )
            self._pending.append(request)

            self.logger.debug(
                "Enqueued request %s, pending count: %d",
                request_id,
                len(self._pending),
            )

            # Start timer on first request
            if self._batch_start_time is None:
                self._batch_start_time = time.monotonic()
                self._schedule_timeout()

            # Check if threshold reached
            if len(self._pending) >= self.batch_size_threshold:
                self.logger.info(
                    "Batch size threshold reached (%d), triggering batch",
                    len(self._pending),
                )
                self._trigger_batch_locked()

        return result_queue

    def flush(self) -> None:
        """Force immediate processing of pending requests."""
        with self._lock:
            if self._pending and not self._processing:
                self.logger.info(
                    "Flush requested, processing %d pending requests",
                    len(self._pending),
                )
                self._trigger_batch_locked()
            elif self._processing:
                self.logger.debug("Flush requested but batch already processing")
            else:
                self.logger.debug("Flush requested but no pending requests")

    def shutdown(self) -> None:
        """Stop accepting new requests and process any pending."""
        with self._lock:
            self._shutdown = True
            if self._timer:
                self._timer.cancel()
                self._timer = None
            if self._pending and not self._processing:
                self.logger.info(
                    "Shutdown: processing %d remaining requests", len(self._pending)
                )
                self._trigger_batch_locked()

    def pending_count(self) -> int:
        """Return the number of pending requests."""
        with self._lock:
            return len(self._pending)

    def is_processing(self) -> bool:
        """Return whether a batch is currently being processed."""
        with self._lock:
            return self._processing

    def _schedule_timeout(self) -> None:
        """Schedule timeout-triggered batch processing."""
        if self._timer:
            self._timer.cancel()
        self._timer = threading.Timer(self.batch_timeout, self._on_timeout)
        self._timer.daemon = True
        self._timer.start()
        self.logger.debug("Scheduled batch timeout in %.1fs", self.batch_timeout)

    def _on_timeout(self) -> None:
        """Called when batch timeout expires."""
        with self._lock:
            if self._pending and not self._processing:
                self.logger.info(
                    "Batch timeout reached, processing %d requests", len(self._pending)
                )
                self._trigger_batch_locked()

    def _trigger_batch_locked(self) -> None:
        """Start batch processing (must hold lock)."""
        if self._processing or not self._pending:
            return

        self._processing = True
        if self._timer:
            self._timer.cancel()
            self._timer = None

        # Take snapshot of pending requests
        batch = self._pending[:]
        self._pending = []
        self._batch_start_time = None

        # Process in background thread to not block new requests
        thread = threading.Thread(
            target=self._process_batch,
            args=(batch,),
            daemon=True,
            name="BatchProcessor",
        )
        thread.start()

    def _process_batch(self, batch: List[PendingRequest]) -> None:
        """Process a batch and deliver results to waiting callers."""
        request_ids = [r.request_id for r in batch]
        payloads = [r.payload for r in batch]

        self.logger.info(
            "Processing batch of %d requests: %s%s",
            len(batch),
            request_ids[:3],
            "..." if len(request_ids) > 3 else "",
        )

        start_time = time.monotonic()

        try:
            # Call the actual batch processor
            results = self.process_batch_fn(payloads)
        except Exception as e:
            self.logger.exception("Batch processing failed")
            # Deliver error to all waiting requests
            results = {
                req.request_id: BatchResult(
                    request_id=req.request_id,
                    success=False,
                    payload={"status": "error", "message": str(e)},
                    error=str(e),
                )
                for req in batch
            }

        elapsed = time.monotonic() - start_time

        # Deliver results to each waiting connection
        success_count = 0
        error_count = 0
        for request in batch:
            result = results.get(request.request_id)
            if result is None:
                result = BatchResult(
                    request_id=request.request_id,
                    success=False,
                    payload={"status": "error", "message": "No result returned"},
                    error="No result returned",
                )
            request.result_queue.put(result)
            if result.success:
                success_count += 1
            else:
                error_count += 1

        self.logger.info(
            "Batch complete: %d requests in %.1fs (%d success, %d errors)",
            len(batch),
            elapsed,
            success_count,
            error_count,
        )

        with self._lock:
            self._processing = False
            # If new requests accumulated during processing, schedule next batch
            if self._pending:
                self.logger.debug(
                    "%d requests accumulated during processing, scheduling next batch",
                    len(self._pending),
                )
                self._batch_start_time = time.monotonic()
                self._schedule_timeout()
