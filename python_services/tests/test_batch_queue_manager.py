"""
Tests for the batch queue manager module.
"""

import threading
import time
from typing import Dict, List
from unittest.mock import MagicMock

import pytest

from batch_queue_manager import BatchQueueManager, BatchResult, PendingRequest


class TestBatchResult:
    """Tests for the BatchResult dataclass."""

    def test_success_result(self):
        result = BatchResult(
            request_id="req-1",
            success=True,
            payload={"status": "ok", "data": [1, 2, 3]},
        )
        assert result.request_id == "req-1"
        assert result.success is True
        assert result.payload["status"] == "ok"
        assert result.error is None

    def test_failure_result(self):
        result = BatchResult(
            request_id="req-2",
            success=False,
            payload={"status": "error", "message": "Something went wrong"},
            error="Something went wrong",
        )
        assert result.request_id == "req-2"
        assert result.success is False
        assert result.error == "Something went wrong"


class TestPendingRequest:
    """Tests for the PendingRequest dataclass."""

    def test_pending_request_creation(self):
        from queue import Queue

        q = Queue()
        req = PendingRequest(
            request_id="test-123",
            payload={"music_file": "/path/to/file.mp3"},
            result_queue=q,
        )
        assert req.request_id == "test-123"
        assert req.payload["music_file"] == "/path/to/file.mp3"
        assert req.result_queue is q
        assert req.enqueue_time > 0


class TestBatchQueueManager:
    """Tests for the BatchQueueManager class."""

    def test_enqueue_returns_queue(self):
        """Test that enqueue returns a Queue for receiving results."""

        def process_batch(payloads: List[dict]) -> Dict[str, BatchResult]:
            return {
                p["request_id"]: BatchResult(
                    request_id=p["request_id"],
                    success=True,
                    payload={"status": "ok"},
                )
                for p in payloads
            }

        manager = BatchQueueManager(
            batch_timeout_seconds=10.0,
            batch_size_threshold=100,
            process_batch_fn=process_batch,
        )

        result_queue = manager.enqueue("req-1", {"request_id": "req-1", "data": "test"})
        assert result_queue is not None
        assert manager.pending_count() == 1

        manager.shutdown()

    def test_count_threshold_triggers_batch(self):
        """Test that reaching count threshold triggers batch processing."""
        processed_batches: List[List[dict]] = []

        def process_batch(payloads: List[dict]) -> Dict[str, BatchResult]:
            processed_batches.append(payloads)
            return {
                p["request_id"]: BatchResult(
                    request_id=p["request_id"],
                    success=True,
                    payload={"status": "ok"},
                )
                for p in payloads
            }

        manager = BatchQueueManager(
            batch_timeout_seconds=60.0,  # Long timeout so it doesn't trigger
            batch_size_threshold=3,
            process_batch_fn=process_batch,
        )

        # Enqueue 3 requests (threshold)
        queues = []
        for i in range(3):
            q = manager.enqueue(f"req-{i}", {"request_id": f"req-{i}"})
            queues.append(q)

        # Wait for results
        results = [q.get(timeout=5) for q in queues]

        assert len(processed_batches) == 1
        assert len(processed_batches[0]) == 3
        assert all(r.success for r in results)

        manager.shutdown()

    def test_timeout_triggers_batch(self):
        """Test that timeout triggers batch processing."""
        processed_batches: List[List[dict]] = []
        processing_event = threading.Event()

        def process_batch(payloads: List[dict]) -> Dict[str, BatchResult]:
            processed_batches.append(payloads)
            processing_event.set()
            return {
                p["request_id"]: BatchResult(
                    request_id=p["request_id"],
                    success=True,
                    payload={"status": "ok"},
                )
                for p in payloads
            }

        manager = BatchQueueManager(
            batch_timeout_seconds=0.2,  # Short timeout
            batch_size_threshold=100,  # High threshold so it doesn't trigger
            process_batch_fn=process_batch,
        )

        # Enqueue 1 request (below threshold)
        q = manager.enqueue("req-1", {"request_id": "req-1"})

        # Wait for timeout to trigger
        assert processing_event.wait(timeout=5.0)
        result = q.get(timeout=1)

        assert len(processed_batches) == 1
        assert result.success

        manager.shutdown()

    def test_flush_triggers_batch(self):
        """Test that flush() triggers immediate batch processing."""
        processed_batches: List[List[dict]] = []

        def process_batch(payloads: List[dict]) -> Dict[str, BatchResult]:
            processed_batches.append(payloads)
            return {
                p["request_id"]: BatchResult(
                    request_id=p["request_id"],
                    success=True,
                    payload={"status": "ok"},
                )
                for p in payloads
            }

        manager = BatchQueueManager(
            batch_timeout_seconds=60.0,  # Long timeout
            batch_size_threshold=100,  # High threshold
            process_batch_fn=process_batch,
        )

        # Enqueue requests
        queues = []
        for i in range(2):
            q = manager.enqueue(f"req-{i}", {"request_id": f"req-{i}"})
            queues.append(q)

        # Trigger flush
        manager.flush()

        # Wait for results
        results = [q.get(timeout=5) for q in queues]

        assert len(processed_batches) == 1
        assert len(processed_batches[0]) == 2
        assert all(r.success for r in results)

        manager.shutdown()

    def test_shutdown_processes_pending(self):
        """Test that shutdown() processes any pending requests."""
        processed_batches: List[List[dict]] = []

        def process_batch(payloads: List[dict]) -> Dict[str, BatchResult]:
            processed_batches.append(payloads)
            return {
                p["request_id"]: BatchResult(
                    request_id=p["request_id"],
                    success=True,
                    payload={"status": "ok"},
                )
                for p in payloads
            }

        manager = BatchQueueManager(
            batch_timeout_seconds=60.0,
            batch_size_threshold=100,
            process_batch_fn=process_batch,
        )

        # Enqueue requests
        queues = []
        for i in range(2):
            q = manager.enqueue(f"req-{i}", {"request_id": f"req-{i}"})
            queues.append(q)

        # Shutdown should process pending
        manager.shutdown()

        # Results should be available
        results = [q.get(timeout=5) for q in queues]
        assert all(r.success for r in results)

    def test_enqueue_after_shutdown_returns_error(self):
        """Test that enqueueing after shutdown returns an error result."""

        def process_batch(payloads: List[dict]) -> Dict[str, BatchResult]:
            return {}

        manager = BatchQueueManager(
            batch_timeout_seconds=60.0,
            batch_size_threshold=100,
            process_batch_fn=process_batch,
        )

        manager.shutdown()

        # Enqueue after shutdown
        q = manager.enqueue("req-1", {"request_id": "req-1"})
        result = q.get(timeout=1)

        assert result.success is False
        assert "shutting down" in result.error.lower()

    def test_process_batch_error_propagates_to_all_requests(self):
        """Test that errors in process_batch are delivered to all waiting requests."""

        def process_batch(payloads: List[dict]) -> Dict[str, BatchResult]:
            raise RuntimeError("Processing failed!")

        manager = BatchQueueManager(
            batch_timeout_seconds=0.1,
            batch_size_threshold=100,
            process_batch_fn=process_batch,
        )

        # Enqueue requests
        queues = []
        for i in range(2):
            q = manager.enqueue(f"req-{i}", {"request_id": f"req-{i}"})
            queues.append(q)

        # Wait for timeout to trigger processing
        results = [q.get(timeout=5) for q in queues]

        assert all(not r.success for r in results)
        assert all("Processing failed" in r.error for r in results)

        manager.shutdown()

    def test_missing_result_returns_error(self):
        """Test that missing results in process_batch response return errors."""

        def process_batch(payloads: List[dict]) -> Dict[str, BatchResult]:
            # Only return result for first request
            if payloads:
                p = payloads[0]
                return {
                    p["request_id"]: BatchResult(
                        request_id=p["request_id"],
                        success=True,
                        payload={"status": "ok"},
                    )
                }
            return {}

        manager = BatchQueueManager(
            batch_timeout_seconds=0.1,
            batch_size_threshold=100,
            process_batch_fn=process_batch,
        )

        q1 = manager.enqueue("req-1", {"request_id": "req-1"})
        q2 = manager.enqueue("req-2", {"request_id": "req-2"})

        result1 = q1.get(timeout=5)
        result2 = q2.get(timeout=5)

        assert result1.success is True
        assert result2.success is False
        assert "No result returned" in result2.error

        manager.shutdown()

    def test_pending_count_and_is_processing(self):
        """Test pending_count() and is_processing() methods."""
        processing_started = threading.Event()
        processing_continue = threading.Event()

        def process_batch(payloads: List[dict]) -> Dict[str, BatchResult]:
            processing_started.set()
            processing_continue.wait(timeout=5)
            return {
                p["request_id"]: BatchResult(
                    request_id=p["request_id"],
                    success=True,
                    payload={"status": "ok"},
                )
                for p in payloads
            }

        manager = BatchQueueManager(
            batch_timeout_seconds=0.1,
            batch_size_threshold=100,
            process_batch_fn=process_batch,
        )

        # Initially no pending
        assert manager.pending_count() == 0
        assert manager.is_processing() is False

        # Enqueue and wait for processing to start
        q = manager.enqueue("req-1", {"request_id": "req-1"})
        processing_started.wait(timeout=5)

        # During processing
        assert manager.is_processing() is True

        # Allow processing to complete
        processing_continue.set()
        q.get(timeout=5)

        # After processing
        assert manager.is_processing() is False

        manager.shutdown()

    def test_concurrent_enqueue(self):
        """Test that concurrent enqueue operations are thread-safe."""

        def process_batch(payloads: List[dict]) -> Dict[str, BatchResult]:
            return {
                p["request_id"]: BatchResult(
                    request_id=p["request_id"],
                    success=True,
                    payload={"status": "ok"},
                )
                for p in payloads
            }

        manager = BatchQueueManager(
            batch_timeout_seconds=0.5,
            batch_size_threshold=100,
            process_batch_fn=process_batch,
        )

        queues = []
        threads = []

        def enqueue_task(request_id):
            q = manager.enqueue(request_id, {"request_id": request_id})
            queues.append((request_id, q))

        # Spawn 10 threads to enqueue concurrently
        for i in range(10):
            t = threading.Thread(target=enqueue_task, args=(f"req-{i}",))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Flush and wait for results
        manager.flush()

        results = []
        for request_id, q in queues:
            result = q.get(timeout=5)
            results.append(result)

        assert len(results) == 10
        assert all(r.success for r in results)

        manager.shutdown()

    def test_requests_accumulate_during_processing(self):
        """Test that new requests queue up while a batch is processing."""
        batch_count = 0
        processing_started = threading.Event()
        processing_continue = threading.Event()

        def process_batch(payloads: List[dict]) -> Dict[str, BatchResult]:
            nonlocal batch_count
            batch_count += 1
            if batch_count == 1:
                processing_started.set()
                processing_continue.wait(timeout=5)
            return {
                p["request_id"]: BatchResult(
                    request_id=p["request_id"],
                    success=True,
                    payload={"status": "ok", "batch": batch_count},
                )
                for p in payloads
            }

        manager = BatchQueueManager(
            batch_timeout_seconds=0.1,
            batch_size_threshold=100,
            process_batch_fn=process_batch,
        )

        # Enqueue first request and wait for processing to start
        q1 = manager.enqueue("req-1", {"request_id": "req-1"})
        processing_started.wait(timeout=5)

        # Enqueue more requests while first batch is processing
        q2 = manager.enqueue("req-2", {"request_id": "req-2"})
        q3 = manager.enqueue("req-3", {"request_id": "req-3"})

        # Allow first batch to complete
        processing_continue.set()

        # Get all results
        r1 = q1.get(timeout=5)
        r2 = q2.get(timeout=5)
        r3 = q3.get(timeout=5)

        # First request in first batch
        assert r1.payload["batch"] == 1
        # Second and third in second batch
        assert r2.payload["batch"] == 2
        assert r3.payload["batch"] == 2

        manager.shutdown()
