import threading
import time

from application_builder import (
    ApplicationBuilder,
    Worker,
    TimedWorker,
    CancellationToken,
    CancellationTokenSource,
)


class _SimpleWorker(Worker):
    """Worker that records stopping_token state and waits for stop."""

    def __init__(self):
        super().__init__()
        self.token_before_stop: CancellationToken = None
        self.token_cancelled_before_stop: bool = None
        self.token_cancelled_after_stop: bool = None
        self._execute_called = threading.Event()
        self._stop_detected = threading.Event()

    def execute(self):
        self.token_before_stop = self.stopping_token
        self.token_cancelled_before_stop = self.stopping_token.is_cancellation_requested
        self._execute_called.set()
        self.wait_for_stop()
        self.token_cancelled_after_stop = self.stopping_token.is_cancellation_requested
        self._stop_detected.set()


class _CallbackTrackingWorker(Worker):
    """Worker that registers a callback on stopping_token."""

    def __init__(self):
        super().__init__()
        self.callback_fired = threading.Event()
        self._execute_called = threading.Event()

    def execute(self):
        self.stopping_token.register(lambda: self.callback_fired.set())
        self._execute_called.set()
        self.wait_for_stop()


class _TimedStoppingWorker(TimedWorker):
    """TimedWorker that exposes stopping_token state."""

    def __init__(self):
        super().__init__(interval_seconds=60)
        self.token_available = threading.Event()
        self.token_not_cancelled: bool = None

    def do_work(self):
        self.token_not_cancelled = not self.stopping_token.is_cancellation_requested
        self.token_available.set()
        self.wait_for_stop(60)


class TestWorkerStoppingToken:
    """Tests for Worker.stopping_token property."""

    def test_stopping_token_returns_cancellation_token(self):
        worker = _SimpleWorker()
        worker.start()
        worker._execute_called.wait(timeout=5)
        assert isinstance(worker.stopping_token, CancellationToken)
        worker.stop()

    def test_stopping_token_not_cancelled_before_stop(self):
        worker = _SimpleWorker()
        worker.start()
        worker._execute_called.wait(timeout=5)
        assert worker.token_cancelled_before_stop is False
        worker.stop()

    def test_stopping_token_cancelled_after_stop(self):
        worker = _SimpleWorker()
        worker.start()
        worker._execute_called.wait(timeout=5)
        worker.stop()
        worker._stop_detected.wait(timeout=5)
        assert worker.token_cancelled_after_stop is True

    def test_stopping_token_fires_callback_on_stop(self):
        worker = _CallbackTrackingWorker()
        worker.start()
        worker._execute_called.wait(timeout=5)
        assert not worker.callback_fired.is_set()
        worker.stop()
        assert worker.callback_fired.wait(timeout=5)

    def test_stopping_token_resets_on_restart(self):
        worker = _SimpleWorker()

        worker.start()
        worker._execute_called.wait(timeout=5)
        first_token = worker.stopping_token
        worker.stop()
        worker._stop_detected.wait(timeout=5)
        assert first_token.is_cancellation_requested is True

        worker._execute_called.clear()
        worker._stop_detected.clear()
        worker.start()
        worker._execute_called.wait(timeout=5)
        second_token = worker.stopping_token
        assert second_token.is_cancellation_requested is False
        assert second_token is not first_token
        worker.stop()

    def test_timed_worker_has_stopping_token(self):
        worker = _TimedStoppingWorker()
        worker.start()
        worker.token_available.wait(timeout=5)
        assert worker.token_not_cancelled is True
        assert isinstance(worker.stopping_token, CancellationToken)
        worker.stop()
        assert worker.stopping_token.is_cancellation_requested is True

    def test_stopping_token_available_before_start(self):
        worker = _SimpleWorker()
        token = worker.stopping_token
        assert isinstance(token, CancellationToken)
        assert token.is_cancellation_requested is False
