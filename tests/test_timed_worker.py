import threading
import time

from application_builder import Worker, TimedWorker, WorkerState, CancellationToken


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

class _SimpleWorker(Worker):
    """Worker that waits for stop signal in execute."""

    def __init__(self):
        super().__init__()
        self._execute_entered = threading.Event()

    def execute(self):
        self._execute_entered.set()
        self.wait_for_stop()


class _FailingWorker(Worker):
    """Worker whose execute raises immediately."""

    def __init__(self):
        super().__init__()
        self._execute_entered = threading.Event()

    def execute(self):
        self._execute_entered.set()
        raise RuntimeError("intentional failure")


class _CountingTimedWorker(TimedWorker):
    """TimedWorker that counts do_work invocations."""

    def __init__(self, interval_seconds: float = 0.05):
        super().__init__(interval_seconds=interval_seconds)
        self.call_count = 0
        self._at_least = threading.Event()
        self._at_least_target = 0
        self._lock = threading.Lock()

    def do_work(self):
        with self._lock:
            self.call_count += 1
            if self.call_count >= self._at_least_target:
                self._at_least.set()

    def wait_for_calls(self, n: int, timeout: float = 5.0) -> bool:
        with self._lock:
            self._at_least_target = n
            self._at_least.clear()
            if self.call_count >= n:
                return True
        return self._at_least.wait(timeout=timeout)


class _FailingTimedWorker(TimedWorker):
    """TimedWorker whose do_work raises on the first call then succeeds."""

    def __init__(self, interval_seconds: float = 0.05):
        super().__init__(interval_seconds=interval_seconds)
        self.call_count = 0
        self._lock = threading.Lock()
        self._at_least = threading.Event()
        self._at_least_target = 0

    def do_work(self):
        with self._lock:
            self.call_count += 1
            if self.call_count >= self._at_least_target:
                self._at_least.set()
        if self.call_count == 1:
            raise RuntimeError("first call fails")

    def wait_for_calls(self, n: int, timeout: float = 5.0) -> bool:
        with self._lock:
            self._at_least_target = n
            self._at_least.clear()
            if self.call_count >= n:
                return True
        return self._at_least.wait(timeout=timeout)


class _RestartTrackingWorker(Worker):
    """Worker that records thread ids across restarts."""

    def __init__(self):
        super().__init__()
        self.thread_ids: list[int] = []
        self._execute_entered = threading.Event()

    def execute(self):
        self.thread_ids.append(threading.current_thread().ident)
        self._execute_entered.set()
        self.wait_for_stop()


# ---------------------------------------------------------------------------
# Tests: Worker lifecycle states
# ---------------------------------------------------------------------------

class TestWorkerLifecycleStates:

    def test_initial_state_is_created(self):
        worker = _SimpleWorker()
        assert worker._state == WorkerState.CREATED

    def test_state_running_after_start(self):
        worker = _SimpleWorker()
        worker.start()
        worker._execute_entered.wait(timeout=5)
        assert worker._state == WorkerState.RUNNING
        worker.stop()

    def test_state_stopped_after_stop(self):
        worker = _SimpleWorker()
        worker.start()
        worker._execute_entered.wait(timeout=5)
        worker.stop()
        assert worker._state == WorkerState.STOPPED


# ---------------------------------------------------------------------------
# Tests: Worker start idempotent
# ---------------------------------------------------------------------------

class TestWorkerStartIdempotent:

    def test_second_start_while_running_does_nothing(self):
        worker = _SimpleWorker()
        worker.start()
        worker._execute_entered.wait(timeout=5)
        first_thread = worker._thread
        worker.start()
        assert worker._thread is first_thread
        worker.stop()


# ---------------------------------------------------------------------------
# Tests: Worker stop when not running
# ---------------------------------------------------------------------------

class TestWorkerStopWhenNotRunning:

    def test_stop_on_created_does_nothing(self):
        worker = _SimpleWorker()
        assert worker._state == WorkerState.CREATED
        worker.stop()
        assert worker._state == WorkerState.CREATED

    def test_stop_on_already_stopped_does_nothing(self):
        worker = _SimpleWorker()
        worker.start()
        worker._execute_entered.wait(timeout=5)
        worker.stop()
        assert worker._state == WorkerState.STOPPED
        worker.stop()
        assert worker._state == WorkerState.STOPPED


# ---------------------------------------------------------------------------
# Tests: Worker FAILED state
# ---------------------------------------------------------------------------

class TestWorkerFailedState:

    def test_exception_in_execute_sets_failed(self):
        worker = _FailingWorker()
        worker.start()
        worker._execute_entered.wait(timeout=5)
        # Give thread time to propagate exception and set state
        deadline = time.time() + 5
        while worker._state != WorkerState.FAILED and time.time() < deadline:
            time.sleep(0.01)
        assert worker._state == WorkerState.FAILED


# ---------------------------------------------------------------------------
# Tests: Worker restart
# ---------------------------------------------------------------------------

class TestWorkerRestart:

    def test_start_stop_start_uses_new_thread(self):
        worker = _RestartTrackingWorker()

        worker.start()
        worker._execute_entered.wait(timeout=5)
        first_thread = worker._thread
        first_cts = worker._stopping_cts
        worker.stop()

        worker._execute_entered.clear()
        worker.start()
        worker._execute_entered.wait(timeout=5)
        second_thread = worker._thread
        second_cts = worker._stopping_cts

        assert first_thread is not second_thread
        assert first_cts is not second_cts
        assert len(worker.thread_ids) == 2
        worker.stop()


# ---------------------------------------------------------------------------
# Tests: TimedWorker do_work called repeatedly
# ---------------------------------------------------------------------------

class TestTimedWorkerRepeatedCalls:

    def test_do_work_called_multiple_times(self):
        worker = _CountingTimedWorker(interval_seconds=0.05)
        worker.start()
        assert worker.wait_for_calls(3, timeout=5)
        worker.stop()
        assert worker.call_count >= 3

    def test_calls_within_expected_timeframe(self):
        interval = 0.1
        worker = _CountingTimedWorker(interval_seconds=interval)
        worker.start()
        start = time.time()
        assert worker.wait_for_calls(3, timeout=5)
        elapsed = time.time() - start
        worker.stop()
        # 3 calls need at least ~2 intervals of wait (first call is immediate)
        assert elapsed < interval * 10, f"Took too long: {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# Tests: TimedWorker stops on stop()
# ---------------------------------------------------------------------------

class TestTimedWorkerStop:

    def test_stops_when_stop_called(self):
        worker = _CountingTimedWorker(interval_seconds=0.05)
        worker.start()
        assert worker.wait_for_calls(1, timeout=5)
        worker.stop()
        assert worker._state == WorkerState.STOPPED
        count_at_stop = worker.call_count
        time.sleep(0.15)
        assert worker.call_count == count_at_stop


# ---------------------------------------------------------------------------
# Tests: TimedWorker exception resilience
# ---------------------------------------------------------------------------

class TestTimedWorkerExceptionResilience:

    def test_exception_in_do_work_does_not_stop_loop(self):
        worker = _FailingTimedWorker(interval_seconds=0.05)
        worker.start()
        assert worker.wait_for_calls(3, timeout=5)
        worker.stop()
        assert worker.call_count >= 3
        assert worker._state == WorkerState.STOPPED


# ---------------------------------------------------------------------------
# Tests: TimedWorker immediate stop
# ---------------------------------------------------------------------------

class TestTimedWorkerImmediateStop:

    def test_stop_during_wait_wakes_immediately(self):
        worker = _CountingTimedWorker(interval_seconds=10.0)
        worker.start()
        assert worker.wait_for_calls(1, timeout=5)
        start = time.time()
        worker.stop()
        elapsed = time.time() - start
        assert elapsed < 2.0, f"Stop took too long: {elapsed:.2f}s"
        assert worker._state == WorkerState.STOPPED


# ---------------------------------------------------------------------------
# Tests: Worker is_stopping flag
# ---------------------------------------------------------------------------

class TestWorkerIsStopping:

    def test_is_stopping_false_before_stop(self):
        worker = _SimpleWorker()
        worker.start()
        worker._execute_entered.wait(timeout=5)
        assert worker.is_stopping() is False
        worker.stop()

    def test_is_stopping_true_after_stop(self):
        worker = _SimpleWorker()
        worker.start()
        worker._execute_entered.wait(timeout=5)
        worker.stop()
        assert worker.is_stopping() is True


# ---------------------------------------------------------------------------
# Tests: Worker wait_for_stop with timeout
# ---------------------------------------------------------------------------

class TestWorkerWaitForStop:

    def test_returns_true_when_stopped(self):
        worker = _SimpleWorker()
        worker.start()
        worker._execute_entered.wait(timeout=5)

        result_holder: list[bool] = []
        done = threading.Event()

        def waiter():
            result_holder.append(worker.wait_for_stop(timeout_seconds=5))
            done.set()

        t = threading.Thread(target=waiter, daemon=True)
        t.start()
        time.sleep(0.05)
        worker.stop()
        done.wait(timeout=5)
        assert result_holder[0] is True

    def test_returns_false_on_timeout(self):
        worker = _SimpleWorker()
        assert worker.wait_for_stop(timeout_seconds=0.05) is False
