import threading
import time

import pytest

from application_builder import (
    CancellationToken,
    CancellationTokenSource,
    JobManager,
)


class _NoOpLogger:
    def trace(self, *a, **kw): pass
    def debug(self, *a, **kw): pass
    def info(self, *a, **kw): pass
    def success(self, *a, **kw): pass
    def warning(self, *a, **kw): pass
    def error(self, *a, **kw): pass
    def critical(self, *a, **kw): pass
    def begin_scope(self, **kw): pass
    def with_context(self, ctx): return self


@pytest.fixture()
def logger():
    return _NoOpLogger()


@pytest.fixture()
def jm(logger):
    return JobManager(logger)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------
class TestConstructor:
    def test_valid_max_concurrent(self, logger):
        jm = JobManager(logger, max_concurrent=2)
        assert jm is not None

    def test_zero_max_concurrent_raises(self, logger):
        with pytest.raises(ValueError):
            JobManager(logger, max_concurrent=0)

    def test_negative_max_concurrent_raises(self, logger):
        with pytest.raises(ValueError):
            JobManager(logger, max_concurrent=-1)


# ---------------------------------------------------------------------------
# start_job basics
# ---------------------------------------------------------------------------
class TestStartJob:
    def test_returns_string_id(self, jm):
        job_id = jm.start_job(lambda: None)
        assert isinstance(job_id, str)
        assert len(job_id) > 0
        jm.wait(job_id, timeout=2)

    def test_not_callable_raises(self, jm):
        with pytest.raises(TypeError):
            jm.start_job(42)

    def test_unique_ids(self, jm):
        ids = [jm.start_job(lambda: None) for _ in range(5)]
        assert len(set(ids)) == 5
        for jid in ids:
            jm.wait(jid, timeout=2)


# ---------------------------------------------------------------------------
# Job execution: args, kwargs, provide_token
# ---------------------------------------------------------------------------
class TestJobExecution:
    def test_func_receives_args(self, jm):
        results = []
        done = threading.Event()

        def worker(a, b):
            results.append((a, b))
            done.set()

        jm.start_job(worker, 1, 2)
        assert done.wait(timeout=2)
        assert results == [(1, 2)]

    def test_func_receives_kwargs(self, jm):
        results = []
        done = threading.Event()

        def worker(x=0, y=0):
            results.append((x, y))
            done.set()

        jm.start_job(worker, x=10, y=20)
        assert done.wait(timeout=2)
        assert results == [(10, 20)]

    def test_provide_token_injects_cancellation_token(self, jm):
        received = []
        done = threading.Event()

        def worker(token, extra):
            received.append((token, extra))
            done.set()

        jm.start_job(worker, "hello", provide_token=True)
        assert done.wait(timeout=2)
        assert len(received) == 1
        assert isinstance(received[0][0], CancellationToken)
        assert received[0][1] == "hello"


# ---------------------------------------------------------------------------
# wait
# ---------------------------------------------------------------------------
class TestWait:
    def test_returns_true_when_completed(self, jm):
        job_id = jm.start_job(lambda: None)
        assert jm.wait(job_id, timeout=2) is True

    def test_returns_true_for_already_completed_job(self, jm):
        job_id = jm.start_job(lambda: None)
        jm.wait(job_id, timeout=2)
        assert jm.wait(job_id, timeout=2) is True

    def test_timeout_returns_false_if_still_running(self, jm):
        blocker = threading.Event()

        def slow():
            blocker.wait(timeout=5)

        job_id = jm.start_job(slow)
        result = jm.wait(job_id, timeout=0.2)
        assert result is False
        blocker.set()
        jm.wait(job_id, timeout=2)


# ---------------------------------------------------------------------------
# is_running
# ---------------------------------------------------------------------------
class TestIsRunning:
    def test_true_while_running(self, jm):
        started = threading.Event()
        blocker = threading.Event()

        def worker():
            started.set()
            blocker.wait(timeout=5)

        job_id = jm.start_job(worker)
        assert started.wait(timeout=2)
        assert jm.is_running(job_id) is True
        blocker.set()
        jm.wait(job_id, timeout=2)

    def test_false_after_completion(self, jm):
        job_id = jm.start_job(lambda: None)
        jm.wait(job_id, timeout=2)
        assert jm.is_running(job_id) is False

    def test_false_for_unknown_id(self, jm):
        assert jm.is_running("nonexistent") is False


# ---------------------------------------------------------------------------
# cancel_job
# ---------------------------------------------------------------------------
class TestCancelJob:
    def test_signals_cancellation_via_token(self, jm):
        cancelled_flag = threading.Event()

        def worker(token):
            while not token.is_cancellation_requested:
                time.sleep(0.01)
            cancelled_flag.set()

        job_id = jm.start_job(worker, provide_token=True)
        time.sleep(0.1)
        jm.cancel_job(job_id)
        assert cancelled_flag.wait(timeout=2)
        jm.wait(job_id, timeout=2)

    def test_cancel_with_wait_joins_thread(self, jm):
        blocker = threading.Event()

        def worker(token):
            while not token.is_cancellation_requested:
                time.sleep(0.01)
            blocker.set()

        job_id = jm.start_job(worker, provide_token=True)
        time.sleep(0.1)
        result = jm.cancel_job(job_id, wait=True, timeout=2)
        assert result is True
        assert blocker.is_set()

    def test_cancel_nonexistent_returns_false(self, jm):
        assert jm.cancel_job("nonexistent") is False

    def test_cancel_job_nowait(self, jm):
        blocker = threading.Event()

        def worker(token):
            while not token.is_cancellation_requested:
                time.sleep(0.01)
            blocker.set()

        job_id = jm.start_job(worker, provide_token=True)
        time.sleep(0.1)
        result = jm.cancel_job_nowait(job_id)
        assert result is True
        assert blocker.wait(timeout=2)
        jm.wait(job_id, timeout=2)


# ---------------------------------------------------------------------------
# cancel_all
# ---------------------------------------------------------------------------
class TestCancelAll:
    def test_cancels_all_running_jobs(self, jm):
        flags = [threading.Event() for _ in range(3)]

        def worker(token, idx):
            while not token.is_cancellation_requested:
                time.sleep(0.01)
            flags[idx].set()

        job_ids = []
        for i in range(3):
            jid = jm.start_job(worker, i, provide_token=True)
            job_ids.append(jid)

        time.sleep(0.1)
        jm.cancel_all(wait=False)
        for flag in flags:
            assert flag.wait(timeout=2)
        for jid in job_ids:
            jm.wait(jid, timeout=2)


# ---------------------------------------------------------------------------
# list_jobs
# ---------------------------------------------------------------------------
class TestListJobs:
    def test_shows_running_job(self, jm):
        started = threading.Event()
        blocker = threading.Event()

        def worker():
            started.set()
            blocker.wait(timeout=5)

        job_id = jm.start_job(worker)
        assert started.wait(timeout=2)
        jobs = jm.list_jobs()
        assert job_id in jobs
        assert jobs[job_id]["status"] == "running"
        blocker.set()
        jm.wait(job_id, timeout=2)

    def test_shows_completed_job(self, jm):
        job_id = jm.start_job(lambda: None)
        jm.wait(job_id, timeout=2)
        jm.cleanup_finished()
        jobs = jm.list_jobs()
        assert job_id in jobs
        assert jobs[job_id]["status"] == "completed"

    def test_shows_faulted_job(self, jm):
        def bad():
            raise RuntimeError("boom")

        job_id = jm.start_job(bad)
        jm.wait(job_id, timeout=2)
        jm.cleanup_finished()
        jobs = jm.list_jobs()
        assert job_id in jobs
        assert jobs[job_id]["status"] == "faulted"


# ---------------------------------------------------------------------------
# get_result
# ---------------------------------------------------------------------------
class TestGetResult:
    def test_none_for_successful_job(self, jm):
        job_id = jm.start_job(lambda: None)
        jm.wait(job_id, timeout=2)
        assert jm.get_result(job_id) is None

    def test_returns_exception_for_failed_job(self, jm):
        def bad():
            raise RuntimeError("test error")

        job_id = jm.start_job(bad)
        jm.wait(job_id, timeout=2)
        exc = jm.get_result(job_id)
        assert isinstance(exc, RuntimeError)
        assert str(exc) == "test error"

    def test_none_for_unknown_job(self, jm):
        assert jm.get_result("nonexistent") is None


# ---------------------------------------------------------------------------
# cleanup_finished
# ---------------------------------------------------------------------------
class TestCleanupFinished:
    def test_already_completed_jobs_return_zero(self, jm):
        """Jobs that complete successfully are automatically cleaned up."""
        ids = [jm.start_job(lambda: None) for _ in range(3)]
        for jid in ids:
            jm.wait(jid, timeout=2)
        count = jm.cleanup_finished()
        assert count == 0

    def test_completed_jobs_appear_in_completed_map(self, jm):
        job_id = jm.start_job(lambda: None)
        jm.wait(job_id, timeout=2)
        jobs = jm.list_jobs()
        assert job_id in jobs
        assert jobs[job_id]["status"] == "completed"

    def test_zero_when_nothing_to_clean(self, jm):
        assert jm.cleanup_finished() == 0

    def test_running_jobs_are_not_cleaned(self, jm):
        blocker = threading.Event()

        def worker():
            blocker.wait(timeout=5)

        job_id = jm.start_job(worker)
        time.sleep(0.1)
        count = jm.cleanup_finished()
        assert count == 0
        blocker.set()
        jm.wait(job_id, timeout=2)


# ---------------------------------------------------------------------------
# Concurrency limit
# ---------------------------------------------------------------------------
class TestConcurrencyLimit:
    def test_jobs_beyond_limit_block(self, logger):
        jm = JobManager(logger, max_concurrent=1)
        order = []
        gate1 = threading.Event()
        gate2 = threading.Event()

        def first():
            order.append("first_start")
            gate1.wait(timeout=5)
            order.append("first_end")

        def second():
            order.append("second_start")
            gate2.set()

        jid1 = jm.start_job(first)
        time.sleep(0.1)

        # second job should be blocked by the semaphore
        t = threading.Thread(target=lambda: jm.start_job(second))
        t.start()
        time.sleep(0.2)

        assert "second_start" not in order
        gate1.set()
        jm.wait(jid1, timeout=2)
        assert gate2.wait(timeout=2)
        assert "second_start" in order
        t.join(timeout=2)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------
class TestErrorHandling:
    def test_exception_captured_and_retrievable(self, jm):
        def bad():
            raise ValueError("captured")

        job_id = jm.start_job(bad)
        jm.wait(job_id, timeout=2)
        exc = jm.get_result(job_id)
        assert isinstance(exc, ValueError)
        assert str(exc) == "captured"

    def test_exception_shows_in_list_jobs(self, jm):
        def bad():
            raise RuntimeError("list check")

        job_id = jm.start_job(bad)
        jm.wait(job_id, timeout=2)
        jm.cleanup_finished()
        jobs = jm.list_jobs()
        assert jobs[job_id]["exception"] is not None
        assert "list check" in jobs[job_id]["exception"]


# ---------------------------------------------------------------------------
# Named jobs
# ---------------------------------------------------------------------------
class TestNamedJobs:
    def test_custom_name_appears_in_list(self, jm):
        started = threading.Event()
        blocker = threading.Event()

        def worker():
            started.set()
            blocker.wait(timeout=5)

        job_id = jm.start_job(worker, name="my-custom-job")
        assert started.wait(timeout=2)
        jobs = jm.list_jobs()
        assert jobs[job_id]["name"] == "my-custom-job"
        blocker.set()
        jm.wait(job_id, timeout=2)

    def test_default_name_generated(self, jm):
        job_id = jm.start_job(lambda: None)
        jm.wait(job_id, timeout=2)
        jm.cleanup_finished()
        jobs = jm.list_jobs()
        assert jobs[job_id]["name"].startswith("ManagedJob-")


# ---------------------------------------------------------------------------
# External cancellation_token
# ---------------------------------------------------------------------------
class TestExternalCancellationToken:
    def test_external_token_cancels_job(self, jm):
        cts = CancellationTokenSource()
        cancelled_flag = threading.Event()

        def worker(token):
            while not token.is_cancellation_requested:
                time.sleep(0.01)
            cancelled_flag.set()

        job_id = jm.start_job(
            worker,
            provide_token=True,
            cancellation_token=cts.token,
        )
        time.sleep(0.1)
        cts.cancel()
        assert cancelled_flag.wait(timeout=2)
        jm.wait(job_id, timeout=2)
        cts.dispose()

    def test_already_cancelled_external_token(self, jm):
        cts = CancellationTokenSource()
        cts.cancel()
        cancelled_flag = threading.Event()

        def worker(token):
            if token.is_cancellation_requested:
                cancelled_flag.set()
            else:
                cancelled_flag.set()

        job_id = jm.start_job(
            worker,
            provide_token=True,
            cancellation_token=cts.token,
        )
        assert cancelled_flag.wait(timeout=2)
        jm.wait(job_id, timeout=2)
        cts.dispose()
