import threading
import time

import pytest

from application_builder import (
    CancellationToken,
    CancellationTokenSource,
    CancellationTokenRegistration,
    OperationCanceledException,
    _NeverCancelledToken,
    create_linked_token,
    with_timeout,
)


# ---------------------------------------------------------------------------
# CancellationToken
# ---------------------------------------------------------------------------
class TestCancellationToken:
    def test_initial_state_not_cancelled(self):
        token = CancellationToken()
        assert token.is_cancellation_requested is False

    def test_initial_state_cancelled(self):
        token = CancellationToken(cancelled=True)
        assert token.is_cancellation_requested is True

    def test_can_be_cancelled_returns_true(self):
        token = CancellationToken()
        assert token.can_be_cancelled is True

    def test_throw_if_cancellation_requested_no_throw(self):
        token = CancellationToken()
        token.throw_if_cancellation_requested()

    def test_throw_if_cancellation_requested_raises(self):
        token = CancellationToken(cancelled=True)
        with pytest.raises(OperationCanceledException) as exc_info:
            token.throw_if_cancellation_requested()
        assert exc_info.value.token is token

    def test_register_callback_fires_on_cancel(self):
        token = CancellationToken()
        called = threading.Event()
        token.register(called.set)
        assert not called.is_set()
        token._cancel()
        assert called.is_set()

    def test_register_callback_on_already_cancelled_fires_immediately(self):
        token = CancellationToken(cancelled=True)
        called = threading.Event()
        token.register(called.set)
        assert called.is_set()

    def test_cancel_is_idempotent(self):
        token = CancellationToken()
        counter = {"n": 0}

        def increment():
            counter["n"] += 1

        token.register(increment)
        token._cancel()
        token._cancel()
        assert counter["n"] == 1

    def test_callbacks_cleared_after_cancel(self):
        token = CancellationToken()
        token.register(lambda: None)
        token._cancel()
        assert len(token._callbacks) == 0

    def test_cancel_with_exception(self):
        token = CancellationToken()
        err = RuntimeError("boom")
        token._cancel(exception=err)
        assert token._exception is err

    def test_multiple_callbacks_all_fire(self):
        token = CancellationToken()
        results = []
        token.register(lambda: results.append("a"))
        token.register(lambda: results.append("b"))
        token.register(lambda: results.append("c"))
        token._cancel()
        assert results == ["a", "b", "c"]

    def test_callback_exception_does_not_prevent_others(self):
        token = CancellationToken()
        called = threading.Event()

        def bad():
            raise ValueError("oops")

        token.register(bad)
        token.register(called.set)
        token._cancel()
        assert called.is_set()


# ---------------------------------------------------------------------------
# _NeverCancelledToken
# ---------------------------------------------------------------------------
class TestNeverCancelledToken:
    def test_can_be_cancelled_is_false(self):
        token = _NeverCancelledToken()
        assert token.can_be_cancelled is False

    def test_cancel_is_noop(self):
        token = _NeverCancelledToken()
        token._cancel()
        assert token.is_cancellation_requested is False

    def test_none_factory_returns_never_cancelled(self):
        token = CancellationToken.none()
        assert isinstance(token, _NeverCancelledToken)
        assert token.can_be_cancelled is False
        assert token.is_cancellation_requested is False


# ---------------------------------------------------------------------------
# OperationCanceledException
# ---------------------------------------------------------------------------
class TestOperationCanceledException:
    def test_default_message(self):
        exc = OperationCanceledException()
        assert "cancelled" in str(exc).lower()

    def test_custom_message(self):
        exc = OperationCanceledException("custom msg")
        assert str(exc) == "custom msg"

    def test_token_attribute(self):
        token = CancellationToken()
        exc = OperationCanceledException(token=token)
        assert exc.token is token

    def test_token_default_none(self):
        exc = OperationCanceledException()
        assert exc.token is None

    def test_is_exception(self):
        assert issubclass(OperationCanceledException, Exception)


# ---------------------------------------------------------------------------
# CancellationTokenRegistration
# ---------------------------------------------------------------------------
class TestCancellationTokenRegistration:
    def test_dispose_unregisters_callback(self):
        token = CancellationToken()
        called = threading.Event()
        reg = token.register(called.set)
        reg.dispose()
        token._cancel()
        assert not called.is_set()

    def test_double_dispose_is_safe(self):
        token = CancellationToken()
        reg = token.register(lambda: None)
        reg.dispose()
        reg.dispose()

    def test_context_manager(self):
        token = CancellationToken()
        called = threading.Event()
        with token.register(called.set) as reg:
            assert isinstance(reg, CancellationTokenRegistration)
        token._cancel()
        assert not called.is_set()

    def test_registration_from_already_cancelled_token_dispose_safe(self):
        token = CancellationToken(cancelled=True)
        reg = token.register(lambda: None)
        reg.dispose()
        reg.dispose()


# ---------------------------------------------------------------------------
# CancellationTokenSource
# ---------------------------------------------------------------------------
class TestCancellationTokenSource:
    def test_token_property(self):
        cts = CancellationTokenSource()
        assert isinstance(cts.token, CancellationToken)

    def test_initial_not_cancelled(self):
        cts = CancellationTokenSource()
        assert cts.is_cancellation_requested is False
        assert cts.token.is_cancellation_requested is False

    def test_cancel(self):
        cts = CancellationTokenSource()
        cts.cancel()
        assert cts.is_cancellation_requested is True
        assert cts.token.is_cancellation_requested is True

    def test_cancel_fires_callback(self):
        cts = CancellationTokenSource()
        called = threading.Event()
        cts.token.register(called.set)
        cts.cancel()
        assert called.is_set()

    def test_cancel_after_with_delay(self):
        cts = CancellationTokenSource()
        cts.cancel_after(0.1)
        assert cts.is_cancellation_requested is False
        time.sleep(0.3)
        assert cts.is_cancellation_requested is True

    def test_cancel_after_zero_cancels_immediately(self):
        cts = CancellationTokenSource()
        cts.cancel_after(0)
        assert cts.is_cancellation_requested is True

    def test_cancel_after_negative_raises(self):
        cts = CancellationTokenSource()
        with pytest.raises(ValueError):
            cts.cancel_after(-1)

    def test_dispose_prevents_token_access(self):
        cts = CancellationTokenSource()
        cts.dispose()
        with pytest.raises(RuntimeError):
            _ = cts.token

    def test_cancel_after_dispose_is_noop(self):
        cts = CancellationTokenSource()
        token = cts.token
        cts.dispose()
        cts.cancel()
        assert token.is_cancellation_requested is False

    def test_context_manager(self):
        with CancellationTokenSource() as cts:
            token = cts.token
            assert isinstance(cts, CancellationTokenSource)
        with pytest.raises(RuntimeError):
            _ = cts.token

    def test_cancel_after_replaces_previous_timer(self):
        cts = CancellationTokenSource()
        cts.cancel_after(5.0)
        cts.cancel_after(0.1)
        time.sleep(0.3)
        assert cts.is_cancellation_requested is True

    def test_double_dispose_is_safe(self):
        cts = CancellationTokenSource()
        cts.dispose()
        cts.dispose()


# ---------------------------------------------------------------------------
# create_linked_token
# ---------------------------------------------------------------------------
class TestCreateLinkedToken:
    def test_cancelling_one_source_cancels_linked(self):
        cts1 = CancellationTokenSource()
        cts2 = CancellationTokenSource()
        linked = create_linked_token(cts1.token, cts2.token)
        assert linked.token.is_cancellation_requested is False
        cts1.cancel()
        assert linked.token.is_cancellation_requested is True

    def test_cancelling_other_source_cancels_linked(self):
        cts1 = CancellationTokenSource()
        cts2 = CancellationTokenSource()
        linked = create_linked_token(cts1.token, cts2.token)
        cts2.cancel()
        assert linked.token.is_cancellation_requested is True

    def test_already_cancelled_token_triggers_immediately(self):
        cts1 = CancellationTokenSource()
        cts1.cancel()
        cts2 = CancellationTokenSource()
        linked = create_linked_token(cts1.token, cts2.token)
        assert linked.token.is_cancellation_requested is True

    def test_linked_single_token(self):
        cts = CancellationTokenSource()
        linked = create_linked_token(cts.token)
        cts.cancel()
        assert linked.token.is_cancellation_requested is True


# ---------------------------------------------------------------------------
# with_timeout
# ---------------------------------------------------------------------------
class TestWithTimeout:
    def test_cancels_after_delay(self):
        cts = with_timeout(0.1)
        assert cts.is_cancellation_requested is False
        time.sleep(0.3)
        assert cts.is_cancellation_requested is True

    def test_zero_timeout_cancels_immediately(self):
        cts = with_timeout(0)
        assert cts.is_cancellation_requested is True

    def test_returns_cancellation_token_source(self):
        cts = with_timeout(1.0)
        assert isinstance(cts, CancellationTokenSource)
        cts.dispose()


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------
class TestThreadSafety:
    def test_concurrent_register_and_cancel(self):
        token = CancellationToken()
        call_count = {"n": 0}
        lock = threading.Lock()
        barrier = threading.Barrier(11)

        def register_many():
            barrier.wait()
            for _ in range(100):
                def cb():
                    with lock:
                        call_count["n"] += 1
                token.register(cb)

        threads = [threading.Thread(target=register_many) for _ in range(10)]
        for t in threads:
            t.start()

        barrier.wait()
        time.sleep(0.05)
        token._cancel()

        for t in threads:
            t.join(timeout=2.0)

        assert call_count["n"] > 0

    def test_concurrent_cancel_calls(self):
        token = CancellationToken()
        counter = {"n": 0}
        lock = threading.Lock()

        def increment():
            with lock:
                counter["n"] += 1

        token.register(increment)
        barrier = threading.Barrier(10)

        def cancel_it():
            barrier.wait()
            token._cancel()

        threads = [threading.Thread(target=cancel_it) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2.0)

        assert counter["n"] == 1
