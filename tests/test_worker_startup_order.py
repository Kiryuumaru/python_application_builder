import threading
import time

from application_builder import (
    ApplicationBuilder,
    ILogger,
    IHostApplicationLifetime,
    HostApplicationLifetime,
    Worker,
)


class _OrderTrackingWorker(Worker):
    """Worker that records when execute() begins relative to startup."""

    def __init__(self, logger: ILogger, lifetime: IHostApplicationLifetime):
        super().__init__()
        self.logger = logger
        self.lifetime = lifetime
        self.app_was_started_before_execute = None
        self._execute_called = threading.Event()

    def execute(self):
        self.app_was_started_before_execute = self.lifetime.application_started.is_cancellation_requested
        self._execute_called.set()
        self.wait_for_stop(timeout_seconds=0.1)


def _find_tracking_worker(provider):
    """Find the _OrderTrackingWorker instance from the hosted service manager."""
    for svc, _ in provider._hosted_service_manager._services:
        if isinstance(svc, _OrderTrackingWorker):
            return svc
    return None


class TestWorkerStartupOrder:
    """Workers must not execute before notify_started() is called."""

    def test_worker_executes_after_application_started(self):
        app = ApplicationBuilder()
        app.add_worker(_OrderTrackingWorker)

        provider = app.build(auto_start_hosted_services=False)

        lifetime = provider.get_service(HostApplicationLifetime)
        lifetime.notify_started()

        provider.start_hosted_services()

        worker = _find_tracking_worker(provider)
        assert worker is not None
        worker._execute_called.wait(timeout=5)
        assert worker.app_was_started_before_execute is True

        provider.stop_hosted_services()

    def test_worker_sees_started_false_when_started_before_notify(self):
        app = ApplicationBuilder()
        app.add_worker(_OrderTrackingWorker)

        provider = app.build(auto_start_hosted_services=True)

        worker = _find_tracking_worker(provider)
        assert worker is not None
        worker._execute_called.wait(timeout=5)
        assert worker.app_was_started_before_execute is False

        provider.stop_hosted_services()
