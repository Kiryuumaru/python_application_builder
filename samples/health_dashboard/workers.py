import random
import time

from application_builder import TimedWorker, ILogger, IConfiguration
from interfaces import IHealthRegistry


class HealthCheckWorker(TimedWorker):
    """Periodically checks all registered endpoints."""

    def __init__(self, registry: IHealthRegistry,
                 config: IConfiguration,
                 logger: ILogger):
        interval = config.get_float("Health:CheckIntervalSeconds", 2.0)
        super().__init__(interval_seconds=interval)
        self._registry = registry
        self._logger = logger
        self._threshold_ms = config.get_float("Health:LatencyThresholdMs", 500.0)

    def do_work(self) -> None:
        endpoints = self._registry.get_endpoints()
        for ep in endpoints:
            latency = random.uniform(50.0, 800.0)
            healthy = latency < self._threshold_ms
            self._registry.record(ep, healthy, round(latency, 1))
            status = "OK" if healthy else "DEGRADED"
            self._logger.debug(f"[Check] {ep} -> {status} ({latency:.0f}ms)")


class DashboardWorker(TimedWorker):
    """Periodically prints a summary dashboard."""

    def __init__(self, registry: IHealthRegistry,
                 config: IConfiguration,
                 logger: ILogger):
        interval = config.get_float("Health:DashboardIntervalSeconds", 5.0)
        super().__init__(interval_seconds=interval)
        self._registry = registry
        self._logger = logger

    def do_work(self) -> None:
        status = self._registry.get_status()
        if not status:
            return

        self._logger.info("===== Health Dashboard =====")
        for ep, info in status.items():
            state = "OK" if info.get("healthy") else "DOWN"
            latency = info.get("latency_ms", 0)
            failures = info.get("consecutive_failures", 0)
            self._logger.info(
                f"  {ep:30s}  {state:6s}  {latency:6.1f}ms  failures={failures}"
            )
        self._logger.info("============================")
