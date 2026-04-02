from application_builder import (
    Worker, TimedWorker, ILogger, IConfiguration,
    IHostApplicationLifetime,
)


class StartupListener(Worker):
    """Registers callbacks on application lifetime events and logs them."""

    def __init__(self, lifetime: IHostApplicationLifetime,
                 logger: ILogger):
        super().__init__()
        self._lifetime = lifetime
        self._logger = logger

    def execute(self) -> None:
        self._lifetime.application_started.register(
            lambda: self._logger.success("[Lifecycle] Application STARTED event fired")
        )
        self._lifetime.application_stopping.register(
            lambda: self._logger.warning("[Lifecycle] Application STOPPING event fired")
        )
        self._lifetime.application_stopped.register(
            lambda: self._logger.info("[Lifecycle] Application STOPPED event fired")
        )

        self._logger.info("[Lifecycle] Lifetime callbacks registered. Waiting...")

        # Keep alive until told to stop
        while not self.is_stopping():
            self.wait_for_stop(1.0)


class HeartbeatWorker(TimedWorker):
    """Emits a heartbeat log every interval."""

    def __init__(self, config: IConfiguration, logger: ILogger):
        interval = config.get_float("Heartbeat:IntervalSeconds", 3.0)
        super().__init__(interval_seconds=interval)
        self._logger = logger
        self._count = 0

    def do_work(self) -> None:
        self._count += 1
        self._logger.info(f"[Heartbeat] tick #{self._count}")
