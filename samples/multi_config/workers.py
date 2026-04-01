from application_builder import TimedWorker, ILogger, IConfiguration
from interfaces import IConfigReporter, IFeatureFlags


class ConfigMonitorWorker(TimedWorker):
    """Periodically reloads config and reports changes."""

    def __init__(self, reporter: IConfigReporter,
                 flags: IFeatureFlags,
                 config: IConfiguration,
                 logger: ILogger):
        interval = config.get_float("Monitoring:IntervalSeconds", 3.0)
        super().__init__(interval_seconds=interval)
        self._reporter = reporter
        self._flags = flags
        self._config = config
        self._logger = logger

    def do_work(self) -> None:
        # Reload configuration from all providers — showcases reload()
        self._config.reload()

        report = self._reporter.report()
        self._logger.info(report)

        flags = self._flags.all_flags()
        enabled = [k for k, v in flags.items() if v]
        disabled = [k for k, v in flags.items() if not v]
        self._logger.info(f"[Flags] Enabled: {enabled} | Disabled: {disabled}")
