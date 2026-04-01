from application_builder import (
    Worker, TimedWorker, ILogger, IConfiguration,
    CliRunnerService, CancellationTokenSource,
    create_linked_token, OperationCanceledException,
)
from interfaces import IJobRegistry


class JobRunnerWorker(Worker):
    """Executes all registered jobs sequentially with timeout cancellation."""

    def __init__(self, registry: IJobRegistry,
                 cli: CliRunnerService,
                 logger: ILogger):
        super().__init__()
        self._registry = registry
        self._cli = cli
        self._logger = logger

    def execute(self) -> None:
        jobs = self._registry.get_jobs()
        self._logger.info(f"[Runner] Starting {len(jobs)} jobs")

        # Source that cancels when the worker is stopped
        app_source = CancellationTokenSource()

        for job in jobs:
            if self.is_stopping():
                app_source.cancel()
                break

            name = job.get_name()
            command = job.get_command()
            timeout = job.get_timeout_seconds()
            self._logger.info(f"[Runner] Starting '{name}': {' '.join(command)}")

            # Per-job timeout source — showcases cancel_after
            job_source = CancellationTokenSource()
            if timeout is not None:
                job_source.cancel_after(timeout)

            # Link job timeout with app shutdown — showcases create_linked_token
            linked = create_linked_token(app_source.token, job_source.token)

            try:
                self._cli.run(
                    command,
                    name=name,
                    cwd=job.get_cwd(),
                    cancellation_token=linked.token,
                )
                self._registry.record_result(name, True, "completed")
                self._logger.info(f"[Runner] '{name}' completed successfully")
            except OperationCanceledException:
                self._registry.record_result(name, False, "timed out or cancelled")
                self._logger.warning(f"[Runner] '{name}' was cancelled/timed out")
            except RuntimeError as e:
                self._registry.record_result(name, False, str(e))
                self._logger.error(f"[Runner] '{name}' failed: {e}")

        self._logger.info("[Runner] All jobs processed")


class ResultReporterWorker(TimedWorker):
    """Periodically reports job results."""

    def __init__(self, registry: IJobRegistry,
                 config: IConfiguration,
                 logger: ILogger):
        interval = config.get_float("Scheduler:ReportIntervalSeconds", 8.0)
        super().__init__(interval_seconds=interval)
        self._registry = registry
        self._logger = logger

    def do_work(self) -> None:
        results = self._registry.get_results()
        if not results:
            self._logger.info("[Reporter] No results yet")
            return

        self._logger.info("===== Job Results =====")
        for name, info in results.items():
            status = "PASS" if info["success"] else "FAIL"
            self._logger.info(f"  {name:20s}  {status:4s}  {info['detail']}")
        self._logger.info("=======================")
