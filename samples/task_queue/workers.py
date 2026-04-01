from application_builder import (
    TimedWorker, ILogger, IConfiguration,
    JobManager, with_timeout,
)
from interfaces import ITaskRepository, ITaskProcessor


class TaskDispatcherWorker(TimedWorker):
    """Dequeues tasks and dispatches them via JobManager with per-task timeouts."""

    def __init__(self, repo: ITaskRepository,
                 processor: ITaskProcessor,
                 job_manager: JobManager,
                 config: IConfiguration,
                 logger: ILogger):
        interval = config.get_int("TaskQueue:DispatchIntervalSeconds", 2)
        super().__init__(interval_seconds=interval)
        self._repo = repo
        self._processor = processor
        self._job_manager = job_manager
        self._timeout = config.get_float("TaskQueue:TaskTimeoutSeconds", 5.0)
        self._logger = logger

    def do_work(self) -> None:
        stats = self._repo.get_stats()
        self._logger.info(
            f"Queue — pending: {stats['pending']}, "
            f"completed: {stats['completed']}, failed: {stats['failed']}"
        )

        task = self._repo.dequeue()
        if task is None:
            self._logger.debug("No tasks in queue")
            return

        task_id = task["id"]
        timeout_source = with_timeout(self._timeout)

        def run_task(token, t=task, ts=timeout_source):
            try:
                success = self._processor.process(t, token)
                if success:
                    self._repo.mark_complete(t["id"])
                else:
                    self._repo.mark_failed(t["id"], "cancelled")
            except Exception as e:
                self._repo.mark_failed(t["id"], str(e))
                self._logger.error(f"Task #{t['id']} failed: {e}")
            finally:
                ts.dispose()

        self._job_manager.start_job(
            run_task,
            name=f"Task-{task_id}",
            provide_token=True,
            cancellation_token=timeout_source.token,
            daemon=True,
        )
