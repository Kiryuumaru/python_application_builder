from application_builder import Worker, ILogger, IHostEnvironment


class EnvironmentReporterWorker(Worker):
    """Reports environment details and adjusts behavior based on environment."""

    def __init__(self, env: IHostEnvironment, logger: ILogger):
        super().__init__()
        self._env = env
        self._logger = logger

    def execute(self) -> None:
        self._logger.info("===== Host Environment =====")
        self._logger.info(f"  Environment : {self._env.environment_name}")
        self._logger.info(f"  Application : {self._env.application_name}")
        self._logger.info(f"  Content Root: {self._env.content_root_path}")
        self._logger.info(f"  Is Dev?     : {self._env.is_development()}")
        self._logger.info(f"  Is Staging? : {self._env.is_staging()}")
        self._logger.info(f"  Is Prod?    : {self._env.is_production()}")
        self._logger.info("============================")

        if self._env.is_development():
            self._logger.warning("[Env] Running in DEVELOPMENT mode — verbose logging enabled")
        elif self._env.is_production():
            self._logger.info("[Env] Running in PRODUCTION mode — optimized settings")
        else:
            self._logger.info(f"[Env] Running in {self._env.environment_name} mode")

        self._logger.success("[Env] Environment check complete")
