from application_builder import Worker, ILogger, IConfiguration


class ConfigReporterWorker(Worker):
    def __init__(self, logger: ILogger, config: IConfiguration) -> None:
        super().__init__()
        self._logger = logger
        self._config = config

    def execute(self) -> None:
        self._logger.info("=== CLI Args Configuration Demo ===")

        server_host = self._config.get("Server:Host") or "(not set)"
        server_port = self._config.get("Server:Port") or "(not set)"
        log_level = self._config.get("Logging:Level") or "(not set)"
        app_name = self._config.get("App:Name") or "(not set)"

        self._logger.info(f"Server Host: {server_host}")
        self._logger.info(f"Server Port: {server_port}")
        self._logger.info(f"Logging Level: {log_level}")
        self._logger.info(f"App Name: {app_name}")
        self._logger.info("Demo complete.")
