from application_builder import Worker, ILogger
from interfaces import IEmailService, ICache


class AppWorker(Worker):
    def __init__(self, logger: ILogger, email: IEmailService, cache: ICache) -> None:
        super().__init__()
        self._logger = logger
        self._email = email
        self._cache = cache

    def execute(self) -> None:
        self._logger.info("=== Validated App Demo ===")

        cached = self._cache.get("user:42")
        self._logger.info(f"Got from cache: {cached}")

        self._email.send("admin@example.com", "Validation passed!")

        self._logger.info("Demo complete.")
