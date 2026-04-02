from application_builder import Worker, ILogger
from interfaces import IRepository


class QueryWorker(Worker):
    def __init__(self, logger: ILogger, repo: IRepository) -> None:
        super().__init__()
        self._logger = logger
        self._repo = repo

    def execute(self) -> None:
        self._logger.info("=== Decorated Services Demo ===")

        # First call — cache miss, logged
        self._repo.query("SELECT * FROM users")

        # Second call same query — cache hit
        self._repo.query("SELECT * FROM users")

        # Different query — cache miss
        self._repo.query("SELECT * FROM orders")

        self._logger.info("Demo complete.")
