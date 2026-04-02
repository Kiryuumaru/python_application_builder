from application_builder import TimedWorker, ILogger, IConfiguration, ScopeFactory
from interfaces import IConnection


class DatabaseWorker(TimedWorker):
    """Runs scoped database operations — connection is disposed when scope exits."""

    QUERIES = [
        "SELECT * FROM users",
        "INSERT INTO orders (item) VALUES ('widget')",
        "UPDATE inventory SET qty = qty - 1 WHERE item = 'widget'",
        "DELETE FROM logs WHERE age > 30",
        "SELECT COUNT(*) FROM orders",
    ]

    def __init__(self, scope_factory: ScopeFactory,
                 config: IConfiguration,
                 logger: ILogger):
        interval = config.get_float("Database:IntervalSeconds", 2.0)
        super().__init__(interval_seconds=interval)
        self._scope_factory = scope_factory
        self._logger = logger
        self._index = 0

    def do_work(self) -> None:
        if self._index >= len(self.QUERIES):
            self._logger.info("[Worker] All queries processed")
            return

        query = self.QUERIES[self._index]
        self._index += 1

        # Connection is created per scope and disposed when scope exits
        with self._scope_factory.create_scope_context() as scope:
            conn = scope.get_required_service(IConnection)
            result = conn.execute(query)
            self._logger.info(f"[Worker] Query result: {result}")
            self._logger.info(f"[Worker] Connection open? {conn.is_open()}")

        # After scope exit, connection should be disposed
        self._logger.info(f"[Worker] After scope — connection open? {conn.is_open()}")
