import time
from application_builder import ILogger
from interfaces import IRepository


class SqlRepository(IRepository):
    def query(self, sql: str) -> str:
        return f"Result of [{sql}]"


class LoggingRepositoryDecorator(IRepository):
    """Decorator that logs every query."""

    def __init__(self, inner: IRepository, logger: ILogger) -> None:
        self._inner = inner
        self._logger = logger

    def query(self, sql: str) -> str:
        self._logger.info(f"[LOG] Executing query: {sql}")
        result = self._inner.query(sql)
        self._logger.info(f"[LOG] Query returned: {result}")
        return result


class CachingRepositoryDecorator(IRepository):
    """Decorator that caches query results."""

    def __init__(self, inner: IRepository, logger: ILogger) -> None:
        self._inner = inner
        self._logger = logger
        self._cache: dict = {}

    def query(self, sql: str) -> str:
        if sql in self._cache:
            self._logger.info(f"[CACHE HIT] {sql}")
            return self._cache[sql]
        self._logger.info(f"[CACHE MISS] {sql}")
        result = self._inner.query(sql)
        self._cache[sql] = result
        return result
