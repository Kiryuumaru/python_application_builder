import uuid

from application_builder import ILogger, IDisposable
from interfaces import IConnection


class ScopedConnection(IConnection, IDisposable):
    """Simulated database connection that tracks open/close state."""

    def __init__(self, logger: ILogger):
        self._id = str(uuid.uuid4())[:8]
        self._open = True
        self._logger = logger
        self._logger.info(f"[Connection {self._id}] OPENED")

    def execute(self, query: str) -> str:
        if not self._open:
            raise RuntimeError(f"Connection {self._id} is closed")
        self._logger.debug(f"[Connection {self._id}] Executing: {query}")
        return f"result-{self._id}"

    def is_open(self) -> bool:
        return self._open

    def dispose(self) -> None:
        if self._open:
            self._open = False
            self._logger.info(f"[Connection {self._id}] DISPOSED (closed)")
