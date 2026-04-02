from abc import ABC, abstractmethod
from typing import List


class IConnection(ABC):
    """A database-like connection that should be disposed."""

    @abstractmethod
    def execute(self, query: str) -> str:
        """Execute a query."""

    @abstractmethod
    def is_open(self) -> bool:
        """Check if the connection is still open."""
