from abc import ABC, abstractmethod


class IRepository(ABC):
    """Data access repository."""

    @abstractmethod
    def query(self, sql: str) -> str:
        ...
