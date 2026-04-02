from abc import ABC, abstractmethod


class IEmailService(ABC):
    """Send email messages."""

    @abstractmethod
    def send(self, to: str, body: str) -> None:
        ...


class ICache(ABC):
    """Caching service."""

    @abstractmethod
    def get(self, key: str) -> str:
        ...
