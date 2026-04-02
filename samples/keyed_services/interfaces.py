from abc import ABC, abstractmethod


class INotificationSender(ABC):
    """Send a notification through a specific channel."""

    @abstractmethod
    def send(self, message: str) -> str:
        ...
