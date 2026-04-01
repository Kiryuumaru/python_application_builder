"""
Event Bus — publish/subscribe event system with scoped handlers.

Showcases: add_scoped_factory, add_transient_factory, ScopeFactory
"""
from abc import ABC, abstractmethod
from typing import List, Any, Dict


class IEvent(ABC):
    """Base event interface."""

    @abstractmethod
    def get_type(self) -> str:
        """Get the event type identifier."""

    @abstractmethod
    def get_payload(self) -> Dict[str, Any]:
        """Get the event payload."""


class IEventHandler(ABC):
    """Handles events of a specific type."""

    @abstractmethod
    def handle(self, event: IEvent) -> None:
        """Process an event."""


class IEventBus(ABC):
    """Publishes events to registered handlers."""

    @abstractmethod
    def publish(self, event: IEvent) -> None:
        """Publish an event to all matching handlers."""


class IEventLog(ABC):
    """Scoped audit log tracking events within a processing context."""

    @abstractmethod
    def append(self, event_type: str, detail: str) -> None:
        """Append a log entry."""

    @abstractmethod
    def entries(self) -> List[str]:
        """Get all log entries for this scope."""


class IEventMetrics(ABC):
    """Thread-safe global event counter."""

    @abstractmethod
    def increment(self, event_type: str) -> None:
        """Increment the count for an event type."""

    @abstractmethod
    def snapshot(self) -> Dict[str, int]:
        """Get a snapshot of all event counts."""

    @abstractmethod
    def total(self) -> int:
        """Get total number of events across all types."""
