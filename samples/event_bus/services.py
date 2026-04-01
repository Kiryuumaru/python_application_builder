import threading
import uuid
from typing import List, Any, Dict

from interfaces import IEvent, IEventHandler, IEventBus, IEventLog, IEventMetrics
from application_builder import ILogger


class SimpleEvent(IEvent):
    """A basic event carrying a type and payload."""

    def __init__(self, event_type: str, payload: Dict[str, Any]) -> None:
        self._type = event_type
        self._payload = payload

    def get_type(self) -> str:
        return self._type

    def get_payload(self) -> Dict[str, Any]:
        return self._payload


class LoggingEventHandler(IEventHandler):
    """Logs every event it receives."""

    def __init__(self, event_log: IEventLog, metrics: IEventMetrics,
                 logger: ILogger):
        self._event_log = event_log
        self._metrics = metrics
        self._logger = logger

    def handle(self, event: IEvent) -> None:
        detail = f"{event.get_type()}: {event.get_payload()}"
        self._event_log.append(event.get_type(), detail)
        self._metrics.increment(event.get_type())
        self._logger.info(f"[LogHandler] {detail}")


class MetricsEventHandler(IEventHandler):
    """Tracks events in the scoped log."""

    def __init__(self, event_log: IEventLog, metrics: IEventMetrics,
                 logger: ILogger):
        self._event_log = event_log
        self._metrics = metrics
        self._logger = logger

    def handle(self, event: IEvent) -> None:
        self._event_log.append(event.get_type(), "counted")
        self._logger.debug(f"[MetricsHandler] {event.get_type()}")


class InMemoryEventBus(IEventBus):
    """Simple event bus dispatching to all registered handlers."""

    def __init__(self, handlers: List[IEventHandler], logger: ILogger):
        self._handlers = handlers
        self._logger = logger

    def publish(self, event: IEvent) -> None:
        for handler in self._handlers:
            try:
                handler.handle(event)
            except Exception as e:
                self._logger.error(f"[EventBus] Handler failed: {e}")


class ScopedEventLog(IEventLog):
    """Per-scope event audit log."""

    def __init__(self) -> None:
        self._id = str(uuid.uuid4())[:8]
        self._entries: List[str] = []

    def append(self, event_type: str, detail: str) -> None:
        self._entries.append(f"[{self._id}] {detail}")

    def entries(self) -> List[str]:
        return list(self._entries)


class EventMetricsCounter(IEventMetrics):
    """Thread-safe singleton event counter."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: Dict[str, int] = {}

    def increment(self, event_type: str) -> None:
        with self._lock:
            self._counts[event_type] = self._counts.get(event_type, 0) + 1

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._counts)

    def total(self) -> int:
        with self._lock:
            return sum(self._counts.values())
