import random

from application_builder import TimedWorker, ILogger, IConfiguration, ScopeFactory
from interfaces import IEventBus, IEventLog, IEventMetrics
from services import SimpleEvent


class EventProducerWorker(TimedWorker):
    """Produces random events within scoped contexts."""

    EVENT_TYPES = ["user.login", "order.placed", "payment.processed",
                   "item.shipped", "user.logout"]

    def __init__(self, scope_factory: ScopeFactory,
                 config: IConfiguration,
                 logger: ILogger):
        interval = config.get_float("Events:ProduceIntervalSeconds", 2.0)
        super().__init__(interval_seconds=interval)
        self._scope_factory = scope_factory
        self._logger = logger

    def do_work(self) -> None:
        # Each batch uses its own scope — scoped IEventLog is unique per context
        with self._scope_factory.create_scope_context() as scope:
            bus = scope.get_required_service(IEventBus)
            event_log = scope.get_required_service(IEventLog)

            event_type = random.choice(self.EVENT_TYPES)
            payload = {"user_id": random.randint(1, 100),
                       "value": round(random.uniform(10, 500), 2)}

            bus.publish(SimpleEvent(event_type, payload))

            entries = event_log.entries()
            self._logger.info(
                f"[Producer] Scope log has {len(entries)} entries after publish"
            )


class EventSummaryWorker(TimedWorker):
    """Periodically reports event bus statistics."""

    def __init__(self, metrics: IEventMetrics,
                 config: IConfiguration,
                 logger: ILogger):
        interval = config.get_float("Events:SummaryIntervalSeconds", 6.0)
        super().__init__(interval_seconds=interval)
        self._metrics = metrics
        self._logger = logger

    def do_work(self) -> None:
        snap = self._metrics.snapshot()
        total = self._metrics.total()
        top_events = sorted(snap.items(), key=lambda x: x[1], reverse=True)[:3]
        top_str = ", ".join(f"{k}={v}" for k, v in top_events)
        self._logger.info(f"[Summary] Total events: {total} | Top: {top_str}")
