import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from application_builder import ApplicationBuilder, ILogger, ServiceProvider
from interfaces import IEventHandler, IEventBus, IEventLog, IEventMetrics
from services import (LoggingEventHandler, MetricsEventHandler,
                      InMemoryEventBus, ScopedEventLog, EventMetricsCounter)
from workers import EventProducerWorker, EventSummaryWorker

app = ApplicationBuilder()

app.add_configuration_dictionary({
    "Events:ProduceIntervalSeconds": "2",
    "Events:SummaryIntervalSeconds": "6",
})

# Singleton metrics counter shared across all scopes
app.add_singleton(IEventMetrics, EventMetricsCounter)

# Scoped event log — each scope gets its own audit trail
app.add_scoped_factory(IEventLog, lambda sp: ScopedEventLog())

# Transient handlers — new instance per resolution, showcases add_transient_factory
app.add_transient_factory(IEventHandler, lambda sp: LoggingEventHandler(
    sp.get_required_service(IEventLog),
    sp.get_required_service(IEventMetrics),
    sp.get_required_service(ILogger),
))
app.add_transient_factory(IEventHandler, lambda sp: MetricsEventHandler(
    sp.get_required_service(IEventLog),
    sp.get_required_service(IEventMetrics),
    sp.get_required_service(ILogger),
))

# Scoped bus — per-scope bus resolves transient handlers
app.add_scoped_factory(IEventBus, lambda sp: InMemoryEventBus(
    sp.get_services(IEventHandler),
    sp.get_required_service(ILogger),
))

app.add_worker(EventProducerWorker)
app.add_worker(EventSummaryWorker)

app.run()
