import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from application_builder import ApplicationBuilder
from interfaces import IHealthRegistry
from services import InMemoryHealthRegistry
from workers import HealthCheckWorker, DashboardWorker

# Pre-build the registry with endpoints loaded from config
registry = InMemoryHealthRegistry()
for ep in ["api.example.com/health", "db.internal:5432", "cache.internal:6379",
           "queue.internal:5672", "storage.blob.core"]:
    registry.add_endpoint(ep)

app = ApplicationBuilder()

app.add_configuration_dictionary({
    "Health:CheckIntervalSeconds": "2",
    "Health:DashboardIntervalSeconds": "5",
    "Health:LatencyThresholdMs": "500",
})

# Register pre-built instance — showcases add_singleton_instance
app.add_singleton_instance(IHealthRegistry, registry)

app.add_worker(HealthCheckWorker)
app.add_worker(DashboardWorker)

app.run()
