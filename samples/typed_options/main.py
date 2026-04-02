import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from application_builder import ApplicationBuilder
from options import DatabaseOptions, CacheOptions
from workers import OptionsReporterWorker

app = ApplicationBuilder()

app.add_configuration_dictionary({
    "Database": {
        "host": "db.production.internal",
        "port": "5433",
        "name": "orders_db",
        "max_connections": "20",
    },
    "Cache": {
        "enabled": "true",
        "ttl_seconds": "600",
        "max_size": "5000",
    },
    "Reporting:IntervalSeconds": "3",
})

# Bind config sections to typed dataclasses
app.configure_options(DatabaseOptions, "Database")
app.configure_options(CacheOptions, "Cache")

app.add_worker(OptionsReporterWorker)

app.run()
