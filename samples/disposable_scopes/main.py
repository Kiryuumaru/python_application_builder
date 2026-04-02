import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from application_builder import ApplicationBuilder
from interfaces import IConnection
from services import ScopedConnection
from workers import DatabaseWorker

app = ApplicationBuilder()

app.add_configuration_dictionary({
    "Database:IntervalSeconds": "2",
})

# Scoped connection — new per scope, disposed on scope exit
app.add_scoped(IConnection, ScopedConnection)

app.add_worker(DatabaseWorker)

app.run()
