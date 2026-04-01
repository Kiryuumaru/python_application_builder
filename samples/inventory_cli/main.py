import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from application_builder import ApplicationBuilder
from interfaces import IInventory, ICommandLog
from services import InMemoryInventory, ScopedCommandLog
from workers import InventoryCommandWorker

app = ApplicationBuilder()

app.add_configuration_dictionary({
    "Inventory:CommandIntervalSeconds": "2",
})

# Singleton inventory shared across all scopes
app.add_singleton(IInventory, InMemoryInventory)

# Scoped command log — each scope gets its own audit trail
app.add_scoped(ICommandLog, ScopedCommandLog)

app.add_worker(InventoryCommandWorker)

app.run()
