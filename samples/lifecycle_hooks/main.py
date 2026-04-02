import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from application_builder import ApplicationBuilder
from workers import StartupListener, HeartbeatWorker

app = ApplicationBuilder()

app.add_configuration_dictionary({
    "Heartbeat:IntervalSeconds": "3",
})

app.add_worker(StartupListener)
app.add_worker(HeartbeatWorker)

app.run()
