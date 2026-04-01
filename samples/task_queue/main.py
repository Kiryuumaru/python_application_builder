import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from application_builder import ApplicationBuilder
from interfaces import ITaskRepository, ITaskProcessor
from services import InMemoryTaskRepository, SimulatedTaskProcessor
from workers import TaskDispatcherWorker


app = ApplicationBuilder()
app.add_configuration_dictionary({
    "TaskQueue": {
        "MaxConcurrent": "3",
        "TaskTimeoutSeconds": "5",
        "DispatchIntervalSeconds": "2",
    }
})

app.add_singleton(ITaskRepository, InMemoryTaskRepository)
app.add_singleton(ITaskProcessor, SimulatedTaskProcessor)
app.add_worker(TaskDispatcherWorker)
app.run()
