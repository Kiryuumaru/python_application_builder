import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from application_builder import ApplicationBuilder
from interfaces import IJobRegistry
from services import InMemoryJobRegistry
from workers import JobRunnerWorker, ResultReporterWorker

app = ApplicationBuilder()

app.add_configuration_dictionary({
    "Scheduler:ReportIntervalSeconds": "8",
})

app.add_singleton(IJobRegistry, InMemoryJobRegistry)

app.add_worker(JobRunnerWorker)
app.add_worker(ResultReporterWorker)

app.run()
