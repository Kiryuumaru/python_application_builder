import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from application_builder import ApplicationBuilder
from workers import EnvironmentReporterWorker

app = ApplicationBuilder()

# IHostEnvironment reads from config — these override defaults
app.add_configuration_dictionary({
    "Environment": "Development",
    "ApplicationName": "EnvAwareDemo",
    "ContentRoot": os.path.dirname(__file__),
})

app.add_worker(EnvironmentReporterWorker)

app.run()
