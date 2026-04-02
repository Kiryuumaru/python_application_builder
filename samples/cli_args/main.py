import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from application_builder import ApplicationBuilder
from workers import ConfigReporterWorker

app = ApplicationBuilder()

# In-memory defaults (lowest priority)
app.add_configuration_dictionary({
    "Server:Host": "localhost",
    "Server:Port": "8080",
    "Logging:Level": "INFO",
})

# Command-line args override defaults (highest priority)
# Uses add_configuration to access the ConfigurationBuilder directly
app.add_configuration(lambda cb: cb.add_command_line(
    args=["--Server:Host=0.0.0.0", "--Server:Port=9090", "--App:Name=CliDemo"],
))

app.add_worker(ConfigReporterWorker)

app.run()
