import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from application_builder import ApplicationBuilder
from interfaces import IConfigReporter, IFeatureFlags
from services import ConfigReporter, ConfigFeatureFlags
from workers import ConfigMonitorWorker

app = ApplicationBuilder()

# Load configuration from JSON file — showcases add_json_file
json_path = os.path.join(os.path.dirname(__file__), 'appsettings.json')
app.add_configuration(lambda cb: cb.add_json_file(json_path))

# Load environment variables with prefix — showcases add_environment_variables
# Set MYAPP_Features__BetaAPI=true to override the JSON value
app.add_configuration(lambda cb: cb.add_environment_variables(prefix="MYAPP_"))

# In-memory overrides take highest priority (loaded last)
app.add_configuration_dictionary({
    "App:Environment": "staging",
})

app.add_singleton(IConfigReporter, ConfigReporter)
app.add_singleton(IFeatureFlags, ConfigFeatureFlags)
app.add_worker(ConfigMonitorWorker)

app.run()
