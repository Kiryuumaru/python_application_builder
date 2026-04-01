import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from application_builder import ApplicationBuilder
from interfaces import IBuildStep
from steps import LintStep, TestStep, PackageStep
from workers import BuildPipelineWorker


app = ApplicationBuilder()
app.add_configuration_dictionary({
    "Build": {
        "ProjectDir": os.path.dirname(__file__),
        "TimeoutSeconds": "30",
    }
})

# Multiple build steps — injected as List[IBuildStep] and run in order
app.add_singleton(IBuildStep, LintStep)
app.add_singleton(IBuildStep, TestStep)
app.add_singleton(IBuildStep, PackageStep)
app.add_worker(BuildPipelineWorker)
app.run()
