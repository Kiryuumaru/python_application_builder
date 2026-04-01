import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from application_builder import ApplicationBuilder
from interfaces import IDataSource, ITransformer, IDataSink, IMetrics
from services import CsvDataSource, UppercaseTransformer, JsonDataSink, InMemoryMetrics
from workers import ProducerWorker, ConsumerWorker, MetricsWorker


app = ApplicationBuilder()
app.add_configuration_dictionary({
    "Pipeline": {
        "ProduceIntervalSeconds": "1",
        "ConsumeIntervalSeconds": "1",
        "MetricsIntervalSeconds": "5",
        "BatchSize": "3",
    }
})

app.add_singleton(IDataSource, CsvDataSource)
app.add_singleton(ITransformer, UppercaseTransformer)
app.add_singleton(IDataSink, JsonDataSink)
app.add_singleton(IMetrics, InMemoryMetrics)

# Three workers running concurrently
app.add_worker(ProducerWorker)
app.add_worker(ConsumerWorker)
app.add_worker(MetricsWorker)
app.run()
