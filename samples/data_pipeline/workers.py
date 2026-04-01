from application_builder import TimedWorker, ILogger, IConfiguration
from interfaces import IDataSource, ITransformer, IDataSink, IMetrics
from services import get_buffer


class ProducerWorker(TimedWorker):
    """Reads batches from the data source and pushes to the buffer."""

    def __init__(self, source: IDataSource,
                 metrics: IMetrics,
                 config: IConfiguration,
                 logger: ILogger):
        interval = config.get_float("Pipeline:ProduceIntervalSeconds", 1.0)
        super().__init__(interval_seconds=interval)
        self._source = source
        self._metrics = metrics
        self._batch_size = config.get_int("Pipeline:BatchSize", 3)
        self._logger = logger

    def do_work(self) -> None:
        if self._source.is_exhausted():
            self._logger.debug("[Producer] Source exhausted")
            return

        batch = self._source.next_batch(self._batch_size)
        if not batch:
            return

        buf = get_buffer()
        for record in batch:
            buf.put(record)

        self._metrics.record_produced(len(batch))
        self._logger.info(f"[Producer] Pushed {len(batch)} records to buffer")


class ConsumerWorker(TimedWorker):
    """Pulls from buffer, transforms, and writes to sink."""

    def __init__(self, transformer: ITransformer,
                 sink: IDataSink,
                 metrics: IMetrics,
                 config: IConfiguration,
                 logger: ILogger):
        interval = config.get_float("Pipeline:ConsumeIntervalSeconds", 1.0)
        super().__init__(interval_seconds=interval)
        self._transformer = transformer
        self._sink = sink
        self._metrics = metrics
        self._logger = logger

    def do_work(self) -> None:
        buf = get_buffer()
        consumed = 0
        while not buf.empty():
            try:
                record = buf.get_nowait()
                transformed = self._transformer.transform(record)
                self._sink.write(transformed)
                consumed += 1
            except Exception as e:
                self._metrics.record_error()
                self._logger.error(f"[Consumer] Error: {e}")

        if consumed > 0:
            self._metrics.record_consumed(consumed)
            self._sink.flush()
            self._logger.info(f"[Consumer] Processed {consumed} records (total: {self._sink.written_count()})")


class MetricsWorker(TimedWorker):
    """Periodically reports pipeline metrics."""

    def __init__(self, metrics: IMetrics,
                 config: IConfiguration,
                 logger: ILogger):
        interval = config.get_float("Pipeline:MetricsIntervalSeconds", 5.0)
        super().__init__(interval_seconds=interval)
        self._metrics = metrics
        self._logger = logger

    def do_work(self) -> None:
        stats = self._metrics.snapshot()
        self._logger.info(
            f"[Metrics] produced={stats['produced']} "
            f"consumed={stats['consumed']} errors={stats['errors']}"
        )
