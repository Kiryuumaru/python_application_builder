import threading
import queue
from typing import Dict, List
from application_builder import ILogger
from interfaces import IDataSource, ITransformer, IDataSink, IMetrics


SAMPLE_DATA = [
    {"id": i, "name": f"record_{i}", "category": ["alpha", "beta", "gamma"][i % 3], "value": i * 10}
    for i in range(1, 21)
]


class CsvDataSource(IDataSource):
    def __init__(self, logger: ILogger):
        self._data = list(SAMPLE_DATA)
        self._lock = threading.Lock()
        self._logger = logger
        self._logger.info(f"DataSource initialized with {len(self._data)} records")

    def next_batch(self, size: int) -> List[Dict]:
        with self._lock:
            batch = self._data[:size]
            self._data = self._data[size:]
            return batch

    def is_exhausted(self) -> bool:
        with self._lock:
            return len(self._data) == 0


class UppercaseTransformer(ITransformer):
    def transform(self, record: Dict) -> Dict:
        result = dict(record)
        if "name" in result:
            result["name"] = result["name"].upper()
        if "category" in result:
            result["category"] = result["category"].upper()
        result["transformed"] = True
        return result


# Thread-safe buffer for producer -> consumer communication
_buffer: queue.Queue = queue.Queue(maxsize=100)


def get_buffer() -> queue.Queue:
    return _buffer


class JsonDataSink(IDataSink):
    def __init__(self, logger: ILogger):
        self._written: List[Dict] = []
        self._lock = threading.Lock()
        self._logger = logger

    def write(self, record: Dict) -> None:
        with self._lock:
            self._written.append(record)

    def flush(self) -> None:
        with self._lock:
            count = len(self._written)
        self._logger.info(f"[Sink] Flushed — {count} records total")

    def written_count(self) -> int:
        with self._lock:
            return len(self._written)


class InMemoryMetrics(IMetrics):
    def __init__(self):
        self._produced = 0
        self._consumed = 0
        self._errors = 0
        self._lock = threading.Lock()

    def record_produced(self, count: int) -> None:
        with self._lock:
            self._produced += count

    def record_consumed(self, count: int) -> None:
        with self._lock:
            self._consumed += count

    def record_error(self) -> None:
        with self._lock:
            self._errors += 1

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            return {
                "produced": self._produced,
                "consumed": self._consumed,
                "errors": self._errors,
            }
