from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class IDataSource(ABC):
    @abstractmethod
    def next_batch(self, size: int) -> List[Dict]:
        pass

    @abstractmethod
    def is_exhausted(self) -> bool:
        pass


class ITransformer(ABC):
    @abstractmethod
    def transform(self, record: Dict) -> Dict:
        pass


class IDataSink(ABC):
    @abstractmethod
    def write(self, record: Dict) -> None:
        pass

    @abstractmethod
    def flush(self) -> None:
        pass

    @abstractmethod
    def written_count(self) -> int:
        pass


class IMetrics(ABC):
    @abstractmethod
    def record_produced(self, count: int) -> None:
        pass

    @abstractmethod
    def record_consumed(self, count: int) -> None:
        pass

    @abstractmethod
    def record_error(self) -> None:
        pass

    @abstractmethod
    def snapshot(self) -> Dict[str, int]:
        pass
