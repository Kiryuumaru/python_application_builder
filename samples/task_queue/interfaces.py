from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from application_builder import CancellationToken


class ITaskRepository(ABC):
    @abstractmethod
    def enqueue(self, task: Dict) -> None:
        pass

    @abstractmethod
    def dequeue(self) -> Optional[Dict]:
        pass

    @abstractmethod
    def pending_count(self) -> int:
        pass

    @abstractmethod
    def mark_complete(self, task_id: int) -> None:
        pass

    @abstractmethod
    def mark_failed(self, task_id: int, error: str) -> None:
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, int]:
        pass


class ITaskProcessor(ABC):
    @abstractmethod
    def process(self, task: Dict, token: CancellationToken) -> bool:
        pass
