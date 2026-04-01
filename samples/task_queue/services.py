import time
import threading
from typing import Dict, List, Optional
from application_builder import ILogger, CancellationToken
from interfaces import ITaskRepository, ITaskProcessor


class InMemoryTaskRepository(ITaskRepository):
    def __init__(self, logger: ILogger):
        self._queue: List[Dict] = []
        self._completed: List[int] = []
        self._failed: Dict[int, str] = {}
        self._lock = threading.Lock()
        self._logger = logger

        # Self-seed on creation
        seed_tasks = [
            {"id": 1, "name": "Send welcome email", "duration": 2},
            {"id": 2, "name": "Generate PDF report", "duration": 3},
            {"id": 3, "name": "Resize images", "duration": 1},
            {"id": 4, "name": "Sync inventory", "duration": 4},
            {"id": 5, "name": "Run slow analytics", "duration": 8},
            {"id": 6, "name": "Cleanup temp files", "duration": 1},
        ]
        for task in seed_tasks:
            self.enqueue(task)

    def enqueue(self, task: Dict) -> None:
        with self._lock:
            self._queue.append(task)
        self._logger.debug(f"Enqueued task #{task['id']}: {task['name']}")

    def dequeue(self) -> Optional[Dict]:
        with self._lock:
            if self._queue:
                return self._queue.pop(0)
        return None

    def pending_count(self) -> int:
        with self._lock:
            return len(self._queue)

    def mark_complete(self, task_id: int) -> None:
        with self._lock:
            self._completed.append(task_id)

    def mark_failed(self, task_id: int, error: str) -> None:
        with self._lock:
            self._failed[task_id] = error

    def get_stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "pending": len(self._queue),
                "completed": len(self._completed),
                "failed": len(self._failed),
            }


class SimulatedTaskProcessor(ITaskProcessor):
    def __init__(self, logger: ILogger):
        self._logger = logger

    def process(self, task: Dict, token: CancellationToken) -> bool:
        task_id = task["id"]
        name = task["name"]
        duration = task.get("duration", 2)

        self._logger.info(f"Processing task #{task_id}: {name} (est. {duration}s)")

        elapsed = 0.0
        while elapsed < duration:
            if token.is_cancellation_requested:
                self._logger.warning(f"Task #{task_id} cancelled after {elapsed:.1f}s")
                return False
            step = min(0.5, duration - elapsed)
            time.sleep(step)
            elapsed += step

        self._logger.success(f"Task #{task_id} completed in {elapsed:.1f}s")
        return True
