import threading
import time
from typing import List, Dict, Optional

from interfaces import IJobDefinition, IJobRegistry


class JobDefinition(IJobDefinition):
    """Simple job definition with command, timeout, and working directory."""

    def __init__(self, name: str, command: List[str],
                 timeout_seconds: Optional[float] = None,
                 cwd: Optional[str] = None) -> None:
        self._name = name
        self._command = command
        self._timeout = timeout_seconds
        self._cwd = cwd

    def get_name(self) -> str:
        return self._name

    def get_command(self) -> List[str]:
        return self._command

    def get_timeout_seconds(self) -> Optional[float]:
        return self._timeout

    def get_cwd(self) -> Optional[str]:
        return self._cwd


class InMemoryJobRegistry(IJobRegistry):
    """Thread-safe job registry with pre-loaded job definitions."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: List[IJobDefinition] = [
            JobDefinition("echo-hello", ["echo", "Hello from job scheduler!"]),
            JobDefinition("list-dir", ["ls", "-la"], timeout_seconds=5.0),
            JobDefinition("slow-task", ["sleep", "2"], timeout_seconds=3.0),
            JobDefinition("timed-out-task", ["sleep", "30"], timeout_seconds=2.0),
            JobDefinition("python-calc", [
                "python3", "-c", "print(sum(range(1000)))"
            ], timeout_seconds=5.0),
        ]
        self._results: Dict[str, Dict] = {}

    def get_jobs(self) -> List[IJobDefinition]:
        with self._lock:
            return list(self._jobs)

    def record_result(self, name: str, success: bool, detail: str) -> None:
        with self._lock:
            self._results[name] = {
                "success": success,
                "detail": detail,
                "timestamp": time.time(),
            }

    def get_results(self) -> Dict[str, Dict]:
        with self._lock:
            return dict(self._results)
