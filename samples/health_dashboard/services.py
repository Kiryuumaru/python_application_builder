import threading
import time
from typing import List, Dict
from interfaces import IHealthRegistry


class InMemoryHealthRegistry(IHealthRegistry):
    """Thread-safe health registry holding endpoint status in memory."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._status: Dict[str, Dict] = {}
        self._endpoints: List[str] = []

    def add_endpoint(self, endpoint: str) -> None:
        """Register an endpoint to monitor."""
        with self._lock:
            if endpoint not in self._endpoints:
                self._endpoints.append(endpoint)
                self._status[endpoint] = {
                    "healthy": None,
                    "latency_ms": 0.0,
                    "last_check": None,
                    "consecutive_failures": 0,
                }

    def record(self, endpoint: str, healthy: bool, latency_ms: float) -> None:
        with self._lock:
            entry = self._status.get(endpoint, {})
            if healthy:
                entry["consecutive_failures"] = 0
            else:
                entry["consecutive_failures"] = entry.get("consecutive_failures", 0) + 1
            entry["healthy"] = healthy
            entry["latency_ms"] = latency_ms
            entry["last_check"] = time.time()
            self._status[endpoint] = entry

    def get_status(self) -> Dict[str, Dict]:
        with self._lock:
            return dict(self._status)

    def get_endpoints(self) -> List[str]:
        with self._lock:
            return list(self._endpoints)
