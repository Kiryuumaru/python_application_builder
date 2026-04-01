"""
Health Dashboard — monitors service endpoints and reports status.

Showcases: add_singleton_instance, config.get_list, TimedWorker
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class IHealthRegistry(ABC):
    """Central registry holding endpoint health status."""

    @abstractmethod
    def record(self, endpoint: str, healthy: bool, latency_ms: float) -> None:
        """Record a health check result."""

    @abstractmethod
    def get_status(self) -> Dict[str, Dict]:
        """Get current status of all endpoints."""

    @abstractmethod
    def get_endpoints(self) -> List[str]:
        """Get list of monitored endpoints."""
