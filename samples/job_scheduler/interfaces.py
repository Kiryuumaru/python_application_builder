"""
Job Scheduler — runs CLI tasks with timeout and cancellation.

Showcases: CliRunnerService, cancel_after, create_linked_token,
           CancellationTokenSource, CancellationToken.register
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class IJobDefinition(ABC):
    """Defines a schedulable CLI job."""

    @abstractmethod
    def get_name(self) -> str:
        """Get the job name."""

    @abstractmethod
    def get_command(self) -> List[str]:
        """Get the command to execute."""

    @abstractmethod
    def get_timeout_seconds(self) -> Optional[float]:
        """Get the timeout, or None for no timeout."""

    @abstractmethod
    def get_cwd(self) -> Optional[str]:
        """Get the working directory, or None for default."""


class IJobRegistry(ABC):
    """Holds the list of jobs to execute."""

    @abstractmethod
    def get_jobs(self) -> List[IJobDefinition]:
        """Get all registered job definitions."""

    @abstractmethod
    def record_result(self, name: str, success: bool, detail: str) -> None:
        """Record the result of a job execution."""

    @abstractmethod
    def get_results(self) -> Dict[str, Dict]:
        """Get all job results."""
