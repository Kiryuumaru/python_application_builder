from abc import ABC, abstractmethod


class IBuildStep(ABC):
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def run(self, project_dir: str) -> bool:
        """Run the step. Return True if passed."""
        pass
