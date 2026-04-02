from abc import ABC, abstractmethod


class IGreeter(ABC):
    """Greets a user."""

    @abstractmethod
    def greet(self, name: str) -> str:
        """Return a greeting message."""


class IFarewell(ABC):
    """Says goodbye."""

    @abstractmethod
    def farewell(self, name: str) -> str:
        """Return a farewell message."""
