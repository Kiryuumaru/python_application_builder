"""
Inventory CLI — simulated CLI inventory management.

Showcases: ServiceProvider.create_scope(), CancellationToken.none(),
           CancellationToken.register(), add_singleton_instance
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, List


class IInventory(ABC):
    """Manages product inventory."""

    @abstractmethod
    def add_product(self, name: str, quantity: int, price: float) -> None:
        """Add or update a product."""

    @abstractmethod
    def remove_product(self, name: str, quantity: int) -> bool:
        """Remove quantity from a product. Return False if insufficient stock."""

    @abstractmethod
    def get_product(self, name: str) -> Optional[Dict]:
        """Get product details or None."""

    @abstractmethod
    def list_products(self) -> List[Dict]:
        """List all products."""

    @abstractmethod
    def total_value(self) -> float:
        """Get total inventory value."""


class ICommandLog(ABC):
    """Scoped log for a single command execution context."""

    @abstractmethod
    def log(self, message: str) -> None:
        """Log a message in this command context."""

    @abstractmethod
    def get_entries(self) -> List[str]:
        """Get all entries for this context."""

    @abstractmethod
    def get_scope_id(self) -> str:
        """Get the unique scope identifier."""
