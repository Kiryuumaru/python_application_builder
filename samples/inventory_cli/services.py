import threading
import uuid
from typing import Dict, Optional, List

from interfaces import IInventory, ICommandLog


class InMemoryInventory(IInventory):
    """Thread-safe in-memory inventory with pre-seeded products."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._products: Dict[str, Dict] = {
            "widget": {"name": "widget", "quantity": 100, "price": 9.99},
            "gadget": {"name": "gadget", "quantity": 50, "price": 24.99},
            "doohickey": {"name": "doohickey", "quantity": 200, "price": 4.99},
            "thingamajig": {"name": "thingamajig", "quantity": 30, "price": 49.99},
        }

    def add_product(self, name: str, quantity: int, price: float) -> None:
        with self._lock:
            if name in self._products:
                self._products[name]["quantity"] += quantity
                self._products[name]["price"] = price
            else:
                self._products[name] = {
                    "name": name,
                    "quantity": quantity,
                    "price": price,
                }

    def remove_product(self, name: str, quantity: int) -> bool:
        with self._lock:
            product = self._products.get(name)
            if product is None or product["quantity"] < quantity:
                return False
            product["quantity"] -= quantity
            return True

    def get_product(self, name: str) -> Optional[Dict]:
        with self._lock:
            p = self._products.get(name)
            return dict(p) if p else None

    def list_products(self) -> List[Dict]:
        with self._lock:
            return [dict(p) for p in self._products.values()]

    def total_value(self) -> float:
        with self._lock:
            return sum(p["quantity"] * p["price"] for p in self._products.values())


class ScopedCommandLog(ICommandLog):
    """Per-scope command execution log."""

    def __init__(self) -> None:
        self._id = str(uuid.uuid4())[:8]
        self._entries: List[str] = []

    def log(self, message: str) -> None:
        self._entries.append(f"[{self._id}] {message}")

    def get_entries(self) -> List[str]:
        return list(self._entries)

    def get_scope_id(self) -> str:
        return self._id
