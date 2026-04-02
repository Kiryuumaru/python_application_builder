from typing import List
from application_builder import Worker, ILogger
from interfaces import IGreeter, IFarewell


class DemoWorker(Worker):
    """Demonstrates the effect of TryAdd, Replace, and RemoveAll."""

    def __init__(self, greeters: List[IGreeter],
                 logger: ILogger):
        super().__init__()
        self._greeters = greeters
        self._logger = logger

    def execute(self) -> None:
        self._logger.info(f"[Demo] Resolved {len(self._greeters)} greeter(s)")
        for g in self._greeters:
            msg = g.greet("World")
            self._logger.info(f"  {g.__class__.__name__}: {msg}")

        self._logger.success("[Demo] Done")
