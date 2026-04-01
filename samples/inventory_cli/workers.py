import random

from application_builder import (
    TimedWorker, ILogger, IConfiguration, ScopeFactory,
    CancellationToken, CancellationTokenSource,
)
from interfaces import IInventory, ICommandLog


# Simulated CLI commands that exercise inventory operations
COMMANDS = [
    ("list", None, None),
    ("add", "sprocket", 25),
    ("remove", "widget", 10),
    ("get", "gadget", None),
    ("remove", "doohickey", 5),
    ("add", "widget", 50),
    ("get", "sprocket", None),
    ("total", None, None),
    ("remove", "thingamajig", 100),
    ("list", None, None),
]


class InventoryCommandWorker(TimedWorker):
    """Processes simulated CLI commands, each in its own scope."""

    def __init__(self, scope_factory: ScopeFactory,
                 config: IConfiguration,
                 logger: ILogger):
        interval = config.get_float("Inventory:CommandIntervalSeconds", 2.0)
        super().__init__(interval_seconds=interval)
        self._scope_factory = scope_factory
        self._logger = logger
        self._cmd_index = 0

    def do_work(self) -> None:
        if self._cmd_index >= len(COMMANDS):
            self._logger.info("[CLI] All commands processed")
            return

        cmd, name, qty = COMMANDS[self._cmd_index]
        self._cmd_index += 1

        # Each command uses its own scope — showcases create_scope via ScopeFactory
        with self._scope_factory.create_scope_context() as scope:
            cmd_log = scope.get_required_service(ICommandLog)
            inventory = scope.get_required_service(IInventory)

            # CancellationToken.none() — a token that's never cancelled
            token = CancellationToken.none()

            # Register a callback on a token — showcases token.register()
            source = CancellationTokenSource()
            with source.token.register(
                lambda: self._logger.debug(f"[CLI] Command '{cmd}' was cancelled")
            ):
                self._execute_command(cmd, name, qty, inventory, cmd_log)

            entries = cmd_log.get_entries()
            for entry in entries:
                self._logger.info(entry)

    def _execute_command(self, cmd: str, name: str, qty: int,
                         inventory: IInventory, cmd_log: ICommandLog) -> None:
        if cmd == "list":
            products = inventory.list_products()
            cmd_log.log(f"LIST -> {len(products)} products")
            for p in products:
                cmd_log.log(f"  {p['name']:15s} qty={p['quantity']:4d} ${p['price']:.2f}")

        elif cmd == "add":
            price = round(random.uniform(5, 100), 2)
            inventory.add_product(name, qty, price)
            cmd_log.log(f"ADD {name} x{qty} @ ${price:.2f}")

        elif cmd == "remove":
            ok = inventory.remove_product(name, qty)
            status = "OK" if ok else "INSUFFICIENT STOCK"
            cmd_log.log(f"REMOVE {name} x{qty} -> {status}")

        elif cmd == "get":
            product = inventory.get_product(name)
            if product:
                cmd_log.log(f"GET {name} -> qty={product['quantity']} ${product['price']:.2f}")
            else:
                cmd_log.log(f"GET {name} -> NOT FOUND")

        elif cmd == "total":
            val = inventory.total_value()
            cmd_log.log(f"TOTAL VALUE -> ${val:.2f}")
