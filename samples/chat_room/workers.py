from typing import List
from application_builder import Worker, ILogger, IConfiguration, ScopeFactory
from interfaces import IMessageFormatter, ISessionContext, IRoomRegistry


SCRIPT = [
    ("Alice", "general", "Hey everyone!"),
    ("Alice", "general", "How's the project going?"),
    ("Bob", "general", "Going great, just pushed a fix"),
    ("Charlie", "random", "Anyone up for lunch?"),
    ("Bob", "general", "Tests are passing now"),
    ("Alice", "random", "Sure, 12:30?"),
    ("Charlie", "random", "Sounds good"),
]


class ChatSimulatorWorker(Worker):
    """Simulates chat sessions using a separate DI scope per user action."""

    def __init__(self, scope_factory: ScopeFactory,
                 registry: IRoomRegistry,
                 config: IConfiguration,
                 logger: ILogger):
        super().__init__()
        self._scope_factory = scope_factory
        self._registry = registry
        self._logger = logger
        self._interval = config.get_float("Chat:SimulationIntervalSeconds", 1.0)

    def execute(self) -> None:
        import time

        self._logger.info("Chat simulation starting...")

        for user, room, text in SCRIPT:
            if self.is_stopping():
                break

            # Each message is handled in its own scope
            with self._scope_factory.create_scope_context() as scope:
                session = scope.get_required_service(ISessionContext)
                session.set_user(user)
                session.set_room(room)

                formatters: List[IMessageFormatter] = scope.get_services(IMessageFormatter)

                self._registry.join(room, user)

                formatted = text
                for fmt in formatters:
                    formatted = fmt.format(user, formatted)

                self._registry.post_message(room, formatted)
                self._logger.info(f"[#{room}] {formatted}")

            self.wait_for_stop(self._interval)

        # Print room histories
        self._logger.info("\n=== Chat History ===")
        for room_name in self._registry.list_rooms():
            history = self._registry.get_history(room_name)
            self._logger.info(f"\n  #{room_name} ({len(history)} messages):")
            for msg in history:
                self._logger.info(f"    {msg}")

        self._logger.success("Chat simulation complete")
