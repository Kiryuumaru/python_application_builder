from application_builder import Worker, ILogger, ScopeFactory
from interfaces import INotificationSender


class NotificationWorker(Worker):
    def __init__(self, logger: ILogger, scope_factory: ScopeFactory) -> None:
        super().__init__()
        self._logger = logger
        self._scope_factory = scope_factory

    def execute(self) -> None:
        self._logger.info("=== Keyed Services Demo ===")

        with self._scope_factory.create_scope_context() as scope:
            channels = ["email", "sms", "slack"]
            for channel in channels:
                sender = scope.get_required_keyed_service(INotificationSender, channel)
                result = sender.send(f"Hello from {channel} channel!")
                self._logger.info(result)

            # Demonstrate that unknown key returns None via get_keyed_service
            unknown = scope.get_keyed_service(INotificationSender, "pigeon")
            if unknown is None:
                self._logger.info("No 'pigeon' sender registered (as expected)")

        self._logger.info("Demo complete.")
