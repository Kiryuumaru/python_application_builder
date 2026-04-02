from application_builder import ILogger
from interfaces import IEmailService, ICache


class SmtpEmailService(IEmailService):
    def __init__(self, logger: ILogger) -> None:
        self._logger = logger

    def send(self, to: str, body: str) -> None:
        self._logger.info(f"Sending email to {to}: {body}")


class RedisCache(ICache):
    def __init__(self, logger: ILogger) -> None:
        self._logger = logger

    def get(self, key: str) -> str:
        self._logger.info(f"Cache lookup: {key}")
        return f"cached_{key}"
