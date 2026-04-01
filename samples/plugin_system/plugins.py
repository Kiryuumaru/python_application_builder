import gzip
import os
from application_builder import ILogger, IConfiguration
from interfaces import IStoragePlugin, ICompressionPlugin, INotificationPlugin


class LocalFileStorage(IStoragePlugin):
    def __init__(self, base_dir: str, logger: ILogger):
        self._base_dir = base_dir
        self._logger = logger

    def store(self, name: str, data: bytes) -> str:
        os.makedirs(self._base_dir, exist_ok=True)
        path = os.path.join(self._base_dir, name)
        with open(path, 'wb') as f:
            f.write(data)
        self._logger.info(f"[LocalStorage] Wrote {len(data)} bytes -> {path}")
        return path

    def exists(self, name: str) -> bool:
        return os.path.exists(os.path.join(self._base_dir, name))


class S3Storage(IStoragePlugin):
    def __init__(self, bucket: str, logger: ILogger):
        self._bucket = bucket
        self._logger = logger
        self._files: dict = {}

    def store(self, name: str, data: bytes) -> str:
        key = f"s3://{self._bucket}/{name}"
        self._files[name] = data
        self._logger.info(f"[S3] Uploaded {len(data)} bytes -> {key}")
        return key

    def exists(self, name: str) -> bool:
        return name in self._files


class GzipCompression(ICompressionPlugin):
    def __init__(self, level: int = 6):
        self._level = level

    def compress(self, data: bytes) -> bytes:
        return gzip.compress(data, compresslevel=self._level)

    def get_extension(self) -> str:
        return ".gz"


class NoCompression(ICompressionPlugin):
    def compress(self, data: bytes) -> bytes:
        return data

    def get_extension(self) -> str:
        return ""


class ConsoleNotifier(INotificationPlugin):
    def __init__(self, logger: ILogger):
        self._logger = logger

    def notify(self, subject: str, body: str) -> None:
        self._logger.info(f"[ConsoleNotify] {subject}: {body}")


class WebhookNotifier(INotificationPlugin):
    def __init__(self, config: IConfiguration, logger: ILogger):
        self._url = config.get("Plugins:Notifications:Webhook:Url", "https://hooks.example.com")
        self._logger = logger

    def notify(self, subject: str, body: str) -> None:
        self._logger.info(f"[Webhook -> {self._url}] {subject}: {body}")
