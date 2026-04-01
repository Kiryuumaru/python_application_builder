import time
from typing import List
from application_builder import Worker, ILogger, IConfiguration
from interfaces import IStoragePlugin, ICompressionPlugin, INotificationPlugin


class BackupWorker(Worker):
    def __init__(self, storage: IStoragePlugin,
                 compression: ICompressionPlugin,
                 notifiers: List[INotificationPlugin],
                 config: IConfiguration,
                 logger: ILogger):
        super().__init__()
        self._storage = storage
        self._compression = compression
        self._notifiers = notifiers
        self._logger = logger

        backup_section = config.get_section("Backup")
        source_dirs_raw = backup_section.get("SourceDirs", "")
        self._source_dirs = [d.strip() for d in source_dirs_raw.split(",") if d.strip()]

    def execute(self) -> None:
        self._logger.info(f"Backing up {len(self._source_dirs)} directories...")

        for source_dir in self._source_dirs:
            if self.is_stopping():
                break
            self._backup_one(source_dir)

        for notifier in self._notifiers:
            notifier.notify("Backup Complete", f"Processed {len(self._source_dirs)} directories")

        self._logger.success("Backup worker finished")

    def _backup_one(self, source_dir: str) -> None:
        self._logger.info(f"  Backing up: {source_dir}")

        simulated_data = f"Contents of {source_dir} at {time.time()}".encode()
        compressed = self._compression.compress(simulated_data)
        ratio = len(compressed) / len(simulated_data) * 100 if simulated_data else 0
        self._logger.info(f"  Compressed: {len(simulated_data)} -> {len(compressed)} bytes ({ratio:.0f}%)")

        ext = self._compression.get_extension()
        dir_name = source_dir.replace("/", "_").strip("_")
        location = self._storage.store(f"backup_{dir_name}{ext}", compressed)
        self._logger.success(f"  Stored: {location}")
