import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from application_builder import ApplicationBuilder, IConfiguration, ILogger, ServiceProvider
from interfaces import IStoragePlugin, ICompressionPlugin, INotificationPlugin
from plugins import LocalFileStorage, S3Storage, GzipCompression, NoCompression, ConsoleNotifier, WebhookNotifier
from workers import BackupWorker


def storage_factory(provider: ServiceProvider) -> IStoragePlugin:
    config = provider.get_required_service(IConfiguration)
    logger = provider.get_required_service(ILogger)
    storage_type = config.get("Plugins:Storage:Type", "local")
    if storage_type == "s3":
        return S3Storage(config.get("Plugins:Storage:Bucket", "my-bucket"), logger)
    return LocalFileStorage(config.get("Plugins:Storage:BasePath", "/tmp/backups"), logger)


def compression_factory(provider: ServiceProvider) -> ICompressionPlugin:
    config = provider.get_required_service(IConfiguration)
    if config.get_bool("Plugins:Compression:Enabled", True):
        return GzipCompression(config.get_int("Plugins:Compression:Level", 6))
    return NoCompression()


app = ApplicationBuilder()
app.add_configuration_dictionary({
    "Plugins": {
        "Storage": {"Type": "local", "BasePath": "/tmp/backups"},
        "Compression": {"Enabled": "true", "Level": "9"},
    },
    "Backup": {
        "SourceDirs": "/home/user/docs,/home/user/photos",
    },
})

app.add_singleton_factory(IStoragePlugin, storage_factory)
app.add_singleton_factory(ICompressionPlugin, compression_factory)
app.add_singleton(INotificationPlugin, ConsoleNotifier)
app.add_singleton(INotificationPlugin, WebhookNotifier)
app.add_worker(BackupWorker)
app.run()
