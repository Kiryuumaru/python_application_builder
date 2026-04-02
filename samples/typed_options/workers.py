from application_builder import (
    TimedWorker, ILogger, IConfiguration,
    IOptions, bind_configuration,
)
from options import DatabaseOptions, CacheOptions


class OptionsReporterWorker(TimedWorker):
    """Periodically reports the current typed options."""

    def __init__(self, db_options: IOptions,
                 config: IConfiguration,
                 logger: ILogger):
        interval = config.get_float("Reporting:IntervalSeconds", 3.0)
        super().__init__(interval_seconds=interval)
        self._db_options = db_options
        self._config = config
        self._logger = logger

    def do_work(self) -> None:
        # IOptions resolves the last registered type (CacheOptions here)
        # For multiple option types, use bind_configuration with get_section
        db_section = self._config.get_section("Database")
        db = bind_configuration(db_section, DatabaseOptions)
        self._logger.info(
            f"[DB Options] host={db.host} port={db.port} "
            f"name={db.name} max_conn={db.max_connections}"
        )

        cache = self._db_options.get_value()
        self._logger.info(
            f"[Cache Options] enabled={cache.enabled} "
            f"ttl={cache.ttl_seconds}s max_size={cache.max_size}"
        )
