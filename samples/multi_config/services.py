from typing import Dict

from application_builder import IConfiguration, ILogger
from interfaces import IConfigReporter, IFeatureFlags


class ConfigReporter(IConfigReporter):
    """Reports configuration from all loaded sources."""

    def __init__(self, config: IConfiguration, logger: ILogger):
        self._config = config
        self._logger = logger

    def report(self) -> str:
        lines = ["=== Configuration Report ==="]

        # Read top-level app settings
        app_name = self._config.get("App:Name", "unknown")
        app_version = self._config.get("App:Version", "0.0.0")
        app_env = self._config.get("App:Environment", "unknown")
        lines.append(f"App: {app_name} v{app_version} ({app_env})")

        # Read database section — showcases get_section
        db_section = self._config.get_section("Database")
        db_host = db_section.get("Host", "unknown")
        db_port = db_section.get("Port", "0")
        db_name = db_section.get("Name", "unknown")
        lines.append(f"Database: {db_host}:{db_port}/{db_name}")

        # Show all children of Monitoring section — showcases get_children
        mon_section = self._config.get_section("Monitoring")
        children = mon_section.get_children()
        lines.append("Monitoring children:")
        for child in children:
            lines.append(f"  {child.key} = {child.value}")

        lines.append("============================")
        return "\n".join(lines)


class ConfigFeatureFlags(IFeatureFlags):
    """Reads feature flags from the Features config section."""

    def __init__(self, config: IConfiguration):
        self._config = config

    def is_enabled(self, feature: str) -> bool:
        return self._config.get_bool(f"Features:{feature}", False)

    def all_flags(self) -> Dict[str, bool]:
        section = self._config.get_section("Features")
        children = section.get_children()
        return {child.key: child.value.lower() in ("true", "1", "yes")
                for child in children if child.value}
