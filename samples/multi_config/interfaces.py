"""
Multi Config — demonstrates multiple configuration sources.

Showcases: add_json_file, add_environment_variables(prefix),
           config.reload(), get_section(), get_children()
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional


class IConfigReporter(ABC):
    """Reports configuration values from different sources."""

    @abstractmethod
    def report(self) -> str:
        """Generate a configuration report."""


class IFeatureFlags(ABC):
    """Reads feature flags from configuration."""

    @abstractmethod
    def is_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled."""

    @abstractmethod
    def all_flags(self) -> Dict[str, bool]:
        """Get all feature flags."""
