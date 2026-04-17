import os
import json
import tempfile
import pytest

from application_builder import (
    ApplicationBuilder,
    ConfigurationBuilder,
    YamlFileConfigurationProvider,
)


def _write_yaml(tmp_path: str, filename: str, content: str) -> str:
    """Write a YAML string to a temp file and return the full path."""
    path = os.path.join(tmp_path, filename)
    with open(path, "w") as f:
        f.write(content)
    return path


# ─── YamlFileConfigurationProvider ───────────────────────────────────


class TestYamlFileConfigurationProvider:
    """Tests for the YAML file configuration provider."""

    def test_string_value(self, tmp_path):
        path = _write_yaml(str(tmp_path), "app.yaml", "App:\n  Name: MyApp\n")
        provider = YamlFileConfigurationProvider(path)
        data = provider.load()
        assert data["App:Name"] == "MyApp"

    def test_integer_value(self, tmp_path):
        path = _write_yaml(str(tmp_path), "app.yaml", "Server:\n  Port: 8080\n")
        provider = YamlFileConfigurationProvider(path)
        data = provider.load()
        assert data["Server:Port"] == "8080"

    def test_float_value(self, tmp_path):
        path = _write_yaml(str(tmp_path), "app.yaml", "Math:\n  Pi: 3.14\n")
        provider = YamlFileConfigurationProvider(path)
        data = provider.load()
        assert data["Math:Pi"] == "3.14"

    def test_boolean_true(self, tmp_path):
        path = _write_yaml(str(tmp_path), "app.yaml", "Feature:\n  Enabled: true\n")
        provider = YamlFileConfigurationProvider(path)
        data = provider.load()
        assert data["Feature:Enabled"] == "True"

    def test_boolean_false(self, tmp_path):
        path = _write_yaml(str(tmp_path), "app.yaml", "Feature:\n  Enabled: false\n")
        provider = YamlFileConfigurationProvider(path)
        data = provider.load()
        assert data["Feature:Enabled"] == "False"

    def test_nested_values(self, tmp_path):
        yaml_content = (
            "Database:\n"
            "  Connection:\n"
            "    Host: localhost\n"
            "    Port: 5432\n"
        )
        path = _write_yaml(str(tmp_path), "app.yaml", yaml_content)
        provider = YamlFileConfigurationProvider(path)
        data = provider.load()
        assert data["Database:Connection:Host"] == "localhost"
        assert data["Database:Connection:Port"] == "5432"

    def test_list_value(self, tmp_path):
        yaml_content = "Tags:\n  Items:\n    - alpha\n    - beta\n    - gamma\n"
        path = _write_yaml(str(tmp_path), "app.yaml", yaml_content)
        provider = YamlFileConfigurationProvider(path)
        data = provider.load()
        assert data["Tags:Items"] == json.dumps(["alpha", "beta", "gamma"])

    def test_missing_file_returns_empty(self):
        provider = YamlFileConfigurationProvider("/nonexistent/path/config.yaml")
        data = provider.load()
        assert data == {}

    def test_empty_file_returns_empty(self, tmp_path):
        path = _write_yaml(str(tmp_path), "empty.yaml", "")
        provider = YamlFileConfigurationProvider(path)
        data = provider.load()
        assert data == {}

    def test_yml_extension(self, tmp_path):
        path = _write_yaml(str(tmp_path), "app.yml", "App:\n  Version: 2\n")
        provider = YamlFileConfigurationProvider(path)
        data = provider.load()
        assert data["App:Version"] == "2"

    def test_null_value(self, tmp_path):
        path = _write_yaml(str(tmp_path), "app.yaml", "App:\n  Optional: null\n")
        provider = YamlFileConfigurationProvider(path)
        data = provider.load()
        assert data["App:Optional"] == "None"

    def test_multiline_string(self, tmp_path):
        yaml_content = "App:\n  Description: |\n    Hello World\n"
        path = _write_yaml(str(tmp_path), "app.yaml", yaml_content)
        provider = YamlFileConfigurationProvider(path)
        data = provider.load()
        assert "Hello World" in data["App:Description"]


# ─── ConfigurationBuilder integration ────────────────────────────────


class TestYamlConfigurationBuilderIntegration:
    """Tests that YAML files integrate with ConfigurationBuilder."""

    def test_add_yaml_file(self, tmp_path):
        yaml_content = (
            "App:\n"
            "  Name: TestApp\n"
            "  Debug: true\n"
            "  MaxRetries: 3\n"
        )
        path = _write_yaml(str(tmp_path), "settings.yaml", yaml_content)

        builder = ConfigurationBuilder()
        builder.add_yaml_file(path)
        config = builder.build()

        assert config.get("App:Name") == "TestApp"
        assert config.get_bool("App:Debug") is True
        assert config.get_int("App:MaxRetries") == 3

    def test_yaml_overrides_json(self, tmp_path):
        json_path = os.path.join(str(tmp_path), "base.json")
        with open(json_path, "w") as f:
            json.dump({"App": {"Name": "FromJson", "Port": 3000}}, f)

        yaml_path = _write_yaml(
            str(tmp_path), "override.yaml", "App:\n  Name: FromYaml\n"
        )

        builder = ConfigurationBuilder()
        builder.add_json_file(json_path)
        builder.add_yaml_file(yaml_path)
        config = builder.build()

        assert config.get("App:Name") == "FromYaml"
        assert config.get_int("App:Port") == 3000

    def test_add_yaml_via_application_builder(self, tmp_path):
        yaml_content = "Logging:\n  Level: DEBUG\n"
        path = _write_yaml(str(tmp_path), "config.yml", yaml_content)

        app = ApplicationBuilder()
        app.add_configuration(lambda b: b.add_yaml_file(path))
        config = app._configuration_builder.build()

        assert config.get("Logging:Level") == "DEBUG"


# ─── Mixed value types in a single file ──────────────────────────────


class TestYamlMixedValueTypes:
    """Tests loading a YAML file with diverse value types."""

    def test_mixed_types(self, tmp_path):
        yaml_content = (
            "App:\n"
            "  Name: MixedApp\n"
            "  Port: 9090\n"
            "  Debug: false\n"
            "  Rate: 0.75\n"
            "  Tags:\n"
            "    - web\n"
            "    - api\n"
            "  Database:\n"
            "    Host: db.local\n"
            "    Port: 5432\n"
        )
        path = _write_yaml(str(tmp_path), "mixed.yaml", yaml_content)

        builder = ConfigurationBuilder()
        builder.add_yaml_file(path)
        config = builder.build()

        assert config.get("App:Name") == "MixedApp"
        assert config.get_int("App:Port") == 9090
        assert config.get_bool("App:Debug") is False
        assert config.get_float("App:Rate") == 0.75
        assert config.get("App:Tags") == json.dumps(["web", "api"])
        assert config.get("App:Database:Host") == "db.local"
        assert config.get_int("App:Database:Port") == 5432
