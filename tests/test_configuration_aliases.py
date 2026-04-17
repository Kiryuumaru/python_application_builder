import os
import json
import pytest

from application_builder import ApplicationBuilder, IConfiguration


def _write_file(tmp_path: str, filename: str, content: str) -> str:
    path = os.path.join(tmp_path, filename)
    with open(path, "w") as f:
        f.write(content)
    return path


class TestAddJsonConfiguration:
    """Tests for ApplicationBuilder.add_json_configuration alias."""

    def test_loads_json_values(self, tmp_path):
        path = _write_file(str(tmp_path), "app.json", json.dumps({
            "App": {"Name": "TestApp", "Port": 8080}
        }))
        app = ApplicationBuilder()
        app.add_json_configuration(path)
        config = app._configuration_builder.build()

        assert config.get("App:Name") == "TestApp"
        assert config.get("App:Port") == "8080"

    def test_returns_self_for_chaining(self, tmp_path):
        path = _write_file(str(tmp_path), "app.json", json.dumps({"Key": "Value"}))
        app = ApplicationBuilder()
        result = app.add_json_configuration(path)
        assert result is app

    def test_equivalent_to_lambda(self, tmp_path):
        path = _write_file(str(tmp_path), "app.json", json.dumps({
            "Database": {"Host": "localhost"}
        }))

        app_alias = ApplicationBuilder()
        app_alias.add_json_configuration(path)
        config_alias = app_alias._configuration_builder.build()

        app_lambda = ApplicationBuilder()
        app_lambda.add_configuration(lambda b: b.add_json_file(path))
        config_lambda = app_lambda._configuration_builder.build()

        assert config_alias.get("Database:Host") == config_lambda.get("Database:Host")

    def test_missing_file_returns_empty(self):
        app = ApplicationBuilder()
        app.add_json_configuration("/nonexistent/path/config.json")
        config = app._configuration_builder.build()
        assert config.get("Missing:Key") is None

    def test_resolves_via_service_provider(self, tmp_path):
        path = _write_file(str(tmp_path), "app.json", json.dumps({
            "App": {"Name": "FromJson"}
        }))
        app = ApplicationBuilder()
        app.add_json_configuration(path)
        provider = app.build(auto_start_hosted_services=False)
        config = provider.get_required_service(IConfiguration)

        assert config.get("App:Name") == "FromJson"


class TestAddYamlConfiguration:
    """Tests for ApplicationBuilder.add_yaml_configuration alias."""

    def test_loads_yaml_values(self, tmp_path):
        path = _write_file(str(tmp_path), "app.yaml", "App:\n  Name: YamlApp\n  Debug: true\n")
        app = ApplicationBuilder()
        app.add_yaml_configuration(path)
        config = app._configuration_builder.build()

        assert config.get("App:Name") == "YamlApp"
        assert config.get("App:Debug") == "True"

    def test_returns_self_for_chaining(self, tmp_path):
        path = _write_file(str(tmp_path), "app.yaml", "Key: Value\n")
        app = ApplicationBuilder()
        result = app.add_yaml_configuration(path)
        assert result is app

    def test_equivalent_to_lambda(self, tmp_path):
        path = _write_file(str(tmp_path), "app.yaml", "Logging:\n  Level: DEBUG\n")

        app_alias = ApplicationBuilder()
        app_alias.add_yaml_configuration(path)
        config_alias = app_alias._configuration_builder.build()

        app_lambda = ApplicationBuilder()
        app_lambda.add_configuration(lambda b: b.add_yaml_file(path))
        config_lambda = app_lambda._configuration_builder.build()

        assert config_alias.get("Logging:Level") == config_lambda.get("Logging:Level")

    def test_missing_file_returns_empty(self):
        app = ApplicationBuilder()
        app.add_yaml_configuration("/nonexistent/path/config.yaml")
        config = app._configuration_builder.build()
        assert config.get("Missing:Key") is None

    def test_yml_extension(self, tmp_path):
        path = _write_file(str(tmp_path), "app.yml", "App:\n  Version: 2\n")
        app = ApplicationBuilder()
        app.add_yaml_configuration(path)
        config = app._configuration_builder.build()
        assert config.get("App:Version") == "2"

    def test_resolves_via_service_provider(self, tmp_path):
        path = _write_file(str(tmp_path), "app.yaml", "App:\n  Name: FromYaml\n")
        app = ApplicationBuilder()
        app.add_yaml_configuration(path)
        provider = app.build(auto_start_hosted_services=False)
        config = provider.get_required_service(IConfiguration)

        assert config.get("App:Name") == "FromYaml"


class TestAddEnvironmentVariablesConfiguration:
    """Tests for ApplicationBuilder.add_environment_variables_configuration alias."""

    def test_loads_env_var(self, monkeypatch):
        monkeypatch.setenv("TESTAPP_DB__HOST", "envhost")
        app = ApplicationBuilder()
        app.add_environment_variables_configuration("TESTAPP_")
        config = app._configuration_builder.build()

        assert config.get("DB:HOST") == "envhost"

    def test_returns_self_for_chaining(self):
        app = ApplicationBuilder()
        result = app.add_environment_variables_configuration()
        assert result is app

    def test_equivalent_to_lambda(self, monkeypatch):
        monkeypatch.setenv("CFGTEST_KEY", "cfgval")

        app_alias = ApplicationBuilder()
        app_alias.add_environment_variables_configuration("CFGTEST_")
        config_alias = app_alias._configuration_builder.build()

        app_lambda = ApplicationBuilder()
        app_lambda.add_configuration(lambda b: b.add_environment_variables("CFGTEST_"))
        config_lambda = app_lambda._configuration_builder.build()

        assert config_alias.get("KEY") == config_lambda.get("KEY")


class TestAddCommandLineConfiguration:
    """Tests for ApplicationBuilder.add_command_line_configuration alias."""

    def test_loads_command_line_args(self):
        app = ApplicationBuilder()
        app.add_command_line_configuration(args=["--App:Name=CLIApp"])
        config = app._configuration_builder.build()

        assert config.get("App:Name") == "CLIApp"

    def test_returns_self_for_chaining(self):
        app = ApplicationBuilder()
        result = app.add_command_line_configuration()
        assert result is app

    def test_equivalent_to_lambda(self):
        args = ["--Server:Port=9090"]

        app_alias = ApplicationBuilder()
        app_alias.add_command_line_configuration(args=args)
        config_alias = app_alias._configuration_builder.build()

        app_lambda = ApplicationBuilder()
        app_lambda.add_configuration(lambda b: b.add_command_line(args))
        config_lambda = app_lambda._configuration_builder.build()

        assert config_alias.get("Server:Port") == config_lambda.get("Server:Port")

    def test_switch_mappings(self):
        app = ApplicationBuilder()
        app.add_command_line_configuration(
            args=["--verbose"],
            switch_mappings={"--verbose": "App:Verbose"}
        )
        config = app._configuration_builder.build()

        assert config.get("App:Verbose") == "true"


class TestAddInMemoryConfiguration:
    """Tests for ApplicationBuilder.add_in_memory_configuration alias."""

    def test_loads_flat_dict(self):
        app = ApplicationBuilder()
        app.add_in_memory_configuration({"App:Name": "MemApp", "App:Port": "3000"})
        config = app._configuration_builder.build()

        assert config.get("App:Name") == "MemApp"
        assert config.get("App:Port") == "3000"

    def test_returns_self_for_chaining(self):
        app = ApplicationBuilder()
        result = app.add_in_memory_configuration({"Key": "Value"})
        assert result is app

    def test_equivalent_to_lambda(self):
        data = {"App:Name": "Test"}

        app_alias = ApplicationBuilder()
        app_alias.add_in_memory_configuration(data)
        config_alias = app_alias._configuration_builder.build()

        app_lambda = ApplicationBuilder()
        app_lambda.add_configuration(lambda b: b.add_in_memory_collection(data))
        config_lambda = app_lambda._configuration_builder.build()

        assert config_alias.get("App:Name") == config_lambda.get("App:Name")


class TestConfigurationAliasChaining:
    """Tests that multiple configuration aliases can be chained."""

    def test_chain_multiple_sources(self, tmp_path):
        json_path = _write_file(str(tmp_path), "app.json", json.dumps({
            "App": {"Name": "JsonApp", "Port": 3000}
        }))
        yaml_path = _write_file(str(tmp_path), "app.yaml", "App:\n  Name: YamlApp\n")

        app = (
            ApplicationBuilder()
            .add_json_configuration(json_path)
            .add_yaml_configuration(yaml_path)
            .add_in_memory_configuration({"App:Debug": "true"})
        )
        config = app._configuration_builder.build()

        assert config.get("App:Name") == "YamlApp"
        assert config.get("App:Port") == "3000"
        assert config.get("App:Debug") == "true"
