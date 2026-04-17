import threading
import json
import pytest
from dataclasses import dataclass, field
from typing import List, Dict

from application_builder import (
    bind_configuration,
    ApplicationBuilder,
    ConfigurationBuilder,
    IOptions,
    IOptionsSnapshot,
    IOptionsMonitor,
    IConfiguration,
    _OptionsImpl,
    _OptionsSnapshotImpl,
    _OptionsMonitorImpl,
)


def _build_config(data):
    return ConfigurationBuilder().add_in_memory_collection(data).build()


# ------------------------------------------------------------------
# bind_configuration — dataclass targets
# ------------------------------------------------------------------


@dataclass
class DatabaseSettings:
    host: str = ""
    port: int = 0
    use_ssl: bool = False
    timeout: float = 0.0


@dataclass
class SettingsWithCollections:
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class SettingsWithDefaults:
    name: str = "default_name"
    retries: int = 3


class TestBindConfigurationDataclass:
    def test_binds_string_field(self):
        config = _build_config({"Db:host": "localhost"})
        section = config.get_section("Db")
        result = bind_configuration(section, DatabaseSettings)
        assert result.host == "localhost"

    def test_binds_int_field(self):
        config = _build_config({"Db:port": "5432"})
        section = config.get_section("Db")
        result = bind_configuration(section, DatabaseSettings)
        assert result.port == 5432

    def test_binds_float_field(self):
        config = _build_config({"Db:timeout": "30.5"})
        section = config.get_section("Db")
        result = bind_configuration(section, DatabaseSettings)
        assert result.timeout == 30.5

    def test_binds_bool_field_true(self):
        config = _build_config({"Db:use_ssl": "true"})
        section = config.get_section("Db")
        result = bind_configuration(section, DatabaseSettings)
        assert result.use_ssl is True

    def test_binds_bool_field_false(self):
        config = _build_config({"Db:use_ssl": "false"})
        section = config.get_section("Db")
        result = bind_configuration(section, DatabaseSettings)
        assert result.use_ssl is False

    def test_uses_defaults_for_missing_fields(self):
        config = _build_config({"Db:host": "myhost"})
        section = config.get_section("Db")
        result = bind_configuration(section, DatabaseSettings)
        assert result.host == "myhost"
        assert result.port == 0
        assert result.use_ssl is False
        assert result.timeout == 0.0

    def test_uses_explicit_defaults_when_field_missing(self):
        config = _build_config({})
        section = config.get_section("App")
        result = bind_configuration(section, SettingsWithDefaults)
        assert result.name == "default_name"
        assert result.retries == 3

    def test_binds_all_fields(self):
        config = _build_config({
            "Db:host": "prod-server",
            "Db:port": "3306",
            "Db:use_ssl": "yes",
            "Db:timeout": "10.0",
        })
        section = config.get_section("Db")
        result = bind_configuration(section, DatabaseSettings)
        assert result.host == "prod-server"
        assert result.port == 3306
        assert result.use_ssl is True
        assert result.timeout == 10.0

    def test_binds_list_from_json(self):
        config = _build_config({"S:tags": '["a", "b", "c"]'})
        section = config.get_section("S")
        result = bind_configuration(section, SettingsWithCollections)
        assert result.tags == ["a", "b", "c"]

    def test_binds_list_from_comma_separated(self):
        config = _build_config({"S:tags": "x, y, z"})
        section = config.get_section("S")
        result = bind_configuration(section, SettingsWithCollections)
        assert result.tags == ["x", "y", "z"]

    def test_binds_dict_from_json(self):
        config = _build_config({"S:metadata": '{"key1": "val1"}'})
        section = config.get_section("S")
        result = bind_configuration(section, SettingsWithCollections)
        assert result.metadata == {"key1": "val1"}

    def test_uses_default_factory_for_missing_collection(self):
        config = _build_config({})
        section = config.get_section("S")
        result = bind_configuration(section, SettingsWithCollections)
        assert result.tags == []
        assert result.metadata == {}


# ------------------------------------------------------------------
# bind_configuration — plain class targets
# ------------------------------------------------------------------


class PlainSettings:
    def __init__(self, host: str = "", port: int = 0, enabled: bool = False):
        self.host = host
        self.port = port
        self.enabled = enabled


class TestBindConfigurationPlainClass:
    def test_binds_fields_from_section(self):
        config = _build_config({
            "App:host": "example.com",
            "App:port": "8080",
            "App:enabled": "true",
        })
        section = config.get_section("App")
        result = bind_configuration(section, PlainSettings)
        assert result.host == "example.com"
        assert result.port == 8080
        assert result.enabled is True

    def test_missing_fields_use_class_defaults(self):
        config = _build_config({"App:host": "myhost"})
        section = config.get_section("App")
        result = bind_configuration(section, PlainSettings)
        assert result.host == "myhost"
        assert result.port == 0
        assert result.enabled is False

    def test_all_fields_missing(self):
        config = _build_config({})
        section = config.get_section("App")
        result = bind_configuration(section, PlainSettings)
        assert result.host == ""
        assert result.port == 0
        assert result.enabled is False


# ------------------------------------------------------------------
# _coerce
# ------------------------------------------------------------------


from application_builder import _coerce


class TestCoerce:
    def test_str_passthrough(self):
        assert _coerce("hello", str) == "hello"

    def test_int_conversion(self):
        assert _coerce("42", int) == 42

    def test_float_conversion(self):
        assert _coerce("3.14", float) == 3.14

    def test_bool_true_values(self):
        for val in ("true", "True", "TRUE", "yes", "Yes", "1", "on", "On"):
            assert _coerce(val, bool) is True

    def test_bool_false_values(self):
        for val in ("false", "False", "FALSE", "no", "No", "0", "off", "Off"):
            assert _coerce(val, bool) is False

    def test_list_from_json(self):
        result = _coerce('[1, 2, 3]', List[str])
        assert result == [1, 2, 3]

    def test_list_from_comma_separated(self):
        result = _coerce("a, b, c", List[str])
        assert result == ["a", "b", "c"]

    def test_dict_from_json(self):
        result = _coerce('{"k": "v"}', Dict[str, str])
        assert result == {"k": "v"}

    def test_unknown_type_returns_raw(self):
        class Custom:
            pass

        assert _coerce("raw_value", Custom) == "raw_value"


# ------------------------------------------------------------------
# _OptionsImpl — singleton caching
# ------------------------------------------------------------------


@dataclass
class CacheTestSettings:
    value: str = ""


class TestOptionsImpl:
    def test_get_value_returns_bound_options(self):
        config = _build_config({"Sec:value": "hello"})
        opts = _OptionsImpl(config, "Sec", CacheTestSettings)
        result = opts.get_value()
        assert isinstance(result, CacheTestSettings)
        assert result.value == "hello"

    def test_get_value_is_cached(self):
        config = _build_config({"Sec:value": "original"})
        opts = _OptionsImpl(config, "Sec", CacheTestSettings)
        first = opts.get_value()
        config._data["Sec:value"] = "changed"
        second = opts.get_value()
        assert first is second
        assert second.value == "original"

    def test_thread_safe_double_checked_locking(self):
        config = _build_config({"Sec:value": "threaded"})
        opts = _OptionsImpl(config, "Sec", CacheTestSettings)
        results = []
        barrier = threading.Barrier(10)

        def worker():
            barrier.wait()
            results.append(opts.get_value())

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        assert all(r is results[0] for r in results)


# ------------------------------------------------------------------
# _OptionsSnapshotImpl — scoped, rebinds each call
# ------------------------------------------------------------------


class TestOptionsSnapshotImpl:
    def test_get_value_returns_bound_options(self):
        config = _build_config({"Sec:value": "snap"})
        snap = _OptionsSnapshotImpl(config, "Sec", CacheTestSettings)
        result = snap.get_value()
        assert isinstance(result, CacheTestSettings)
        assert result.value == "snap"

    def test_get_value_rebinds_on_each_call(self):
        config = _build_config({"Sec:value": "v1"})
        snap = _OptionsSnapshotImpl(config, "Sec", CacheTestSettings)
        first = snap.get_value()
        assert first.value == "v1"

        config._data["Sec:value"] = "v2"
        second = snap.get_value()
        assert second.value == "v2"
        assert first is not second


# ------------------------------------------------------------------
# _OptionsMonitorImpl — latest config + change callbacks
# ------------------------------------------------------------------


class TestOptionsMonitorImpl:
    def test_get_current_value_reflects_latest_config(self):
        config = _build_config({"Sec:value": "initial"})
        monitor = _OptionsMonitorImpl(config, "Sec", CacheTestSettings)
        assert monitor.get_current_value().value == "initial"

        config._data["Sec:value"] = "updated"
        assert monitor.get_current_value().value == "updated"

    def test_on_change_registers_callback(self):
        config = _build_config({"Sec:value": "start"})
        monitor = _OptionsMonitorImpl(config, "Sec", CacheTestSettings)
        received = []
        monitor.on_change(lambda v: received.append(v))
        monitor._notify_change()
        assert len(received) == 1
        assert isinstance(received[0], CacheTestSettings)

    def test_notify_change_fires_with_current_value(self):
        config = _build_config({"Sec:value": "before"})
        monitor = _OptionsMonitorImpl(config, "Sec", CacheTestSettings)
        received = []
        monitor.on_change(lambda v: received.append(v.value))
        config._data["Sec:value"] = "after"
        monitor._notify_change()
        assert received == ["after"]

    def test_multiple_callbacks_all_fire(self):
        config = _build_config({"Sec:value": "x"})
        monitor = _OptionsMonitorImpl(config, "Sec", CacheTestSettings)
        results_a = []
        results_b = []
        monitor.on_change(lambda v: results_a.append(v.value))
        monitor.on_change(lambda v: results_b.append(v.value))
        monitor._notify_change()
        assert len(results_a) == 1
        assert len(results_b) == 1

    def test_unregister_via_dispose(self):
        config = _build_config({"Sec:value": "x"})
        monitor = _OptionsMonitorImpl(config, "Sec", CacheTestSettings)
        received = []
        reg = monitor.on_change(lambda v: received.append(v.value))
        monitor._notify_change()
        assert len(received) == 1

        reg.dispose()
        monitor._notify_change()
        assert len(received) == 1

    def test_callback_exception_does_not_break_others(self):
        config = _build_config({"Sec:value": "x"})
        monitor = _OptionsMonitorImpl(config, "Sec", CacheTestSettings)
        results = []

        def bad_callback(v):
            raise RuntimeError("boom")

        monitor.on_change(bad_callback)
        monitor.on_change(lambda v: results.append(v.value))
        monitor._notify_change()
        assert len(results) == 1


# ------------------------------------------------------------------
# ApplicationBuilder.configure_options integration
# ------------------------------------------------------------------


@dataclass
class AppFeatureSettings:
    enabled: bool = False
    max_retries: int = 1


class TestConfigureOptionsIntegration:
    def test_registers_ioptions(self):
        builder = ApplicationBuilder()
        builder.add_in_memory_configuration({
            "Feature:enabled": "true",
            "Feature:max_retries": "5",
        })
        builder.configure_options(AppFeatureSettings, "Feature")
        provider = builder.build(auto_start_hosted_services=False)
        opts = provider.get_required_service(IOptions)
        value = opts.get_value()
        assert isinstance(value, AppFeatureSettings)
        assert value.enabled is True
        assert value.max_retries == 5

    def test_registers_ioptions_snapshot(self):
        builder = ApplicationBuilder()
        builder.add_in_memory_configuration({
            "Feature:enabled": "false",
        })
        builder.configure_options(AppFeatureSettings, "Feature")
        provider = builder.build(auto_start_hosted_services=False)
        snap = provider.get_required_service(IOptionsSnapshot)
        value = snap.get_value()
        assert isinstance(value, AppFeatureSettings)
        assert value.enabled is False

    def test_registers_ioptions_monitor(self):
        builder = ApplicationBuilder()
        builder.add_in_memory_configuration({
            "Feature:enabled": "on",
            "Feature:max_retries": "10",
        })
        builder.configure_options(AppFeatureSettings, "Feature")
        provider = builder.build(auto_start_hosted_services=False)
        monitor = provider.get_required_service(IOptionsMonitor)
        value = monitor.get_current_value()
        assert isinstance(value, AppFeatureSettings)
        assert value.enabled is True
        assert value.max_retries == 10
