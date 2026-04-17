import os
import threading

import pytest

from application_builder import (
    HostEnvironment,
    IHostEnvironment,
    HostApplicationLifetime,
    IHostApplicationLifetime,
    ApplicationBuilder,
    CancellationToken,
)


# ---------------------------------------------------------------------------
# HostEnvironment
# ---------------------------------------------------------------------------
class TestHostEnvironmentDefaults:
    def test_default_environment_name_is_production(self):
        env = HostEnvironment()
        assert env.environment_name == "Production"

    def test_default_application_name(self):
        env = HostEnvironment()
        assert env.application_name == "Application"

    def test_default_content_root_path_is_cwd(self):
        env = HostEnvironment()
        assert env.content_root_path == os.getcwd()

    def test_implements_interface(self):
        env = HostEnvironment()
        assert isinstance(env, IHostEnvironment)


class TestHostEnvironmentCustomValues:
    def test_custom_environment_name(self):
        env = HostEnvironment(environment_name="Development")
        assert env.environment_name == "Development"

    def test_custom_application_name(self):
        env = HostEnvironment(application_name="MyApp")
        assert env.application_name == "MyApp"

    def test_custom_content_root_path(self):
        env = HostEnvironment(content_root_path="/custom/path")
        assert env.content_root_path == "/custom/path"

    def test_all_custom_values(self):
        env = HostEnvironment(
            environment_name="Staging",
            application_name="TestApp",
            content_root_path="/app/root",
        )
        assert env.environment_name == "Staging"
        assert env.application_name == "TestApp"
        assert env.content_root_path == "/app/root"


class TestHostEnvironmentChecks:
    def test_is_development_true(self):
        env = HostEnvironment(environment_name="Development")
        assert env.is_development() is True

    def test_is_development_case_insensitive(self):
        env = HostEnvironment(environment_name="DEVELOPMENT")
        assert env.is_development() is True

    def test_is_development_mixed_case(self):
        env = HostEnvironment(environment_name="dEvElOpMeNt")
        assert env.is_development() is True

    def test_is_development_false_for_production(self):
        env = HostEnvironment(environment_name="Production")
        assert env.is_development() is False

    def test_is_staging_true(self):
        env = HostEnvironment(environment_name="Staging")
        assert env.is_staging() is True

    def test_is_staging_case_insensitive(self):
        env = HostEnvironment(environment_name="STAGING")
        assert env.is_staging() is True

    def test_is_staging_false_for_other(self):
        env = HostEnvironment(environment_name="Development")
        assert env.is_staging() is False

    def test_is_production_true(self):
        env = HostEnvironment(environment_name="Production")
        assert env.is_production() is True

    def test_is_production_case_insensitive(self):
        env = HostEnvironment(environment_name="PRODUCTION")
        assert env.is_production() is True

    def test_is_production_false_for_custom(self):
        env = HostEnvironment(environment_name="Custom")
        assert env.is_production() is False

    def test_all_checks_false_for_unknown_environment(self):
        env = HostEnvironment(environment_name="Testing")
        assert env.is_development() is False
        assert env.is_staging() is False
        assert env.is_production() is False


# ---------------------------------------------------------------------------
# HostApplicationLifetime
# ---------------------------------------------------------------------------
class TestHostApplicationLifetimeInitialState:
    def test_application_started_not_cancelled(self):
        lifetime = HostApplicationLifetime()
        assert lifetime.application_started.is_cancellation_requested is False

    def test_application_stopping_not_cancelled(self):
        lifetime = HostApplicationLifetime()
        assert lifetime.application_stopping.is_cancellation_requested is False

    def test_application_stopped_not_cancelled(self):
        lifetime = HostApplicationLifetime()
        assert lifetime.application_stopped.is_cancellation_requested is False

    def test_tokens_are_cancellation_tokens(self):
        lifetime = HostApplicationLifetime()
        assert isinstance(lifetime.application_started, CancellationToken)
        assert isinstance(lifetime.application_stopping, CancellationToken)
        assert isinstance(lifetime.application_stopped, CancellationToken)

    def test_implements_interface(self):
        lifetime = HostApplicationLifetime()
        assert isinstance(lifetime, IHostApplicationLifetime)


class TestHostApplicationLifetimeNotifyStarted:
    def test_notify_started_cancels_started_token(self):
        lifetime = HostApplicationLifetime()
        lifetime.notify_started()
        assert lifetime.application_started.is_cancellation_requested is True

    def test_notify_started_does_not_affect_stopping(self):
        lifetime = HostApplicationLifetime()
        lifetime.notify_started()
        assert lifetime.application_stopping.is_cancellation_requested is False

    def test_notify_started_does_not_affect_stopped(self):
        lifetime = HostApplicationLifetime()
        lifetime.notify_started()
        assert lifetime.application_stopped.is_cancellation_requested is False


class TestHostApplicationLifetimeNotifyStopping:
    def test_notify_stopping_cancels_stopping_token(self):
        lifetime = HostApplicationLifetime()
        lifetime.notify_stopping()
        assert lifetime.application_stopping.is_cancellation_requested is True

    def test_notify_stopping_does_not_affect_started(self):
        lifetime = HostApplicationLifetime()
        lifetime.notify_stopping()
        assert lifetime.application_started.is_cancellation_requested is False

    def test_notify_stopping_does_not_affect_stopped(self):
        lifetime = HostApplicationLifetime()
        lifetime.notify_stopping()
        assert lifetime.application_stopped.is_cancellation_requested is False


class TestHostApplicationLifetimeNotifyStopped:
    def test_notify_stopped_cancels_stopped_token(self):
        lifetime = HostApplicationLifetime()
        lifetime.notify_stopped()
        assert lifetime.application_stopped.is_cancellation_requested is True

    def test_notify_stopped_does_not_affect_started(self):
        lifetime = HostApplicationLifetime()
        lifetime.notify_stopped()
        assert lifetime.application_started.is_cancellation_requested is False

    def test_notify_stopped_does_not_affect_stopping(self):
        lifetime = HostApplicationLifetime()
        lifetime.notify_stopped()
        assert lifetime.application_stopping.is_cancellation_requested is False


class TestHostApplicationLifetimeStopApplication:
    def test_stop_application_triggers_stopping(self):
        lifetime = HostApplicationLifetime()
        lifetime.stop_application()
        assert lifetime.application_stopping.is_cancellation_requested is True

    def test_stop_application_does_not_trigger_started(self):
        lifetime = HostApplicationLifetime()
        lifetime.stop_application()
        assert lifetime.application_started.is_cancellation_requested is False

    def test_stop_application_does_not_trigger_stopped(self):
        lifetime = HostApplicationLifetime()
        lifetime.stop_application()
        assert lifetime.application_stopped.is_cancellation_requested is False


class TestHostApplicationLifetimeCallbacks:
    def test_callback_fires_on_started(self):
        lifetime = HostApplicationLifetime()
        called = threading.Event()
        lifetime.application_started.register(called.set)
        assert not called.is_set()
        lifetime.notify_started()
        assert called.is_set()

    def test_callback_fires_on_stopping(self):
        lifetime = HostApplicationLifetime()
        called = threading.Event()
        lifetime.application_stopping.register(called.set)
        lifetime.notify_stopping()
        assert called.is_set()

    def test_callback_fires_on_stopped(self):
        lifetime = HostApplicationLifetime()
        called = threading.Event()
        lifetime.application_stopped.register(called.set)
        lifetime.notify_stopped()
        assert called.is_set()

    def test_multiple_callbacks_fire_on_started(self):
        lifetime = HostApplicationLifetime()
        results = []
        lifetime.application_started.register(lambda: results.append("a"))
        lifetime.application_started.register(lambda: results.append("b"))
        lifetime.notify_started()
        assert results == ["a", "b"]


class TestHostApplicationLifetimeFullLifecycle:
    def test_full_lifecycle_sequence(self):
        lifetime = HostApplicationLifetime()
        events = []

        lifetime.application_started.register(lambda: events.append("started"))
        lifetime.application_stopping.register(lambda: events.append("stopping"))
        lifetime.application_stopped.register(lambda: events.append("stopped"))

        lifetime.notify_started()
        assert events == ["started"]

        lifetime.notify_stopping()
        assert events == ["started", "stopping"]

        lifetime.notify_stopped()
        assert events == ["started", "stopping", "stopped"]

    def test_tokens_reflect_state_after_full_lifecycle(self):
        lifetime = HostApplicationLifetime()
        lifetime.notify_started()
        lifetime.notify_stopping()
        lifetime.notify_stopped()

        assert lifetime.application_started.is_cancellation_requested is True
        assert lifetime.application_stopping.is_cancellation_requested is True
        assert lifetime.application_stopped.is_cancellation_requested is True

    def test_stop_application_within_lifecycle(self):
        lifetime = HostApplicationLifetime()
        events = []

        lifetime.application_started.register(lambda: events.append("started"))
        lifetime.application_stopping.register(lambda: events.append("stopping"))
        lifetime.application_stopped.register(lambda: events.append("stopped"))

        lifetime.notify_started()
        lifetime.stop_application()
        lifetime.notify_stopped()

        assert events == ["started", "stopping", "stopped"]


# ---------------------------------------------------------------------------
# ApplicationBuilder Integration
# ---------------------------------------------------------------------------
class TestApplicationBuilderHostIntegration:
    def test_builder_registers_host_environment(self):
        builder = ApplicationBuilder()
        provider = builder.build()
        env = provider.get_service(IHostEnvironment)
        assert env is not None
        assert isinstance(env, HostEnvironment)

    def test_builder_registers_host_application_lifetime(self):
        builder = ApplicationBuilder()
        provider = builder.build()
        lifetime = provider.get_service(IHostApplicationLifetime)
        assert lifetime is not None
        assert isinstance(lifetime, HostApplicationLifetime)

    def test_builder_resolves_same_lifetime_by_interface_and_concrete(self):
        builder = ApplicationBuilder()
        provider = builder.build()
        by_interface = provider.get_service(IHostApplicationLifetime)
        by_concrete = provider.get_service(HostApplicationLifetime)
        assert by_interface is by_concrete

    def test_builder_host_environment_defaults(self):
        builder = ApplicationBuilder()
        provider = builder.build()
        env = provider.get_service(IHostEnvironment)
        assert env.environment_name == "Production"
        assert env.application_name == "Application"

    def test_builder_environment_from_config(self):
        builder = ApplicationBuilder()
        builder.add_configuration_dictionary({"Environment": "Development"})
        provider = builder.build()
        env = provider.get_service(IHostEnvironment)
        assert env.environment_name == "Development"

    def test_builder_application_name_from_config(self):
        builder = ApplicationBuilder()
        builder.add_configuration_dictionary({"ApplicationName": "CustomApp"})
        provider = builder.build()
        env = provider.get_service(IHostEnvironment)
        assert env.application_name == "CustomApp"

    def test_builder_content_root_from_config(self):
        builder = ApplicationBuilder()
        builder.add_configuration_dictionary({"ContentRoot": "/my/root"})
        provider = builder.build()
        env = provider.get_service(IHostEnvironment)
        assert env.content_root_path == "/my/root"
