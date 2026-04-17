import threading
from abc import ABC, abstractmethod

import pytest

from application_builder import (
    ApplicationBuilder,
    ServiceProvider,
    ServiceDescriptor,
    ServiceLifetime,
    IConfiguration,
    ILogger,
    JobManager,
    IHostEnvironment,
    IHostApplicationLifetime,
    MiddlewarePipeline,
    ScopeFactory,
    IWorker,
    Worker,
)


# ---------------------------------------------------------------------------
# Test abstractions and implementations
# ---------------------------------------------------------------------------

class IAnimal(ABC):
    @abstractmethod
    def speak(self) -> str:
        pass


class Dog(IAnimal):
    def speak(self) -> str:
        return "woof"


class Cat(IAnimal):
    def speak(self) -> str:
        return "meow"


class IEngine(ABC):
    @abstractmethod
    def name(self) -> str:
        pass


class V8Engine(IEngine):
    def name(self) -> str:
        return "V8"


class ElectricEngine(IEngine):
    def name(self) -> str:
        return "Electric"


class IUnresolvable(ABC):
    @abstractmethod
    def value(self) -> str:
        pass


class BrokenService(IUnresolvable):
    def __init__(self, missing_dep: IEngine):
        self.missing_dep = missing_dep

    def value(self) -> str:
        return "broken"


class TrackingWorker(Worker):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()

    def execute(self):
        self.started.set()
        self.wait_for_stop(timeout_seconds=0.1)


class SecondWorker(Worker):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()

    def execute(self):
        self.started.set()
        self.wait_for_stop(timeout_seconds=0.1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestChaining:
    """All registration methods return self for fluent chaining."""

    def test_add_singleton_returns_self(self):
        app = ApplicationBuilder()
        result = app.add_singleton(IAnimal, Dog)
        assert result is app

    def test_add_singleton_instance_returns_self(self):
        app = ApplicationBuilder()
        result = app.add_singleton_instance(IAnimal, Dog())
        assert result is app

    def test_add_singleton_factory_returns_self(self):
        app = ApplicationBuilder()
        result = app.add_singleton_factory(IAnimal, lambda sp: Dog())
        assert result is app

    def test_add_scoped_returns_self(self):
        app = ApplicationBuilder()
        result = app.add_scoped(IAnimal, Dog)
        assert result is app

    def test_add_transient_returns_self(self):
        app = ApplicationBuilder()
        result = app.add_transient(IAnimal, Dog)
        assert result is app

    def test_try_add_singleton_returns_self(self):
        app = ApplicationBuilder()
        result = app.try_add_singleton(IAnimal, Dog)
        assert result is app

    def test_try_add_scoped_returns_self(self):
        app = ApplicationBuilder()
        result = app.try_add_scoped(IAnimal, Dog)
        assert result is app

    def test_try_add_transient_returns_self(self):
        app = ApplicationBuilder()
        result = app.try_add_transient(IAnimal, Dog)
        assert result is app

    def test_replace_returns_self(self):
        app = ApplicationBuilder()
        descriptor = ServiceDescriptor(
            service_type=IAnimal,
            implementation_type=Dog,
            lifetime=ServiceLifetime.SINGLETON,
        )
        result = app.replace(descriptor)
        assert result is app

    def test_remove_all_returns_self(self):
        app = ApplicationBuilder()
        result = app.remove_all(IAnimal)
        assert result is app

    def test_add_keyed_singleton_returns_self(self):
        app = ApplicationBuilder()
        result = app.add_keyed_singleton(IAnimal, "k1", Dog)
        assert result is app

    def test_add_keyed_scoped_returns_self(self):
        app = ApplicationBuilder()
        result = app.add_keyed_scoped(IAnimal, "k1", Dog)
        assert result is app

    def test_add_keyed_transient_returns_self(self):
        app = ApplicationBuilder()
        result = app.add_keyed_transient(IAnimal, "k1", Dog)
        assert result is app

    def test_add_keyed_singleton_factory_returns_self(self):
        app = ApplicationBuilder()
        result = app.add_keyed_singleton_factory(IAnimal, "k1", lambda sp: Dog())
        assert result is app

    def test_decorate_returns_self(self):
        app = ApplicationBuilder()
        result = app.decorate(IAnimal, lambda sp, inner: inner)
        assert result is app

    def test_add_worker_returns_self(self):
        app = ApplicationBuilder()
        result = app.add_worker(TrackingWorker)
        assert result is app

    def test_add_configuration_dictionary_returns_self(self):
        app = ApplicationBuilder()
        result = app.add_configuration_dictionary({"key": "value"})
        assert result is app

    def test_set_validate_on_build_returns_self(self):
        app = ApplicationBuilder()
        result = app.set_validate_on_build(True)
        assert result is app

    def test_set_validate_scopes_returns_self(self):
        app = ApplicationBuilder()
        result = app.set_validate_scopes(True)
        assert result is app

    def test_fluent_chain_multiple_calls(self):
        app = ApplicationBuilder()
        result = (
            app
            .add_singleton(IAnimal, Dog)
            .add_transient(IEngine, V8Engine)
            .add_configuration_dictionary({"x": "1"})
            .set_validate_on_build(False)
        )
        assert result is app


class TestBuildDefaults:
    """build() auto-registers core infrastructure services."""

    def test_iconfiguration_available(self):
        provider = ApplicationBuilder().build(auto_start_hosted_services=False)
        config = provider.get_service(IConfiguration)
        assert config is not None

    def test_ilogger_available(self):
        provider = ApplicationBuilder().build(auto_start_hosted_services=False)
        logger = provider.get_service(ILogger)
        assert logger is not None

    def test_job_manager_available(self):
        provider = ApplicationBuilder().build(auto_start_hosted_services=False)
        jm = provider.get_service(JobManager)
        assert jm is not None

    def test_ihost_environment_available(self):
        provider = ApplicationBuilder().build(auto_start_hosted_services=False)
        env = provider.get_service(IHostEnvironment)
        assert env is not None

    def test_ihost_application_lifetime_available(self):
        provider = ApplicationBuilder().build(auto_start_hosted_services=False)
        lifetime = provider.get_service(IHostApplicationLifetime)
        assert lifetime is not None

    def test_middleware_pipeline_available(self):
        provider = ApplicationBuilder().build(auto_start_hosted_services=False)
        pipeline = provider.get_service(MiddlewarePipeline)
        assert pipeline is not None

    def test_default_environment_is_production(self):
        provider = ApplicationBuilder().build(auto_start_hosted_services=False)
        env = provider.get_required_service(IHostEnvironment)
        assert env.environment_name == "Production"


class TestAddConfigurationDictionary:
    """add_configuration_dictionary flattens nested dicts with colon-delimited keys."""

    def test_flat_dictionary(self):
        app = ApplicationBuilder()
        app.add_configuration_dictionary({"Greeting": "hello"})
        provider = app.build(auto_start_hosted_services=False)
        config = provider.get_required_service(IConfiguration)
        assert config.get("Greeting") == "hello"

    def test_nested_dictionary_flattened_with_colons(self):
        app = ApplicationBuilder()
        app.add_configuration_dictionary({
            "Database": {
                "Host": "localhost",
                "Port": 5432,
            }
        })
        provider = app.build(auto_start_hosted_services=False)
        config = provider.get_required_service(IConfiguration)
        assert config.get("Database:Host") == "localhost"
        assert config.get("Database:Port") == "5432"

    def test_deeply_nested_dictionary(self):
        app = ApplicationBuilder()
        app.add_configuration_dictionary({
            "Level1": {
                "Level2": {
                    "Level3": "deep_value"
                }
            }
        })
        provider = app.build(auto_start_hosted_services=False)
        config = provider.get_required_service(IConfiguration)
        assert config.get("Level1:Level2:Level3") == "deep_value"

    def test_non_string_values_converted_to_string(self):
        app = ApplicationBuilder()
        app.add_configuration_dictionary({"Count": 42, "Active": True})
        provider = app.build(auto_start_hosted_services=False)
        config = provider.get_required_service(IConfiguration)
        assert config.get("Count") == "42"
        assert config.get("Active") == "True"


class TestTryAdd:
    """try_add_* skips registration if type already present."""

    def test_try_add_singleton_skips_when_already_registered(self):
        app = ApplicationBuilder()
        app.add_singleton(IAnimal, Dog)
        app.try_add_singleton(IAnimal, Cat)
        provider = app.build(auto_start_hosted_services=False)
        animal = provider.get_required_service(IAnimal)
        assert animal.speak() == "woof"

    def test_try_add_singleton_registers_when_not_present(self):
        app = ApplicationBuilder()
        app.try_add_singleton(IAnimal, Dog)
        provider = app.build(auto_start_hosted_services=False)
        animal = provider.get_required_service(IAnimal)
        assert animal.speak() == "woof"

    def test_try_add_scoped_skips_when_already_registered(self):
        app = ApplicationBuilder()
        app.add_scoped(IAnimal, Dog)
        app.try_add_scoped(IAnimal, Cat)
        provider = app.build(auto_start_hosted_services=False)
        scope = provider.create_scope()
        animal = scope.get_required_service(IAnimal)
        assert animal.speak() == "woof"

    def test_try_add_scoped_registers_when_not_present(self):
        app = ApplicationBuilder()
        app.try_add_scoped(IAnimal, Dog)
        provider = app.build(auto_start_hosted_services=False)
        scope = provider.create_scope()
        animal = scope.get_required_service(IAnimal)
        assert animal.speak() == "woof"

    def test_try_add_transient_skips_when_already_registered(self):
        app = ApplicationBuilder()
        app.add_transient(IAnimal, Dog)
        app.try_add_transient(IAnimal, Cat)
        provider = app.build(auto_start_hosted_services=False)
        animal = provider.get_required_service(IAnimal)
        assert animal.speak() == "woof"

    def test_try_add_transient_registers_when_not_present(self):
        app = ApplicationBuilder()
        app.try_add_transient(IAnimal, Dog)
        provider = app.build(auto_start_hosted_services=False)
        animal = provider.get_required_service(IAnimal)
        assert animal.speak() == "woof"


class TestReplace:
    """replace() replaces existing registration; adds if none exists."""

    def test_replace_existing_registration(self):
        app = ApplicationBuilder()
        app.add_singleton(IAnimal, Dog)
        app.replace(ServiceDescriptor(
            service_type=IAnimal,
            implementation_type=Cat,
            lifetime=ServiceLifetime.SINGLETON,
        ))
        provider = app.build(auto_start_hosted_services=False)
        animal = provider.get_required_service(IAnimal)
        assert animal.speak() == "meow"

    def test_replace_adds_when_no_existing(self):
        app = ApplicationBuilder()
        app.replace(ServiceDescriptor(
            service_type=IAnimal,
            implementation_type=Dog,
            lifetime=ServiceLifetime.SINGLETON,
        ))
        provider = app.build(auto_start_hosted_services=False)
        animal = provider.get_required_service(IAnimal)
        assert animal.speak() == "woof"


class TestRemoveAll:
    """remove_all() removes all registrations for a type."""

    def test_remove_all_clears_service_type(self):
        app = ApplicationBuilder()
        app.add_singleton(IAnimal, Dog)
        app.add_singleton(IAnimal, Cat)
        app.remove_all(IAnimal)
        provider = app.build(auto_start_hosted_services=False)
        animal = provider.get_service(IAnimal)
        assert animal is None

    def test_remove_all_does_not_affect_other_types(self):
        app = ApplicationBuilder()
        app.add_singleton(IAnimal, Dog)
        app.add_singleton(IEngine, V8Engine)
        app.remove_all(IAnimal)
        provider = app.build(auto_start_hosted_services=False)
        engine = provider.get_required_service(IEngine)
        assert engine.name() == "V8"


class TestKeyedServices:
    """Keyed services resolve independently by key."""

    def test_keyed_singleton_different_keys_resolve_independently(self):
        app = ApplicationBuilder()
        app.add_keyed_singleton(IAnimal, "dog", Dog)
        app.add_keyed_singleton(IAnimal, "cat", Cat)
        provider = app.build(auto_start_hosted_services=False)
        dog = provider.get_required_keyed_service(IAnimal, "dog")
        cat = provider.get_required_keyed_service(IAnimal, "cat")
        assert dog.speak() == "woof"
        assert cat.speak() == "meow"

    def test_keyed_scoped_different_keys(self):
        app = ApplicationBuilder()
        app.add_keyed_scoped(IAnimal, "dog", Dog)
        app.add_keyed_scoped(IAnimal, "cat", Cat)
        provider = app.build(auto_start_hosted_services=False)
        scope = provider.create_scope()
        dog = scope.get_required_keyed_service(IAnimal, "dog")
        cat = scope.get_required_keyed_service(IAnimal, "cat")
        assert dog.speak() == "woof"
        assert cat.speak() == "meow"

    def test_keyed_transient_different_keys(self):
        app = ApplicationBuilder()
        app.add_keyed_transient(IAnimal, "dog", Dog)
        app.add_keyed_transient(IAnimal, "cat", Cat)
        provider = app.build(auto_start_hosted_services=False)
        dog = provider.get_required_keyed_service(IAnimal, "dog")
        cat = provider.get_required_keyed_service(IAnimal, "cat")
        assert dog.speak() == "woof"
        assert cat.speak() == "meow"

    def test_keyed_singleton_factory(self):
        app = ApplicationBuilder()
        app.add_keyed_singleton_factory(IAnimal, "factory_dog", lambda sp: Dog())
        provider = app.build(auto_start_hosted_services=False)
        dog = provider.get_required_keyed_service(IAnimal, "factory_dog")
        assert dog.speak() == "woof"

    def test_missing_keyed_service_raises(self):
        app = ApplicationBuilder()
        provider = app.build(auto_start_hosted_services=False)
        with pytest.raises(KeyError):
            provider.get_required_keyed_service(IAnimal, "nonexistent")

    def test_keyed_and_unkeyed_are_independent(self):
        app = ApplicationBuilder()
        app.add_singleton(IAnimal, Dog)
        app.add_keyed_singleton(IAnimal, "cat", Cat)
        provider = app.build(auto_start_hosted_services=False)
        unkeyed = provider.get_required_service(IAnimal)
        keyed = provider.get_required_keyed_service(IAnimal, "cat")
        assert unkeyed.speak() == "woof"
        assert keyed.speak() == "meow"


class TestDecorate:
    """decorate() wraps the resolved service."""

    def test_decorator_wraps_resolved_service(self):
        class LoudAnimal(IAnimal):
            def __init__(self, inner: IAnimal):
                self._inner = inner

            def speak(self) -> str:
                return self._inner.speak().upper()

        app = ApplicationBuilder()
        app.add_singleton(IAnimal, Dog)
        app.decorate(IAnimal, lambda sp, inner: LoudAnimal(inner))
        provider = app.build(auto_start_hosted_services=False)
        animal = provider.get_required_service(IAnimal)
        assert animal.speak() == "WOOF"

    def test_multiple_decorators_applied_in_order(self):
        app = ApplicationBuilder()
        app.add_singleton_factory(IAnimal, lambda sp: Dog())
        app.decorate(IAnimal, lambda sp, inner: type(
            "Prefixed", (IAnimal,),
            {"speak": lambda self: "prefix-" + inner.speak(),
             "__init__": lambda self: None}
        )())

        def suffix_decorator(sp, inner):
            class Suffixed(IAnimal):
                def speak(self_inner) -> str:
                    return inner.speak() + "-suffix"
            return Suffixed()

        app.decorate(IAnimal, suffix_decorator)
        provider = app.build(auto_start_hosted_services=False)
        animal = provider.get_required_service(IAnimal)
        assert animal.speak() == "prefix-woof-suffix"


class TestAddWorker:
    """add_worker registers and starts workers on build."""

    def test_worker_starts_on_build(self):
        app = ApplicationBuilder()
        app.add_worker(TrackingWorker)
        provider = app.build(auto_start_hosted_services=True)
        try:
            workers = provider.get_services(IWorker)
            tracking = [w for w in workers if isinstance(w, TrackingWorker)]
            assert len(tracking) >= 1
            tracking[0].started.wait(timeout=5)
            assert tracking[0].started.is_set()
        finally:
            provider.stop_hosted_services()

    def test_auto_start_false_workers_not_started(self):
        app = ApplicationBuilder()
        app.add_worker(TrackingWorker)
        provider = app.build(auto_start_hosted_services=False)
        workers = provider.get_services(IWorker)
        tracking = [w for w in workers if isinstance(w, TrackingWorker)]
        assert len(tracking) >= 1
        assert not tracking[0].started.is_set()


class TestValidateOnBuild:
    """set_validate_on_build raises ValueError for unresolvable dependencies."""

    def test_invalid_registration_raises_on_build(self):
        app = ApplicationBuilder()
        app.add_singleton(IUnresolvable, BrokenService)
        app.set_validate_on_build(True)
        with pytest.raises(ValueError, match="ValidateOnBuild failed"):
            app.build(auto_start_hosted_services=False)

    def test_valid_registration_does_not_raise(self):
        app = ApplicationBuilder()
        app.add_singleton(IEngine, V8Engine)
        app.add_singleton(IUnresolvable, BrokenService)
        app.set_validate_on_build(True)
        provider = app.build(auto_start_hosted_services=False)
        assert provider is not None

    def test_validate_on_build_disabled_by_default(self):
        app = ApplicationBuilder()
        app.add_singleton(IUnresolvable, BrokenService)
        provider = app.build(auto_start_hosted_services=False)
        assert provider is not None


class TestValidateScopes:
    """set_validate_scopes prevents resolving scoped services from root."""

    def test_scoped_from_root_raises_runtime_error(self):
        app = ApplicationBuilder()
        app.add_scoped(IAnimal, Dog)
        app.set_validate_scopes(True)
        provider = app.build(auto_start_hosted_services=False)
        with pytest.raises(RuntimeError, match="Cannot resolve scoped service"):
            provider.get_service(IAnimal)

    def test_scoped_from_scope_succeeds(self):
        app = ApplicationBuilder()
        app.add_scoped(IAnimal, Dog)
        app.set_validate_scopes(True)
        provider = app.build(auto_start_hosted_services=False)
        scope = provider.create_scope()
        animal = scope.get_required_service(IAnimal)
        assert animal.speak() == "woof"

    def test_scope_validation_disabled_by_default(self):
        app = ApplicationBuilder()
        app.add_scoped(IAnimal, Dog)
        provider = app.build(auto_start_hosted_services=False)
        animal = provider.get_service(IAnimal)
        assert animal is not None


class TestBuildAutoStartFalse:
    """build(auto_start_hosted_services=False) skips starting workers."""

    def test_workers_not_started_when_false(self):
        app = ApplicationBuilder()
        app.add_worker(TrackingWorker)
        provider = app.build(auto_start_hosted_services=False)
        workers = provider.get_services(IWorker)
        tracking = [w for w in workers if isinstance(w, TrackingWorker)]
        assert len(tracking) >= 1
        assert not tracking[0].started.is_set()

    def test_manual_start_after_build(self):
        app = ApplicationBuilder()
        app.add_worker(TrackingWorker)
        provider = app.build(auto_start_hosted_services=False)
        provider.start_hosted_services()
        try:
            workers = provider.get_services(IWorker)
            tracking = [w for w in workers if isinstance(w, TrackingWorker)]
            assert len(tracking) >= 1
            tracking[0].started.wait(timeout=5)
            assert tracking[0].started.is_set()
        finally:
            provider.stop_hosted_services()


class TestMultipleBuilds:
    """Can call build multiple times; each returns a fresh provider."""

    def test_separate_providers_from_multiple_builds(self):
        app = ApplicationBuilder()
        app.add_singleton(IAnimal, Dog)
        provider1 = app.build(auto_start_hosted_services=False)
        provider2 = app.build(auto_start_hosted_services=False)
        assert provider1 is not provider2

    def test_singletons_are_distinct_across_builds(self):
        app = ApplicationBuilder()
        app.add_singleton(IAnimal, Dog)
        provider1 = app.build(auto_start_hosted_services=False)
        provider2 = app.build(auto_start_hosted_services=False)
        dog1 = provider1.get_required_service(IAnimal)
        dog2 = provider2.get_required_service(IAnimal)
        assert dog1 is not dog2


class TestHostEnvironmentFromConfig:
    """IHostEnvironment reads environment name from configuration."""

    def test_environment_name_from_config_dictionary(self):
        app = ApplicationBuilder()
        app.add_configuration_dictionary({"Environment": "Development"})
        provider = app.build(auto_start_hosted_services=False)
        env = provider.get_required_service(IHostEnvironment)
        assert env.environment_name == "Development"

    def test_application_name_from_config_dictionary(self):
        app = ApplicationBuilder()
        app.add_configuration_dictionary({"ApplicationName": "MyApp"})
        provider = app.build(auto_start_hosted_services=False)
        env = provider.get_required_service(IHostEnvironment)
        assert env.application_name == "MyApp"

    def test_is_development_helper(self):
        app = ApplicationBuilder()
        app.add_configuration_dictionary({"Environment": "Development"})
        provider = app.build(auto_start_hosted_services=False)
        env = provider.get_required_service(IHostEnvironment)
        assert env.is_development() is True
        assert env.is_production() is False
