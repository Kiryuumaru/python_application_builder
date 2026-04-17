from abc import ABC, abstractmethod
from typing import List

import pytest

from application_builder import (
    ApplicationBuilder,
    ServiceProvider,
    ServiceDescriptor,
    ServiceLifetime,
    ServiceScope,
    ScopeFactory,
    IDisposable,
    ILogger,
)


# ---------------------------------------------------------------------------
# Test abstractions and implementations
# ---------------------------------------------------------------------------

class IGreeter(ABC):
    @abstractmethod
    def greet(self) -> str:
        pass


class IFarewell(ABC):
    @abstractmethod
    def farewell(self) -> str:
        pass


class IRepository(ABC):
    @abstractmethod
    def name(self) -> str:
        pass


class SimpleGreeter(IGreeter):
    def greet(self) -> str:
        return "hello"


class FancyGreeter(IGreeter):
    def greet(self) -> str:
        return "greetings"


class SimpleFarewell(IFarewell):
    def farewell(self) -> str:
        return "goodbye"


class RepositoryA(IRepository):
    def name(self) -> str:
        return "A"


class RepositoryB(IRepository):
    def name(self) -> str:
        return "B"


class ServiceWithDependency:
    def __init__(self, greeter: IGreeter):
        self.greeter = greeter


class ServiceWithMultipleDeps:
    def __init__(self, greeter: IGreeter, farewell: IFarewell):
        self.greeter = greeter
        self.farewell = farewell


class ServiceWithListDep:
    def __init__(self, repos: List[IRepository]):
        self.repos = repos


class ServiceWithUnresolvable:
    """Has a required parameter whose type is not registered."""

    def __init__(self, missing: IRepository):
        self.missing = missing


class DisposableService(IFarewell, IDisposable):
    def __init__(self):
        self.disposed = False
        self.dispose_count = 0

    def farewell(self) -> str:
        return "disposed" if self.disposed else "alive"

    def dispose(self) -> None:
        self.disposed = True
        self.dispose_count += 1


class NonDisposableScoped:
    """Scoped service that does NOT implement IDisposable."""
    pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSingletonLifetime:

    def test_same_instance_returned_across_calls(self):
        app = ApplicationBuilder()
        app.add_singleton(IGreeter, SimpleGreeter)
        provider = app.build()

        first = provider.get_required_service(IGreeter)
        second = provider.get_required_service(IGreeter)

        assert first is second

    def test_same_instance_across_scopes(self):
        app = ApplicationBuilder()
        app.add_singleton(IGreeter, SimpleGreeter)
        provider = app.build()

        scope1 = provider.create_scope()
        scope2 = provider.create_scope()

        s1 = scope1.get_required_service(IGreeter)
        s2 = scope2.get_required_service(IGreeter)

        assert s1 is s2

    def test_factory_based_singleton(self):
        call_count = 0

        def factory(sp):
            nonlocal call_count
            call_count += 1
            return SimpleGreeter()

        app = ApplicationBuilder()
        app.add_singleton_factory(IGreeter, factory)
        provider = app.build()

        first = provider.get_required_service(IGreeter)
        second = provider.get_required_service(IGreeter)

        assert first is second
        assert call_count == 1


class TestScopedLifetime:

    def test_same_instance_within_scope(self):
        app = ApplicationBuilder()
        app.add_scoped(IGreeter, SimpleGreeter)
        provider = app.build()

        scope = provider.create_scope()
        first = scope.get_required_service(IGreeter)
        second = scope.get_required_service(IGreeter)

        assert first is second

    def test_different_instances_across_scopes(self):
        app = ApplicationBuilder()
        app.add_scoped(IGreeter, SimpleGreeter)
        provider = app.build()

        scope1 = provider.create_scope()
        scope2 = provider.create_scope()

        s1 = scope1.get_required_service(IGreeter)
        s2 = scope2.get_required_service(IGreeter)

        assert s1 is not s2

    def test_factory_based_scoped(self):
        call_count = 0

        def factory(sp):
            nonlocal call_count
            call_count += 1
            return SimpleGreeter()

        app = ApplicationBuilder()
        app.add_scoped_factory(IGreeter, factory)
        provider = app.build()

        scope = provider.create_scope()
        first = scope.get_required_service(IGreeter)
        second = scope.get_required_service(IGreeter)

        assert first is second
        assert call_count == 1


class TestTransientLifetime:

    def test_different_instance_each_call(self):
        app = ApplicationBuilder()
        app.add_transient(IGreeter, SimpleGreeter)
        provider = app.build()

        first = provider.get_required_service(IGreeter)
        second = provider.get_required_service(IGreeter)

        assert first is not second

    def test_factory_based_transient(self):
        call_count = 0

        def factory(sp):
            nonlocal call_count
            call_count += 1
            return SimpleGreeter()

        app = ApplicationBuilder()
        app.add_transient_factory(IGreeter, factory)
        provider = app.build()

        provider.get_required_service(IGreeter)
        provider.get_required_service(IGreeter)

        assert call_count == 2


class TestInstanceRegistration:

    def test_returns_exact_instance(self):
        instance = SimpleGreeter()
        app = ApplicationBuilder()
        app.add_singleton_instance(IGreeter, instance)
        provider = app.build()

        resolved = provider.get_required_service(IGreeter)

        assert resolved is instance


class TestConstructorInjection:

    def test_resolves_single_dependency(self):
        app = ApplicationBuilder()
        app.add_singleton(IGreeter, SimpleGreeter)
        app.add_singleton(ServiceWithDependency)
        provider = app.build()

        svc = provider.get_required_service(ServiceWithDependency)

        assert isinstance(svc.greeter, SimpleGreeter)

    def test_resolves_multiple_dependencies(self):
        app = ApplicationBuilder()
        app.add_singleton(IGreeter, SimpleGreeter)
        app.add_singleton(IFarewell, SimpleFarewell)
        app.add_singleton(ServiceWithMultipleDeps)
        provider = app.build()

        svc = provider.get_required_service(ServiceWithMultipleDeps)

        assert svc.greeter.greet() == "hello"
        assert svc.farewell.farewell() == "goodbye"

    def test_resolves_list_dependency(self):
        app = ApplicationBuilder()
        app.add_singleton(IRepository, RepositoryA)
        app.add_singleton(IRepository, RepositoryB)
        app.add_singleton(ServiceWithListDep)
        provider = app.build()

        svc = provider.get_required_service(ServiceWithListDep)

        names = [r.name() for r in svc.repos]
        assert "A" in names
        assert "B" in names

    def test_raises_for_unresolvable_required_param(self):
        app = ApplicationBuilder()
        app.add_singleton(ServiceWithUnresolvable)
        app.set_validate_on_build(True)

        with pytest.raises(ValueError):
            app.build()


class TestGetService:

    def test_returns_none_for_unregistered_type(self):
        app = ApplicationBuilder()
        provider = app.build()

        result = provider.get_service(IGreeter)

        assert result is None


class TestGetRequiredService:

    def test_raises_for_unregistered_type(self):
        app = ApplicationBuilder()
        provider = app.build()

        with pytest.raises(KeyError):
            provider.get_required_service(IGreeter)


class TestGetServices:

    def test_returns_all_registered_implementations(self):
        app = ApplicationBuilder()
        app.add_singleton(IRepository, RepositoryA)
        app.add_singleton(IRepository, RepositoryB)
        provider = app.build()

        services = provider.get_services(IRepository)

        assert len(services) == 2
        names = {s.name() for s in services}
        assert names == {"A", "B"}

    def test_returns_empty_list_for_unregistered(self):
        app = ApplicationBuilder()
        provider = app.build()

        services = provider.get_services(IGreeter)

        assert services == []


class TestKeyedServices:

    def test_keyed_singleton_resolves_by_key(self):
        app = ApplicationBuilder()
        app.add_keyed_singleton(IGreeter, "simple", SimpleGreeter)
        provider = app.build()

        svc = provider.get_keyed_service(IGreeter, "simple")

        assert isinstance(svc, SimpleGreeter)

    def test_different_keys_return_different_instances(self):
        app = ApplicationBuilder()
        app.add_keyed_singleton(IGreeter, "simple", SimpleGreeter)
        app.add_keyed_singleton(IGreeter, "fancy", FancyGreeter)
        provider = app.build()

        simple = provider.get_keyed_service(IGreeter, "simple")
        fancy = provider.get_keyed_service(IGreeter, "fancy")

        assert isinstance(simple, SimpleGreeter)
        assert isinstance(fancy, FancyGreeter)
        assert simple is not fancy

    def test_get_required_keyed_service_raises_for_missing_key(self):
        app = ApplicationBuilder()
        provider = app.build()

        with pytest.raises(KeyError):
            provider.get_required_keyed_service(IGreeter, "nope")


class TestDecoratorSupport:

    def test_decorate_wraps_resolved_instance(self):
        class GreeterDecorator(IGreeter):
            def __init__(self, inner: IGreeter):
                self.inner = inner

            def greet(self) -> str:
                return f"decorated({self.inner.greet()})"

        app = ApplicationBuilder()
        app.add_singleton(IGreeter, SimpleGreeter)
        app.decorate(IGreeter, lambda sp, inner: GreeterDecorator(inner))
        provider = app.build()

        svc = provider.get_required_service(IGreeter)

        assert svc.greet() == "decorated(hello)"

    def test_multiple_decorators_chain(self):
        class PrefixDecorator(IGreeter):
            def __init__(self, inner: IGreeter):
                self.inner = inner

            def greet(self) -> str:
                return f"prefix-{self.inner.greet()}"

        class SuffixDecorator(IGreeter):
            def __init__(self, inner: IGreeter):
                self.inner = inner

            def greet(self) -> str:
                return f"{self.inner.greet()}-suffix"

        app = ApplicationBuilder()
        app.add_singleton(IGreeter, SimpleGreeter)
        app.decorate(IGreeter, lambda sp, inner: PrefixDecorator(inner))
        app.decorate(IGreeter, lambda sp, inner: SuffixDecorator(inner))
        provider = app.build()

        svc = provider.get_required_service(IGreeter)

        assert svc.greet() == "prefix-hello-suffix"


class TestTryAdd:

    def test_skips_if_already_registered(self):
        app = ApplicationBuilder()
        app.add_singleton(IGreeter, SimpleGreeter)
        app.try_add_singleton(IGreeter, FancyGreeter)
        provider = app.build()

        svc = provider.get_required_service(IGreeter)

        assert isinstance(svc, SimpleGreeter)

    def test_registers_if_not_present(self):
        app = ApplicationBuilder()
        app.try_add_singleton(IGreeter, SimpleGreeter)
        provider = app.build()

        svc = provider.get_required_service(IGreeter)

        assert isinstance(svc, SimpleGreeter)


class TestReplace:

    def test_replaces_first_matching_registration(self):
        app = ApplicationBuilder()
        app.add_singleton(IGreeter, SimpleGreeter)
        app.replace(ServiceDescriptor(
            service_type=IGreeter,
            implementation_type=FancyGreeter,
            lifetime=ServiceLifetime.SINGLETON,
        ))
        provider = app.build()

        svc = provider.get_required_service(IGreeter)

        assert isinstance(svc, FancyGreeter)

    def test_adds_if_none_exists(self):
        app = ApplicationBuilder()
        app.replace(ServiceDescriptor(
            service_type=IGreeter,
            implementation_type=SimpleGreeter,
            lifetime=ServiceLifetime.SINGLETON,
        ))
        provider = app.build()

        svc = provider.get_required_service(IGreeter)

        assert isinstance(svc, SimpleGreeter)


class TestRemoveAll:

    def test_removes_all_registrations_for_type(self):
        app = ApplicationBuilder()
        app.add_singleton(IGreeter, SimpleGreeter)
        app.add_singleton(IGreeter, FancyGreeter)
        app.remove_all(IGreeter)
        provider = app.build()

        result = provider.get_service(IGreeter)

        assert result is None


class TestScopeValidation:

    def test_raises_for_scoped_from_root_when_enabled(self):
        app = ApplicationBuilder()
        app.add_scoped(IGreeter, SimpleGreeter)
        app.set_validate_scopes(True)
        provider = app.build()

        with pytest.raises(RuntimeError):
            provider.get_service(IGreeter)

    def test_scoped_resolves_fine_from_scope(self):
        app = ApplicationBuilder()
        app.add_scoped(IGreeter, SimpleGreeter)
        app.set_validate_scopes(True)
        provider = app.build()

        scope = provider.create_scope()
        svc = scope.get_required_service(IGreeter)

        assert isinstance(svc, SimpleGreeter)


class TestValidateOnBuild:

    def test_raises_for_unresolvable_dependencies(self):
        app = ApplicationBuilder()
        app.add_singleton(ServiceWithUnresolvable)
        app.set_validate_on_build(True)

        with pytest.raises(ValueError):
            app.build()


class TestScopeDisposal:

    def test_dispose_calls_idisposable_on_scoped_instances(self):
        app = ApplicationBuilder()
        app.add_scoped(IFarewell, DisposableService)
        provider = app.build()

        scope = provider.create_scope()
        svc = scope.get_required_service(IFarewell)
        assert not svc.disposed

        scope.dispose()
        assert svc.disposed

    def test_double_dispose_is_safe(self):
        app = ApplicationBuilder()
        app.add_scoped(IFarewell, DisposableService)
        provider = app.build()

        scope = provider.create_scope()
        svc = scope.get_required_service(IFarewell)

        scope.dispose()
        scope.dispose()

        assert svc.dispose_count == 1


class TestScopeFactoryContextManager:

    def test_context_manager_disposes_on_exit(self):
        app = ApplicationBuilder()
        app.add_scoped(IFarewell, DisposableService)
        provider = app.build()

        scope_factory = provider.get_required_service(ScopeFactory)

        with scope_factory.create_scope_context() as scope:
            svc = scope.get_required_service(IFarewell)
            assert not svc.disposed

        assert svc.disposed
