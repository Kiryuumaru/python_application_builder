import uuid
from abc import ABC, abstractmethod

import pytest

from application_builder import (
    ApplicationBuilder,
    ServiceProvider,
    ServiceScope,
    ScopeFactory,
    IDisposable,
    ServiceDescriptor,
    ServiceLifetime,
)


# ---------------------------------------------------------------------------
# Test abstractions and implementations
# ---------------------------------------------------------------------------


class ITracker(ABC):
    @abstractmethod
    def get_id(self) -> str: ...


class DisposableService(IDisposable):
    def __init__(self):
        self.disposed = False

    def dispose(self) -> None:
        self.disposed = True


class DisposableTracker(ITracker, IDisposable):
    def __init__(self):
        self._id = str(uuid.uuid4())
        self.disposed = False
        self.dispose_count = 0

    def get_id(self) -> str:
        return self._id

    def dispose(self) -> None:
        self.disposed = True
        self.dispose_count += 1


class NonDisposableService:
    def __init__(self):
        self.value = "alive"


class FailingDisposable(IDisposable):
    def __init__(self):
        self.disposed = False

    def dispose(self) -> None:
        self.disposed = True
        raise RuntimeError("dispose failed")


class ISingletonMarker(ABC):
    @abstractmethod
    def marker_id(self) -> str: ...


class SingletonService(ISingletonMarker):
    def __init__(self):
        self._id = str(uuid.uuid4())

    def marker_id(self) -> str:
        return self._id


class IScopedMarker(ABC):
    @abstractmethod
    def marker_id(self) -> str: ...


class ScopedService(IScopedMarker):
    def __init__(self):
        self._id = str(uuid.uuid4())

    def marker_id(self) -> str:
        return self._id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_provider_with(*descriptors: ServiceDescriptor) -> ServiceProvider:
    return ServiceProvider(list(descriptors))


# ---------------------------------------------------------------------------
# Tests: ServiceScope basics
# ---------------------------------------------------------------------------


class TestServiceScopeIsServiceProvider:
    def test_scope_is_instance_of_service_provider(self):
        provider = _build_provider_with()
        scope = ServiceScope(provider)

        assert isinstance(scope, ServiceProvider)

    def test_scope_shares_singleton_instances_with_root(self):
        desc = ServiceDescriptor(
            service_type=ISingletonMarker,
            implementation_type=SingletonService,
            lifetime=ServiceLifetime.SINGLETON,
        )
        provider = _build_provider_with(desc)
        singleton_from_root = provider.get_required_service(ISingletonMarker)

        scope = ServiceScope(provider)
        singleton_from_scope = scope.get_required_service(ISingletonMarker)

        assert singleton_from_root is singleton_from_scope

    def test_scope_has_own_scoped_instances(self):
        desc = ServiceDescriptor(
            service_type=IScopedMarker,
            implementation_type=ScopedService,
            lifetime=ServiceLifetime.SCOPED,
        )
        provider = _build_provider_with(desc)
        scope = ServiceScope(provider)
        scoped_instance = scope.get_required_service(IScopedMarker)

        assert scoped_instance is not None
        assert isinstance(scoped_instance, ScopedService)

    def test_scope_is_scope_flag_is_true(self):
        provider = _build_provider_with()
        scope = ServiceScope(provider)

        assert scope._is_scope is True


# ---------------------------------------------------------------------------
# Tests: Scope disposal
# ---------------------------------------------------------------------------


class TestScopeDisposal:
    def test_dispose_calls_dispose_on_disposable_scoped_instances(self):
        desc = ServiceDescriptor(
            service_type=ITracker,
            implementation_type=DisposableTracker,
            lifetime=ServiceLifetime.SCOPED,
        )
        provider = _build_provider_with(desc)
        scope = ServiceScope(provider)
        tracker = scope.get_required_service(ITracker)

        assert not tracker.disposed
        scope.dispose()
        assert tracker.disposed

    def test_dispose_does_not_affect_non_disposable_instances(self):
        desc = ServiceDescriptor(
            service_type=NonDisposableService,
            implementation_type=NonDisposableService,
            lifetime=ServiceLifetime.SCOPED,
        )
        provider = _build_provider_with(desc)
        scope = ServiceScope(provider)
        service = scope.get_required_service(NonDisposableService)

        scope.dispose()
        assert service.value == "alive"

    def test_scoped_instances_cleared_after_disposal(self):
        desc = ServiceDescriptor(
            service_type=ITracker,
            implementation_type=DisposableTracker,
            lifetime=ServiceLifetime.SCOPED,
        )
        provider = _build_provider_with(desc)
        scope = ServiceScope(provider)
        scope.get_required_service(ITracker)

        assert len(scope._scoped_instances) > 0
        scope.dispose()
        assert len(scope._scoped_instances) == 0

    def test_dispose_sets_disposed_flag(self):
        provider = _build_provider_with()
        scope = ServiceScope(provider)

        assert scope._disposed is False
        scope.dispose()
        assert scope._disposed is True


# ---------------------------------------------------------------------------
# Tests: Double disposal
# ---------------------------------------------------------------------------


class TestDoubleDisposal:
    def test_second_dispose_is_noop(self):
        desc = ServiceDescriptor(
            service_type=ITracker,
            implementation_type=DisposableTracker,
            lifetime=ServiceLifetime.SCOPED,
        )
        provider = _build_provider_with(desc)
        scope = ServiceScope(provider)
        tracker = scope.get_required_service(ITracker)

        scope.dispose()
        assert tracker.dispose_count == 1

        scope.dispose()
        assert tracker.dispose_count == 1

    def test_double_dispose_does_not_raise(self):
        provider = _build_provider_with()
        scope = ServiceScope(provider)

        scope.dispose()
        scope.dispose()


# ---------------------------------------------------------------------------
# Tests: Disposal error handling
# ---------------------------------------------------------------------------


class TestDisposalErrorHandling:
    def test_exception_in_one_dispose_does_not_prevent_others(self):
        failing_desc = ServiceDescriptor(
            service_type=FailingDisposable,
            implementation_type=FailingDisposable,
            lifetime=ServiceLifetime.SCOPED,
        )
        tracker_desc = ServiceDescriptor(
            service_type=ITracker,
            implementation_type=DisposableTracker,
            lifetime=ServiceLifetime.SCOPED,
        )
        provider = _build_provider_with(failing_desc, tracker_desc)
        scope = ServiceScope(provider)

        failing = scope.get_required_service(FailingDisposable)
        tracker = scope.get_required_service(ITracker)

        scope.dispose()

        assert failing.disposed
        assert tracker.disposed

    def test_dispose_does_not_propagate_exceptions(self):
        failing_desc = ServiceDescriptor(
            service_type=FailingDisposable,
            implementation_type=FailingDisposable,
            lifetime=ServiceLifetime.SCOPED,
        )
        provider = _build_provider_with(failing_desc)
        scope = ServiceScope(provider)
        scope.get_required_service(FailingDisposable)

        scope.dispose()


# ---------------------------------------------------------------------------
# Tests: ScopeFactory.create_scope_context
# ---------------------------------------------------------------------------


class TestScopeFactoryContextManager:
    def test_create_scope_context_returns_context_manager(self):
        provider = _build_provider_with()
        factory = ScopeFactory(provider)
        ctx = factory.create_scope_context()

        assert hasattr(ctx, "__enter__")
        assert hasattr(ctx, "__exit__")

    def test_scope_is_usable_within_context(self):
        desc = ServiceDescriptor(
            service_type=IScopedMarker,
            implementation_type=ScopedService,
            lifetime=ServiceLifetime.SCOPED,
        )
        provider = _build_provider_with(desc)
        factory = ScopeFactory(provider)

        with factory.create_scope_context() as scope:
            service = scope.get_required_service(IScopedMarker)
            assert isinstance(service, ScopedService)

    def test_scope_is_disposed_on_context_exit(self):
        desc = ServiceDescriptor(
            service_type=ITracker,
            implementation_type=DisposableTracker,
            lifetime=ServiceLifetime.SCOPED,
        )
        provider = _build_provider_with(desc)
        factory = ScopeFactory(provider)
        tracker_ref = None

        with factory.create_scope_context() as scope:
            tracker_ref = scope.get_required_service(ITracker)
            assert not tracker_ref.disposed

        assert tracker_ref.disposed

    def test_scope_is_disposed_on_context_exit_even_on_exception(self):
        desc = ServiceDescriptor(
            service_type=ITracker,
            implementation_type=DisposableTracker,
            lifetime=ServiceLifetime.SCOPED,
        )
        provider = _build_provider_with(desc)
        factory = ScopeFactory(provider)
        tracker_ref = None

        with pytest.raises(ValueError):
            with factory.create_scope_context() as scope:
                tracker_ref = scope.get_required_service(ITracker)
                raise ValueError("simulated error")

        assert tracker_ref.disposed

    def test_context_yields_service_provider_instance(self):
        provider = _build_provider_with()
        factory = ScopeFactory(provider)

        with factory.create_scope_context() as scope:
            assert isinstance(scope, ServiceProvider)
            assert isinstance(scope, ServiceScope)


# ---------------------------------------------------------------------------
# Tests: ScopeFactory from ApplicationBuilder
# ---------------------------------------------------------------------------


class TestScopeFactoryFromApplicationBuilder:
    def test_scope_factory_is_available_after_build(self):
        builder = ApplicationBuilder()
        provider = builder.build(auto_start_hosted_services=False)

        factory = provider.get_service(ScopeFactory)
        assert factory is not None
        assert isinstance(factory, ScopeFactory)

    def test_context_manager_works_end_to_end(self):
        builder = ApplicationBuilder()
        builder.add_scoped(IScopedMarker, ScopedService)
        provider = builder.build(auto_start_hosted_services=False)

        factory = provider.get_required_service(ScopeFactory)
        tracker_ref = None

        with factory.create_scope_context() as scope:
            service = scope.get_required_service(IScopedMarker)
            assert isinstance(service, ScopedService)

    def test_disposable_scoped_service_disposed_via_builder(self):
        builder = ApplicationBuilder()
        builder.add_scoped(ITracker, DisposableTracker)
        provider = builder.build(auto_start_hosted_services=False)

        factory = provider.get_required_service(ScopeFactory)
        tracker_ref = None

        with factory.create_scope_context() as scope:
            tracker_ref = scope.get_required_service(ITracker)
            assert not tracker_ref.disposed

        assert tracker_ref.disposed


# ---------------------------------------------------------------------------
# Tests: Scoped service identity
# ---------------------------------------------------------------------------


class TestScopedServiceIdentity:
    def test_same_scoped_service_within_one_scope(self):
        desc = ServiceDescriptor(
            service_type=IScopedMarker,
            implementation_type=ScopedService,
            lifetime=ServiceLifetime.SCOPED,
        )
        provider = _build_provider_with(desc)
        scope = ServiceScope(provider)

        first = scope.get_required_service(IScopedMarker)
        second = scope.get_required_service(IScopedMarker)

        assert first is second

    def test_different_scoped_service_across_scopes(self):
        desc = ServiceDescriptor(
            service_type=IScopedMarker,
            implementation_type=ScopedService,
            lifetime=ServiceLifetime.SCOPED,
        )
        provider = _build_provider_with(desc)
        scope1 = ServiceScope(provider)
        scope2 = ServiceScope(provider)

        service1 = scope1.get_required_service(IScopedMarker)
        service2 = scope2.get_required_service(IScopedMarker)

        assert service1 is not service2
        assert service1.marker_id() != service2.marker_id()


# ---------------------------------------------------------------------------
# Tests: Singleton shared across scopes
# ---------------------------------------------------------------------------


class TestSingletonSharedAcrossScopes:
    def test_singleton_from_scope1_is_same_as_scope2(self):
        desc = ServiceDescriptor(
            service_type=ISingletonMarker,
            implementation_type=SingletonService,
            lifetime=ServiceLifetime.SINGLETON,
        )
        provider = _build_provider_with(desc)
        scope1 = ServiceScope(provider)
        scope2 = ServiceScope(provider)

        singleton1 = scope1.get_required_service(ISingletonMarker)
        singleton2 = scope2.get_required_service(ISingletonMarker)

        assert singleton1 is singleton2

    def test_singleton_from_scope_is_same_as_root(self):
        desc = ServiceDescriptor(
            service_type=ISingletonMarker,
            implementation_type=SingletonService,
            lifetime=ServiceLifetime.SINGLETON,
        )
        provider = _build_provider_with(desc)
        root_singleton = provider.get_required_service(ISingletonMarker)

        scope = ServiceScope(provider)
        scope_singleton = scope.get_required_service(ISingletonMarker)

        assert root_singleton is scope_singleton

    def test_singleton_survives_scope_disposal(self):
        singleton_desc = ServiceDescriptor(
            service_type=ISingletonMarker,
            implementation_type=SingletonService,
            lifetime=ServiceLifetime.SINGLETON,
        )
        scoped_desc = ServiceDescriptor(
            service_type=IScopedMarker,
            implementation_type=ScopedService,
            lifetime=ServiceLifetime.SCOPED,
        )
        provider = _build_provider_with(singleton_desc, scoped_desc)
        scope = ServiceScope(provider)

        singleton = scope.get_required_service(ISingletonMarker)
        scope.dispose()

        root_singleton = provider.get_required_service(ISingletonMarker)
        assert root_singleton is singleton
