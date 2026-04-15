import threading
from abc import ABC, abstractmethod
from application_builder import ApplicationBuilder, ServiceLifetime


class ICounter(ABC):
    @abstractmethod
    def get_count(self) -> int:
        pass


class ExpensiveService(ICounter):
    _creation_count = 0
    _creation_lock = threading.Lock()

    def __init__(self):
        with ExpensiveService._creation_lock:
            ExpensiveService._creation_count += 1

    def get_count(self) -> int:
        return ExpensiveService._creation_count

    @classmethod
    def reset(cls):
        with cls._creation_lock:
            cls._creation_count = 0


class TestSingletonThreadSafety:

    def test_singleton_created_once_under_concurrent_access(self):
        ExpensiveService.reset()

        app = ApplicationBuilder()
        app.add_singleton(ICounter, ExpensiveService)
        provider = app.build()

        barrier = threading.Barrier(10)
        results = [None] * 10
        errors = []

        def resolve_singleton(index):
            try:
                barrier.wait(timeout=5)
                scope = provider.create_scope()
                service = scope.get_required_service(ICounter)
                results[index] = service
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(10):
            t = threading.Thread(target=resolve_singleton, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Errors during resolution: {errors}"
        assert ExpensiveService._creation_count == 1, (
            f"Singleton constructor called {ExpensiveService._creation_count} times, expected 1"
        )

        first = results[0]
        for i, result in enumerate(results):
            assert result is first, (
                f"Thread {i} got a different instance (id={id(result)}) "
                f"than thread 0 (id={id(first)})"
            )

    def test_singleton_lock_shared_across_scopes(self):
        ExpensiveService.reset()

        app = ApplicationBuilder()
        app.add_singleton(ICounter, ExpensiveService)
        provider = app.build()

        scope1 = provider.create_scope()
        scope2 = provider.create_scope()

        service1 = scope1.get_required_service(ICounter)
        service2 = scope2.get_required_service(ICounter)

        assert service1 is service2
        assert ExpensiveService._creation_count == 1
