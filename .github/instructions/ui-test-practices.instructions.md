---
applyTo: 'tests/**/*.py'
---
# Test Rules

## Core Principle

Tests exercise the framework through its public API. No reaching into private state, no monkey-patching internals, no bypassing constructor injection.

---

## Prohibited Patterns

- NEVER access private attributes (`obj._field`) in tests to set state
- NEVER monkey-patch framework internals to make a test pass
- NEVER use `time.sleep()` in tests (flaky — use events or polling helpers)
- NEVER use `threading.Event().wait()` without a timeout
- NEVER call internal `_` functions/methods directly when a public API exists
- NEVER instantiate concrete framework services without going through `ApplicationBuilder` or `ServiceProvider`
- NEVER share mutable state across tests via module-level globals
- NEVER catch and swallow exceptions in tests (let them surface)
- NEVER assert on log message contents that are not part of the contract
- NEVER depend on test execution order
- NEVER use real network, real filesystem (outside `tmp_path`), or real environment variables without `monkeypatch`

---

## Required Test Actions

- MUST construct services via `ApplicationBuilder` or `ServiceProvider`
- MUST register dependencies via `add_singleton`, `add_scoped`, or `add_transient`
- MUST resolve services via `get_service()` or constructor injection
- MUST use `pytest` fixtures from `conftest.py` for shared setup
- MUST use `tmp_path` for filesystem operations
- MUST use `monkeypatch` for environment variables and module attributes
- MUST use `caplog` for log assertions

---

## Required Wait Strategies

- For worker startup, USE a synchronization primitive (`threading.Event`) signaled by the worker
- For worker stop, USE `worker.wait_for_stop(timeout)` or assert `is_stopping()`
- For background completion, USE `event.wait(timeout=N)` with explicit timeout and assert the result
- For polling, USE a bounded loop with `time.monotonic()` deadline, not unbounded `while True`

```python
done = threading.Event()

class SignallingWorker(Worker):
    def execute(self) -> None:
        done.set()

# ...register and start...
assert done.wait(timeout=5.0), "worker did not start within 5s"
```

---

## Selector Priority for Test Targets

When asserting against framework state, prefer in this order:

1. PREFER public API return values (`provider.get_service(IService)`)
2. PREFER public properties on resolved services
3. PREFER captured side effects via test doubles registered in DI
4. PREFER `caplog.records` for log-driven assertions
5. AVOID introspecting `provider._descriptors` or other private framework state

---

## Test Doubles via DI

Replace real implementations with fakes/spies by registering them in the test's `ApplicationBuilder`:

```python
class FakeEmailSender(IEmailSender):
    def __init__(self) -> None:
        self.sent: list[str] = []
    def send(self, to: str) -> None:
        self.sent.append(to)

fake = FakeEmailSender()
app = ApplicationBuilder()
app.add_singleton(IEmailSender, instance=fake)
provider = app.build()
```

- MUST register fakes through the same DI mechanism as production code
- MUST NOT replace services by overwriting `provider._services` or similar
- MUST verify behavior on the fake instance, not on the real implementation

---

## Cancellation in Tests

- MUST use `CancellationToken` to stop long-running operations under test
- MUST assert that operations honor cancellation within a bounded timeout
- MUST NOT rely on process exit to clean up worker threads

```python
cts = CancellationTokenSource()
cts.cancel_after(1.0)
worker.run(cts.token)
assert worker.is_stopping()
```

---

## Required Naming and Layout

- Test files MUST live under `tests/`
- Test files MUST be named `test_{subject}.py`
- Test functions MUST be named `test_{behavior}`
- Shared fixtures MUST live in `tests/conftest.py`
- Helpers MUST live in `tests/helpers/` or be exposed as fixtures

---

## Pre-Commit Verification

- MUST run `python -m pytest` and see 100% pass
- MUST have zero `time.sleep` calls in test code
- MUST have zero direct accesses to private attributes (`_name`) of framework types in assertions
- MUST have all worker-based tests use explicit timeouts on waits
- MUST have all filesystem tests use `tmp_path`
- MUST have all environment-dependent tests use `monkeypatch`
