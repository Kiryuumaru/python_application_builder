# Convert Existing Project to Python Application Builder

You are converting an existing Python project to use the **Python Application Builder** framework — a lightweight dependency injection framework built on clean architecture principles. It ships as a single module (`application_builder.py`) providing an IoC container with automatic constructor injection, multi-source configuration, background workers, structured logging, cooperative cancellation, job management, and a middleware pipeline.

Refer to the [python_application_builder](https://github.com/Kiryuumaru/python_application_builder) repository for more detailed information and guidelines.

---

## Step 1: Install the Rules and Documentation First

Before changing any existing file, install the project's rule set and framework documentation. These rules determine how to refactor the existing code into the layered architecture. Every step that follows depends on them.

Clone the framework repository into a temporary directory:

```bash
python -c "import subprocess, tempfile, os; subprocess.run(['git', 'clone', 'https://github.com/Kiryuumaru/python_application_builder.git', os.path.join(tempfile.gettempdir(), 'pab_source')])"
```

Copy the agent instruction files into the project:

```bash
python -c "import shutil, tempfile, os; src=os.path.join(tempfile.gettempdir(), 'pab_source', '.github', 'instructions'); dst=os.path.join('.github', 'instructions'); os.makedirs(dst, exist_ok=True); [shutil.copy2(os.path.join(src, f), os.path.join(dst, f)) for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]"
```

Copy the framework documentation:

```bash
python -c "import shutil, tempfile, os; src=os.path.join(tempfile.gettempdir(), 'pab_source', 'docs'); dst='docs/application_builder'; os.makedirs(dst, exist_ok=True); [shutil.copy2(os.path.join(src, f), os.path.join(dst, f)) for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]"
```

---

## Step 2: Read All Rules End-to-End

Read every file in `.github/instructions/` before refactoring any code. Treat them as MUST/NEVER constraints, not suggestions:

- `project-context.instructions.md` — terminology, breaking-change policy (this project is unreleased; refactor freely)
- `rule-style.instructions.md` — how rules are written
- `architecture.instructions.md` — layering, folder structure, DI lifetimes, workers, ports/adapters, naming, prohibited patterns
- `code-quality.instructions.md` — None handling, docstrings, comments, constructor simplicity
- `documentation.instructions.md` — when to update docs
- `workflow.instructions.md` — build/test commands, pre-commit checks
- `agent-terminal.instructions.md` — terminal usage rules (no `&&`, `|`, `;`, redirections)
- `ui-test-practices.instructions.md` — pytest conventions, no `time.sleep`, DI test doubles

Also skim `docs/application_builder/` for the public API surface.

---

## Step 3: Audit the Existing Project Against the Rules

Map the existing code to the rules from Step 2. Produce an audit before changing anything:

1. Identify entry point(s) — these become the composition root in `presentation/main.py`
2. Identify each existing class and classify it per `architecture.instructions.md` File Placement table:
   - Pure-logic classes with no I/O → `domain/{feature}/services/`
   - Identity-bearing types → `domain/{feature}/entities/`
   - Immutable value types → `domain/{feature}/value_objects/`
   - Orchestration with I/O → `application/{feature}/services/`
   - HTTP/DB/file adapters → `infrastructure/{provider}/adapters/` or `repositories/`
   - CLI/HTTP entry points → `presentation/`
3. For each class identified, record:
   - Which dependencies it needs (these become `__init__` parameters)
   - Whether an ABC interface exists (if not, one MUST be created)
   - Required lifetime per "DI Lifetime Rules by Layer" (Singleton/Scoped/Transient)
4. Identify configuration sources (env vars, JSON, YAML, hardcoded) — all consolidate through `IConfiguration`
5. Identify background tasks, threads, and `time.sleep` loops — these become `Worker` or `TimedWorker`
6. Identify logging (`print`, stdlib `logging`, custom loggers) — all consolidate through `ILogger`

---

## Step 4: Plan the Target Structure From the Rules

Using the audit and the rules, design the target layout. The architecture rules require:

```
src/
├── application_builder.py
├── domain/
│   ├── __init__.py
│   ├── shared/
│   │   ├── interfaces/          <- IUnitOfWork, marker interfaces (one type per file)
│   │   ├── models/              <- Entity, AggregateRoot, ValueObject
│   │   ├── exceptions/
│   │   └── constants/
│   └── {feature}/
│       ├── entities/
│       ├── value_objects/
│       ├── interfaces/          <- I{Feature}Repository, I{Feature}UnitOfWork
│       ├── services/            <- pure domain services
│       ├── events/
│       └── exceptions/
├── application/
│   ├── __init__.py
│   ├── shared/
│   └── {feature}/
│       ├── interfaces/          <- I{Feature}Service, I{External}Provider, internal abstractions
│       ├── services/
│       ├── workers/
│       └── event_handlers/
├── infrastructure/
│   ├── __init__.py
│   └── {provider}_{feature}/
│       ├── adapters/
│       └── repositories/
└── presentation/
    ├── __init__.py
    └── main.py                  <- ONLY module that imports from infrastructure
```

Each layer module MUST expose `register(builder: ApplicationBuilder) -> None`. `presentation/main.py` calls each in dependency order: domain → application → infrastructure → presentation.

---

## Step 5: Install the Framework Module

Copy the framework module into the source directory (adjust `src` to match the existing layout if different):

```bash
python -c "import shutil, tempfile, os; shutil.copy2(os.path.join(tempfile.gettempdir(), 'pab_source', 'src', 'application_builder.py'), os.path.join('src', 'application_builder.py'))"
```

---

## Step 6: Add Framework Dependencies

Add the following pinned dependencies to `requirements.txt`:

```
loguru==0.7.3
pyyaml==6.0.3
```

Install them:

```bash
pip install -r requirements.txt
```

---

## Step 7: Define ABC Interfaces Before Refactoring Implementations

For each class identified in the audit, create an ABC interface in the appropriate module per the rules:

```python
from abc import ABC, abstractmethod


class IMyService(ABC):
    @abstractmethod
    def do_work(self) -> None:
        ...
```

Update the implementation to:
- Implement the ABC
- Accept dependencies via `__init__` with type hints
- Drop manual instantiation of dependencies — the container resolves them
- Be module-private (`_MyService` or omit from `__all__`) since it is resolved via DI

```python
class _MyService(IMyService):
    def __init__(self, logger: ILogger):
        self._logger = logger

    def do_work(self) -> None:
        self._logger.info("Working")
```

---

## Step 8: Convert Background Tasks to Workers

Replace `threading.Thread` + `time.sleep` loops with `Worker` or `TimedWorker`. Workers MUST check `self.is_stopping()` and use `self.wait_for_stop(timeout)` per the architecture rules.

Periodic task:

```python
from application_builder import TimedWorker, ILogger


class ItemProcessorWorker(TimedWorker):
    def __init__(self, logger: ILogger):
        super().__init__(interval=30.0)
        self._logger = logger

    def do_work(self) -> None:
        self._logger.info("Processing items")
```

Free-running task:

```python
from application_builder import Worker, ILogger


class StreamWorker(Worker):
    def __init__(self, logger: ILogger):
        super().__init__()
        self._logger = logger

    def execute(self) -> None:
        while not self.is_stopping():
            try:
                self._process_next()
            except Exception as e:
                self._logger.error(f"Error: {e}")
                self.wait_for_stop(5.0)

    def _process_next(self) -> None:
        ...
```

---

## Step 9: Convert Configuration to IConfiguration

Replace hardcoded values, custom parsers, and scattered `os.environ` calls. Inject `IConfiguration` and use colon-delimited keys per the configuration convention:

```python
class _MyService(IMyService):
    def __init__(self, config: IConfiguration, logger: ILogger):
        self._name = config.get("App:Name", "DefaultName")
        self._debug = config.get("App:Debug", "false") == "true"
        self._logger = logger
```

---

## Step 10: Convert Logging to ILogger

Replace `print()`, stdlib `logging`, and other loggers with `ILogger`. Inject it via constructor — do not create loggers at module scope.

---

## Step 11: Wire the Composition Root

Create `presentation/main.py` (the only module permitted to import from `infrastructure.*`):

```python
from application_builder import ApplicationBuilder

import domain
import application
import infrastructure
import presentation


def main() -> None:
    app = ApplicationBuilder()

    app.add_json_file_configuration("appsettings.json")
    app.add_configuration_dictionary({
        "App": {"Name": "MyApplication"},
    })

    domain.register(app)
    application.register(app)
    infrastructure.register(app)
    presentation.register(app)

    app.run()


if __name__ == "__main__":
    main()
```

Each layer's `register(builder)` function only calls `add_singleton`, `add_scoped`, `add_transient`, or `add_worker`. It MUST NOT instantiate services directly and MUST NOT perform I/O.

---

## Step 12: Verify the Conversion

1. The framework module (`application_builder.py`) is in the source directory
2. All dependencies are injected via constructor type hints — no manual instantiation remains
3. All background tasks use `Worker` or `TimedWorker` (zero `time.sleep` in long-running code)
4. All configuration access goes through `IConfiguration`
5. All logging uses `ILogger`
6. Only `presentation/main.py` imports from `infrastructure.*`
7. Each layer exposes a `register(builder)` function

Run the application:

```bash
cd src
python -m presentation.main
```

Expected: the application starts, structured log messages appear via loguru, and Ctrl+C triggers graceful shutdown of all workers.

---

## Step 13: Pre-Commit Verification

Per `workflow.instructions.md`:

```bash
cd src
python -m py_compile application_builder.py
```

```bash
python -m pytest
```

Both MUST pass.

---

## Conversion Checklist

- [ ] Agent instruction files copied to `.github/instructions/` (Step 1)
- [ ] Framework documentation copied to `docs/application_builder/` (Step 1)
- [ ] All rules read end-to-end (Step 2)
- [ ] Existing code audited and classified by layer (Step 3)
- [ ] Target structure planned per architecture rules (Step 4)
- [ ] Framework module (`application_builder.py`) installed (Step 5)
- [ ] Dependencies pinned and installed (Step 6)
- [ ] ABC interfaces defined for every cross-layer dependency (Step 7)
- [ ] Background threads/tasks converted to `Worker` / `TimedWorker` (Step 8)
- [ ] Configuration consolidated through `IConfiguration` (Step 9)
- [ ] Logging consolidated through `ILogger` (Step 10)
- [ ] Composition root wired in `presentation/main.py` with layered `register()` calls (Step 11)
- [ ] Application starts and shuts down gracefully (Step 12)
- [ ] Pre-commit checks pass (Step 13)

---

## Reminder: Rules Take Precedence

The rules in `.github/instructions/` govern every refactor decision:

- Where each type of file lives (entities, value objects, services, adapters, etc.)
- Which layer may import from which other layer
- Which lifetime each service uses
- Where each ABC interface lives (a single `interfaces/` package per feature)
- Naming conventions for modules, classes, and ABCs
- Prohibited patterns that MUST NEVER be introduced (no `time.sleep` in workers, no bare `except`, no module-level singletons, no infrastructure imports outside `main.py`, etc.)

Do not improvise structure. The rules are the contract.
