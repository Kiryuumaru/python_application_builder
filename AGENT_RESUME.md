# Convert Existing Project to Python Application Builder

You are converting an existing Python project to use the **Python Application Builder** framework — a lightweight dependency injection framework built on clean architecture principles. It ships as a single module (`application_builder.py`) providing an IoC container with automatic constructor injection, multi-source configuration, background workers, structured logging, cooperative cancellation, job management, and a middleware pipeline.

Follow these steps to integrate the framework into the current project.

---

## Step 1: Analyze the Existing Project

Before making changes, understand the current codebase:

1. Identify the entry point(s) (e.g., `main.py`, `app.py`, `run.py`)
2. Identify existing services, classes, and their dependencies
3. Identify configuration sources (environment variables, config files, hardcoded values)
4. Identify any background tasks, scheduled jobs, or long-running loops
5. Identify logging setup (print statements, stdlib logging, third-party loggers)
6. Note the existing directory structure and where source files live

---

## Step 2: Install the Framework

Clone the framework repository into a temporary directory and copy the core module, documentation, and agent instructions into your project. The following commands use Python for cross-platform compatibility:

```bash
python -c "import subprocess, tempfile, os; subprocess.run(['git', 'clone', 'https://github.com/Kiryuumaru/python_application_builder.git', os.path.join(tempfile.gettempdir(), 'pab_source')])"
```

Copy the framework module into your source directory (adjust `src` to match your project layout):

```bash
python -c "import shutil, tempfile, os; shutil.copy2(os.path.join(tempfile.gettempdir(), 'pab_source', 'src', 'application_builder.py'), os.path.join('src', 'application_builder.py'))"
```

Copy the framework documentation:

```bash
python -c "import shutil, tempfile, os; src=os.path.join(tempfile.gettempdir(), 'pab_source', 'docs'); dst='docs/application_builder'; os.makedirs(dst, exist_ok=True); [shutil.copy2(os.path.join(src, f), os.path.join(dst, f)) for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]"
```

Copy the agent instruction files:

```bash
python -c "import shutil, tempfile, os; src=os.path.join(tempfile.gettempdir(), 'pab_source', '.github', 'instructions'); dst=os.path.join('.github', 'instructions'); os.makedirs(dst, exist_ok=True); [shutil.copy2(os.path.join(src, f), os.path.join(dst, f)) for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]"
```

---

## Step 3: Add Framework Dependencies

Add the following to the project's `requirements.txt` (or equivalent dependency file) if not already present:

```
loguru==0.7.3
pyyaml==6.0.3
```

Install them:

```bash
pip install -r requirements.txt
```

---

## Step 4: Define Service Interfaces

For each existing class that other classes depend on, create an ABC interface:

```python
from abc import ABC, abstractmethod

class IMyService(ABC):
    @abstractmethod
    def do_work(self) -> None:
        pass
```

Then update the existing class to implement the interface:

```python
class MyService(IMyService):
    def __init__(self, logger: ILogger):
        self.logger = logger

    def do_work(self) -> None:
        self.logger.info("Working")
```

Guidelines:
- Define the ABC interface before the implementation
- Accept dependencies via `__init__` parameters with type hints
- Remove manual instantiation of dependencies — let the container inject them

---

## Step 5: Register Services

In the entry point, register all services with the `ApplicationBuilder`:

```python
from application_builder import ApplicationBuilder

app = ApplicationBuilder()

# Register services with appropriate lifetimes
app.add_singleton(IMyService, MyService)        # One instance for app lifetime
app.add_scoped(IRequestHandler, RequestHandler) # One instance per scope
app.add_transient(IValidator, Validator)         # New instance every time
```

Lifetime selection:
- **Singleton** — thread-safe, stateless or shared state, configuration, connection pools
- **Scoped** — request-level state, unit of work, database sessions
- **Transient** — lightweight, stateless utilities, messages, short-lived objects

---

## Step 6: Convert Configuration

Replace hardcoded values, custom config parsers, or scattered `os.environ` calls with the framework's configuration system:

```python
app = ApplicationBuilder()

# Environment variables (loaded by default, add prefix filter if needed)
app.add_environment_variables_configuration("MYAPP_")

# JSON config file
app.add_json_file_configuration("appsettings.json")

# YAML config file
app.add_yaml_file_configuration("config.yaml")

# In-memory overrides
app.add_configuration_dictionary({
    "App": {
        "Name": "MyApplication",
        "Debug": "false"
    }
})
```

In services, inject `IConfiguration` and access values with colon-delimited keys:

```python
class MyService:
    def __init__(self, config: IConfiguration):
        self.name = config.get("App:Name", "DefaultName")
        self.debug = config.get("App:Debug", "false") == "true"
```

---

## Step 7: Convert Background Tasks

Replace `threading.Thread`, `time.sleep` loops, or scheduled tasks with Workers:

**Before:**

```python
import threading
import time

def background_task():
    while running:
        process_items()
        time.sleep(30)

thread = threading.Thread(target=background_task, daemon=True)
thread.start()
```

**After:**

```python
from application_builder import TimedWorker, ILogger

class ItemProcessorWorker(TimedWorker):
    def __init__(self, logger: ILogger):
        super().__init__(interval=30.0)
        self.logger = logger

    def do_work(self):
        self.logger.info("Processing items")
        self.process_items()

    def process_items(self):
        # Existing logic here
        pass
```

For free-running loops, use `Worker` instead of `TimedWorker`:

```python
from application_builder import Worker, ILogger

class StreamWorker(Worker):
    def __init__(self, logger: ILogger):
        super().__init__()
        self.logger = logger

    def execute(self):
        self.logger.info("Stream worker started")
        while not self.is_stopping():
            try:
                self.process_next()
            except Exception as e:
                self.logger.error(f"Error: {e}")
                self.wait_for_stop(5.0)
```

Register workers:

```python
app.add_worker(ItemProcessorWorker)
app.add_worker(StreamWorker)
```

Worker rules:
- MUST check `self.is_stopping()` in loops
- MUST use `self.wait_for_stop(timeout)` instead of `time.sleep()`
- MUST handle exceptions within the execute loop

---

## Step 8: Convert Logging

Replace `print()`, stdlib `logging`, or other loggers with the framework's `ILogger`:

**Before:**

```python
import logging
logger = logging.getLogger(__name__)
logger.info("Starting up")
```

**After:**

```python
from application_builder import ILogger

class MyService:
    def __init__(self, logger: ILogger):
        self.logger = logger

    def start(self):
        self.logger.info("Starting up")
```

The framework automatically provides contextual logging backed by loguru. Each service gets a logger scoped to its class name.

---

## Step 9: Create the Entry Point

Replace the existing entry point with the `ApplicationBuilder` pattern:

```python
from application_builder import ApplicationBuilder, IConfiguration, ILogger, Worker

# Import your interfaces and implementations
from my_services import IMyService, MyService
from my_workers import ItemProcessorWorker

app = ApplicationBuilder()

# Configuration
app.add_json_file_configuration("appsettings.json")
app.add_configuration_dictionary({
    "App": {"Name": "MyApplication"}
})

# Services
app.add_singleton(IMyService, MyService)

# Workers
app.add_worker(ItemProcessorWorker)

# Run (blocks until Ctrl+C / SIGTERM)
app.run()
```

---

## Step 10: Verify the Conversion

After completing all steps, verify:

1. The project has `application_builder.py` in the source directory
2. All dependencies are injected via constructor type hints — no manual instantiation
3. All background tasks use `Worker` or `TimedWorker`
4. All configuration access goes through `IConfiguration`
5. All logging uses `ILogger`
6. The application starts and shuts down gracefully

Run the application:

```bash
cd src
python main.py
```

Expected: the application starts, structured log messages appear via loguru, and Ctrl+C triggers graceful shutdown of all workers.

---

## Conversion Checklist

- [ ] Framework module (`application_builder.py`) copied into source directory
- [ ] Framework documentation copied to `docs/application_builder/`
- [ ] Agent instruction files copied to `.github/instructions/`
- [ ] Dependencies (`loguru`, `pyyaml`) added and installed
- [ ] ABC interfaces defined for service contracts
- [ ] Existing classes updated to accept dependencies via `__init__` type hints
- [ ] Services registered with `ApplicationBuilder` using appropriate lifetimes
- [ ] Manual instantiation replaced with container resolution
- [ ] Configuration consolidated through `IConfiguration`
- [ ] Background threads/tasks converted to `Worker` or `TimedWorker`
- [ ] Logging converted to `ILogger`
- [ ] Entry point uses `ApplicationBuilder.run()` or `ApplicationBuilder.build()`
- [ ] Application starts and shuts down gracefully

---

## Architecture Rules

- Define ABC interfaces for service contracts before implementations
- Accept dependencies via `__init__` parameters with type hints
- Never use global state or module-level singletons for service resolution
- Singleton services must be thread-safe and must not hold references to scoped or transient services
- Workers must check `self.is_stopping()` in loops and use `self.wait_for_stop(timeout)` instead of `time.sleep()`
- Workers must handle exceptions within the execute loop
- Refer to `docs/application_builder/` for detailed framework documentation
