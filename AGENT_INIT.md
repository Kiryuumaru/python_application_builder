# Initialize Python Application Builder Project

Copy this entire document into your AI agent prompt to scaffold a new project using the [python_application_builder](https://github.com/Kiryuumaru/python_application_builder) framework.

---

## Instructions

You are initializing a Python project that uses the **Python Application Builder** framework — a lightweight dependency injection framework built on clean architecture principles. It ships as a single module (`application_builder.py`) providing an IoC container with automatic constructor injection, multi-source configuration, background workers, structured logging, cooperative cancellation, job management, and a middleware pipeline.

Follow these steps to set up the project in the current directory.

---

## Step 1: Create Project Structure

Create the following directory layout:

```
<project_root>/
├── src/
│   ├── application_builder.py   <- Framework module (copied from the repository)
│   ├── main.py                  <- Application entry point
│   └── requirements.txt         <- Dependencies
└── tests/
    └── (test files)
```

---

## Step 2: Install the Framework

Clone the framework repository and copy the core module into your project:

```bash
git clone https://github.com/Kiryuumaru/python_application_builder.git /tmp/pab_source
cp /tmp/pab_source/src/application_builder.py src/application_builder.py
```

---

## Step 3: Create requirements.txt

Create `src/requirements.txt` with the following pinned dependencies:

```
loguru==0.7.3
pyyaml==6.0.3
```

---

## Step 4: Install Dependencies

```bash
cd src
pip install -r requirements.txt
```

---

## Step 5: Create the Entry Point

Create `src/main.py` with the following starter template:

```python
from application_builder import ApplicationBuilder, IConfiguration, ILogger, Worker


class MainWorker(Worker):
    def __init__(self, config: IConfiguration, logger: ILogger):
        super().__init__()
        self.config = config
        self.logger = logger

    def execute(self):
        self.logger.info("Application started")
        while not self.is_stopping():
            self.wait_for_stop(1.0)


app = ApplicationBuilder()
app.add_configuration_dictionary({
    "App": {
        "Name": "MyApplication"
    }
})
app.add_worker(MainWorker)
app.run()
```

---

## Step 6: Verify the Setup

Run the application to confirm everything works:

```bash
cd src
python main.py
```

The application should start, print a log message, and block until you press Ctrl+C.

---

## Framework Reference

### Core Imports

All framework types are imported from a single module:

```python
from application_builder import (
    ApplicationBuilder,     # Central builder for configuring and running the app
    IConfiguration,         # ABC for reading configuration values
    ILogger,                # ABC for structured logging
    Worker,                 # ABC for free-running background workers
    TimedWorker,            # ABC for interval-based background workers
    CancellationToken,      # Cooperative cancellation token
    CancellationTokenSource,# Creates and controls cancellation tokens
    JobManager,             # Manages concurrent background tasks
    CliRunnerService,       # Runs external CLI processes
    MiddlewarePipeline,     # Composable request/context processing pipeline
    IHostEnvironment,       # ABC for environment info (env name, content root)
    IHostApplicationLifetime, # ABC for application lifetime events
    IOptions,               # ABC for typed options (singleton snapshot)
    IOptionsSnapshot,       # ABC for typed options (scoped, reloads per scope)
    IOptionsMonitor,        # ABC for typed options (singleton, live reload)
    ServiceLifetime,        # Enum: SINGLETON, SCOPED, TRANSIENT
)
```

### Service Registration

```python
app = ApplicationBuilder()

# Concrete type (self-binding)
app.add_singleton(MyService)

# Interface -> Implementation
app.add_singleton(IMyService, MyService)
app.add_scoped(IMyService, MyService)
app.add_transient(IMyService, MyService)

# Factory function
app.add_singleton_factory(IMyService, lambda sp: MyService(sp.get_required_service(IConfig)))

# Pre-built instance
app.add_singleton_instance(IMyService, my_instance)

# Keyed / named services
app.add_keyed_singleton(IStorage, "primary", SqlStorage)

# Conditional registration (only if not already registered)
app.try_add_singleton(IMyService, MyService)

# Decoration (wraps existing registration)
app.decorate(IMyService, lambda sp, inner: LoggingDecorator(inner, sp.get_required_service(ILogger)))
```

### Service Lifetimes

| Lifetime | Behavior | Use For |
|----------|----------|---------|
| Singleton | One instance for the entire application | Stateless services, configuration, shared state |
| Scoped | One instance per scope (request, unit-of-work) | Request-level or unit-of-work services |
| Transient | New instance on every resolution | Stateless utilities, messages, short-lived objects |

### Configuration Sources

```python
# Environment variables (loaded by default)
app.add_environment_variables_configuration("MYAPP_")

# JSON file
app.add_json_file_configuration("appsettings.json")

# YAML file
app.add_yaml_file_configuration("appsettings.yaml")

# Command-line arguments
app.add_command_line_configuration()

# In-memory flat key-value pairs
app.add_in_memory_configuration({"App:Name": "MyApp", "App:Port": "8080"})

# In-memory nested dictionary (highest priority — last wins)
app.add_configuration_dictionary({
    "Database": {"Host": "localhost", "Port": 5432},
    "Logging": {"Level": "INFO"}
})
```

Configuration keys use colon-delimited hierarchy: `Section:SubSection:Key`. Later sources override earlier ones.

### Background Workers

```python
# Free-running worker — implement execute()
class DataWorker(Worker):
    def __init__(self, logger: ILogger):
        super().__init__()
        self.logger = logger

    def execute(self):
        while not self.is_stopping():
            self.logger.info("Processing...")
            self.wait_for_stop(5.0)

# Interval-based worker — implement do_work()
class PingWorker(TimedWorker):
    def __init__(self, logger: ILogger):
        super().__init__(interval_seconds=30)
        self.logger = logger

    def do_work(self):
        self.logger.info("Ping!")

app.add_worker(DataWorker)
app.add_worker(PingWorker)
```

### Defining Service Interfaces

Use `ABC` and `@abstractmethod` for service contracts:

```python
from abc import ABC, abstractmethod

class IGreeter(ABC):
    @abstractmethod
    def greet(self, name: str) -> str:
        pass

class FriendlyGreeter(IGreeter):
    def __init__(self, logger: ILogger):
        self.logger = logger

    def greet(self, name: str) -> str:
        message = f"Hello, {name}!"
        self.logger.info(message)
        return message
```

### Constructor Injection

The framework reads `__init__` type hints to resolve dependencies automatically. Declare dependencies as typed constructor parameters:

```python
class MyService:
    def __init__(self, config: IConfiguration, logger: ILogger, greeter: IGreeter):
        self.config = config
        self.logger = logger
        self.greeter = greeter
```

### Testing

Pass mocks directly via constructor injection:

```python
from unittest.mock import Mock

mock_logger = Mock()
mock_config = Mock()
mock_config.get.return_value = "test_value"

service = MyService(mock_config, mock_logger)
service.do_something()
mock_logger.info.assert_called_once()
```

For integration tests, build a real container without starting workers:

```python
app = ApplicationBuilder()
app.add_configuration_dictionary({"key": "value"})
app.add_singleton(IMyService, TestMyService)
provider = app.build(auto_start_hosted_services=False)

service = provider.get_required_service(IMyService)
assert service is not None
```

---

## Architecture Rules

- Define ABC interfaces for service contracts before implementations
- Accept dependencies via `__init__` parameters with type hints
- Never use global state or module-level singletons for service resolution
- Singleton services must be thread-safe and must not hold references to scoped or transient services
- Workers must check `self.is_stopping()` in loops and use `self.wait_for_stop(timeout)` instead of `time.sleep()`
- Workers must handle exceptions within the execute loop

---

## Additional Resources

- [Full Documentation](https://github.com/Kiryuumaru/python_application_builder/tree/main/docs)
- [20 Sample Applications](https://github.com/Kiryuumaru/python_application_builder/tree/main/samples)
- [API Reference](https://github.com/Kiryuumaru/python_application_builder/blob/main/docs/api-reference.md)
