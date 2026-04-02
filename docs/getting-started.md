# Getting Started

This guide walks you through installing Python Application Builder, creating your first application, and understanding the project structure.

## Prerequisites

- Python 3.8 or later
- pip package manager

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Kiryuumaru/python_application_builder.git
cd python_application_builder/src
pip install -r requirements.txt
```

The only external dependency is [loguru](https://github.com/Delgan/loguru) 0.7.3 for structured logging.

## Project Structure

The framework ships as a single module:

```
src/
├── application_builder.py    # The entire framework
├── requirements.txt          # Dependencies
└── main.py                   # Entry point / demo (optional)
```

To use the framework in your project, copy `application_builder.py` into your source tree and import from it.

## Your First Application

### Minimal Example

The simplest possible application:

```python
from application_builder import ApplicationBuilder, ILogger, Worker

class HelloWorker(Worker):
    def __init__(self, logger: ILogger):
        super().__init__()
        self.logger = logger

    def execute(self):
        self.logger.info("Hello from the worker!")

app = ApplicationBuilder()
app.add_worker(HelloWorker)
app.run()
```

This does the following:

1. Creates an `ApplicationBuilder` — the central configuration object
2. Registers `HelloWorker` as a background worker
3. Calls `run()`, which builds the service provider, starts all workers, and blocks until Ctrl+C

### Adding Configuration

```python
from application_builder import ApplicationBuilder, IConfiguration, ILogger, Worker

class GreetingWorker(Worker):
    def __init__(self, config: IConfiguration, logger: ILogger):
        super().__init__()
        self.name = config.get("App:Name", "World")
        self.logger = logger

    def execute(self):
        self.logger.info(f"Hello, {self.name}!")

app = ApplicationBuilder()
app.add_configuration_dictionary({"App": {"Name": "My Application"}})
app.add_worker(GreetingWorker)
app.run()
```

The framework automatically injects `IConfiguration` and `ILogger` into the worker constructor based on type hints.

### Adding Services

Define an interface and implementation, register them, and let the container wire everything together:

```python
from abc import ABC, abstractmethod
from application_builder import ApplicationBuilder, ILogger, Worker

class IGreeter(ABC):
    @abstractmethod
    def greet(self, name: str) -> str:
        pass

class FriendlyGreeter(IGreeter):
    def __init__(self, logger: ILogger):
        self.logger = logger

    def greet(self, name: str) -> str:
        message = f"Hello, {name}! Nice to meet you!"
        self.logger.info(message)
        return message

class GreetingWorker(Worker):
    def __init__(self, greeter: IGreeter, logger: ILogger):
        super().__init__()
        self.greeter = greeter
        self.logger = logger

    def execute(self):
        self.greeter.greet("World")

app = ApplicationBuilder()
app.add_singleton(IGreeter, FriendlyGreeter)
app.add_worker(GreetingWorker)
app.run()
```

## How It Works

### The Build Pipeline

When you call `app.run()` (or `app.build()`), the framework:

1. Builds the `Configuration` from all registered configuration providers
2. Registers built-in services: `IConfiguration`, `ILogger`, `JobManager`, `CliRunnerService`, `IHostEnvironment`, `IHostApplicationLifetime`, `MiddlewarePipeline`
3. Creates a `ServiceProvider` that resolves services on demand
4. Starts all registered workers in dedicated background threads
5. Blocks the main thread until a shutdown signal (Ctrl+C / SIGTERM)

### Constructor Injection

The framework reads `__init__` type hints to resolve dependencies automatically:

```python
class MyService:
    def __init__(self, config: IConfiguration, logger: ILogger):
        # config and logger are resolved from the container
        self.config = config
        self.logger = logger
```

If a parameter's type matches a registered service, the framework injects it. If a required dependency cannot be resolved, it raises `ValueError` at instantiation time.

### `build()` vs `run()`

| Method | Behavior |
|--------|----------|
| `build()` | Creates the `ServiceProvider`, optionally starts workers, and returns the provider for manual control |
| `run()` | Calls `build()`, then blocks the main thread, handling SIGINT/SIGTERM for graceful shutdown |

Use `build()` when you need the service provider for integration with another framework (e.g., a web server). Use `run()` for standalone applications.

```python
# Manual control with build()
provider = app.build(auto_start_hosted_services=False)
service = provider.get_required_service(MyService)
service.do_something()
provider.stop_hosted_services()

# Automatic lifecycle with run()
app.run()  # Blocks until shutdown
```

## Next Steps

- [Dependency Injection](dependency-injection.md) — service registration, lifetimes, scopes
- [Configuration](configuration.md) — multi-source configuration, typed options
- [Workers](workers.md) — background workers and timed workers
- [Logging](logging.md) — structured logging with loguru
- [Advanced Features](advanced-features.md) — cancellation tokens, job management, middleware
- [API Reference](api-reference.md) — complete class and method reference
- [Samples](samples.md) — 20 runnable example applications
