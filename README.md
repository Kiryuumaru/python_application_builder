# Python Application Builder

A dependency injection framework for Python inspired by .NET's Generic Host and `IHostBuilder` pattern. Ships as a single module with everything needed for building structured, composable applications: an IoC container with automatic constructor injection, multi-source configuration, background workers, structured logging, cooperative cancellation, job management, and a middleware pipeline.

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE.txt)

## Features

| Feature | Description |
|---------|-------------|
| **Dependency Injection** | IoC container with automatic constructor injection and multi-binding support |
| **Service Lifetimes** | Singleton, Scoped, and Transient lifetimes with scope validation |
| **Configuration** | Multi-source: environment variables, JSON files, command-line args, in-memory dictionaries |
| **Typed Options** | Bind configuration sections to dataclasses via `IOptions` / `IOptionsSnapshot` / `IOptionsMonitor` |
| **Background Workers** | `Worker` (free-running) and `TimedWorker` (interval-based) with graceful shutdown |
| **Structured Logging** | Contextual logging backed by loguru with scoped enrichment |
| **Job Management** | `JobManager` for concurrent background tasks with cancellation and concurrency limits |
| **Cancellation Tokens** | Cooperative cancellation modeled after C#'s `CancellationToken` / `CancellationTokenSource` |
| **Middleware Pipeline** | Composable `MiddlewarePipeline` for request/context processing |
| **Keyed Services** | Named service registrations resolved by key |
| **Service Decoration** | Wrap existing registrations with decorator factories |
| **Host Lifetime** | `IHostApplicationLifetime` events for started/stopping/stopped hooks |
| **CLI Runner** | `CliRunnerService` for running external processes under job management |

## Quick Start

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
app.add_configuration_dictionary({"App": {"Name": "My App"}})
app.add_worker(GreetingWorker)
app.run()
```

## Installation

```bash
pip install -r requirements.txt
```

The only external dependency is [loguru](https://github.com/Delgan/loguru) 0.7.3.

## Documentation

Full documentation lives in the [`docs/`](docs/) folder:

| Document | Description |
|----------|-------------|
| [Getting Started](docs/getting-started.md) | Installation, first application, project structure |
| [Dependency Injection](docs/dependency-injection.md) | Service registration, lifetimes, scopes, keyed services, decoration |
| [Configuration](docs/configuration.md) | Providers, hierarchical keys, typed access, options pattern |
| [Workers](docs/workers.md) | Background workers, timed workers, lifecycle management |
| [Logging](docs/logging.md) | ILogger, log levels, scopes, contextual logging |
| [Advanced Features](docs/advanced-features.md) | Cancellation tokens, job management, middleware, CLI runner, host lifetime |
| [API Reference](docs/api-reference.md) | Complete class and method reference |
| [Samples](docs/samples.md) | Guide to all 20 sample applications |

## Architecture

```
+----------------------------------------------------+
|               APPLICATION LAYER                    |
|  Workers, Services, Entry Points                   |
+----------------------------------------------------+
                       |
                       v
+----------------------------------------------------+
|            DEPENDENCY INJECTION CORE               |
|  ApplicationBuilder -> ServiceProvider             |
|  ServiceDescriptor, ServiceScope, ScopeFactory     |
|  Lifetimes: Singleton | Scoped | Transient         |
+----------------------------------------------------+
                       |
                       v
+----------------------------------------------------+
|            INFRASTRUCTURE SERVICES                 |
|  Configuration    Logging       JobManager         |
|  IOptions         ILogger       CancellationToken  |
|  Middleware        CliRunner     HostLifetime       |
+----------------------------------------------------+
```

## Service Lifetimes at a Glance

```python
app = ApplicationBuilder()

# One instance for the entire application
app.add_singleton(DatabaseService)

# One instance per scope (request, unit-of-work)
app.add_scoped(RequestContext)

# New instance on every resolution
app.add_transient(EmailMessage)
```

## Registration Patterns

```python
# Concrete type
app.add_singleton(MyService)

# Interface -> Implementation
app.add_singleton(IMyService, MyService)

# Factory function
app.add_singleton_factory(IMyService, lambda sp: MyService(sp.get_required_service(IConfig)))

# Pre-built instance
app.add_singleton_instance(IMyService, my_instance)

# Keyed / named
app.add_keyed_singleton(IStorage, "primary", SqlStorage)
app.add_keyed_singleton(IStorage, "cache", RedisStorage)

# Conditional (only if not already registered)
app.try_add_singleton(IMyService, MyService)

# Decoration
app.decorate(IMyService, lambda sp, inner: LoggingDecorator(inner, sp.get_required_service(ILogger)))
```

## Configuration Sources

```python
app = ApplicationBuilder()

# Environment variables (loaded by default)
# Use prefix to filter: MYAPP_DATABASE__HOST -> Database:HOST
app.add_configuration(lambda b: b.add_environment_variables("MYAPP_"))

# JSON file
app.add_configuration(lambda b: b.add_json_file("appsettings.json"))

# Command-line arguments
app.add_configuration(lambda b: b.add_command_line())

# In-memory dictionary (highest priority — last wins)
app.add_configuration_dictionary({
    "Database": {"Host": "localhost", "Port": 5432},
    "Logging": {"Level": "INFO"}
})
```

Later sources override earlier ones. Keys use colon-delimited hierarchy: `Database:Host`.

## Background Workers

```python
class DataWorker(Worker):
    def __init__(self, service: IDataService, logger: ILogger):
        super().__init__()
        self.service = service
        self.logger = logger

    def execute(self):
        while not self.is_stopping():
            try:
                self.service.process_next()
            except Exception as e:
                self.logger.error(f"Error: {e}")
                self.wait_for_stop(5.0)

class PingWorker(TimedWorker):
    def __init__(self, logger: ILogger):
        super().__init__(interval_seconds=30)
        self.logger = logger

    def do_work(self):
        self.logger.info("Ping!")

app.add_worker(DataWorker)
app.add_worker(PingWorker)
```

## Samples

The [`samples/`](samples/) directory contains 20 runnable examples covering every major feature. See the [Samples Guide](docs/samples.md) for details.

| Sample | Feature |
|--------|---------|
| `build_runner` | Multi-binding with `List[T]` injection |
| `chat_room` | Scoped services and multi-binding formatters |
| `cli_args` | Command-line argument configuration |
| `data_pipeline` | Multi-worker producer/consumer |
| `decorated_services` | Service decoration (logging + caching) |
| `disposable_scopes` | `IDisposable` cleanup in scopes |
| `env_aware` | `IHostEnvironment` and environment-based config |
| `event_bus` | Transient/scoped/singleton lifetime interplay |
| `health_dashboard` | Pre-built singleton via `add_singleton_instance` |
| `inventory_cli` | Combined singleton + scoped services |
| `job_scheduler` | `JobManager` background task execution |
| `keyed_services` | Keyed/named service resolution |
| `lifecycle_hooks` | `IHostApplicationLifetime` events |
| `middleware_demo` | `MiddlewarePipeline` composition |
| `multi_config` | Multi-source configuration with priority |
| `plugin_system` | Factory-based dynamic plugin selection |
| `service_collection` | `try_add`, `replace`, `remove_all` APIs |
| `task_queue` | Job queue with concurrency limits |
| `typed_options` | `configure_options` for dataclass binding |
| `validated_app` | Build-time and scope validation |

## Testing

Constructor injection makes testing straightforward — pass mocks directly:

```python
from unittest.mock import Mock

mock_logger = Mock()
mock_config = Mock()
mock_config.get.return_value = "test_value"

service = MyService(mock_config, mock_logger)
service.do_something()

mock_logger.info.assert_called_once()
```

For integration tests, build a real container:

```python
app = ApplicationBuilder()
app.add_configuration_dictionary({"key": "value"})
app.add_singleton(IMyService, TestMyService)
provider = app.build(auto_start_hosted_services=False)

service = provider.get_required_service(IMyService)
assert service is not None
```

## Development Setup

```bash
git clone https://github.com/Kiryuumaru/python_application_builder.git
cd python_application_builder/src
pip install -r requirements.txt
python -m pytest
```

## License

MIT License — see [LICENSE.txt](LICENSE.txt) for details.

## Acknowledgments

- [loguru](https://github.com/Delgan/loguru) — structured logging
- [.NET Generic Host](https://learn.microsoft.com/en-us/dotnet/core/extensions/generic-host) — architectural inspiration