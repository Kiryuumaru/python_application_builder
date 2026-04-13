# API Reference

Complete reference for all public classes, methods, and functions in `application_builder.py`.

## Table of Contents

- [ApplicationBuilder](#applicationbuilder)
- [ServiceProvider](#serviceprovider)
- [ServiceScope](#servicescope)
- [ScopeFactory](#scopefactory)
- [ServiceDescriptor](#servicedescriptor)
- [ServiceLifetime](#servicelifetime)
- [IConfiguration / IConfigurationSection](#iconfiguration--iconfigurationsection)
- [Configuration](#configuration)
- [ConfigurationBuilder](#configurationbuilder)
- [ConfigurationSection](#configurationsection)
- [Configuration Providers](#configuration-providers)
- [IOptions / IOptionsSnapshot / IOptionsMonitor](#ioptions--ioptionssnapshot--ioptionsmonitor)
- [bind_configuration](#bind_configuration)
- [ILogger](#ilogger)
- [LoguruLogger](#logurulogger)
- [LogScope](#logscope)
- [Worker / IWorker](#worker--iworker)
- [TimedWorker](#timedworker)
- [WorkerState](#workerstate)
- [WorkerManager](#workermanager)
- [JobManager](#jobmanager)
- [CancellationToken](#cancellationtoken)
- [CancellationTokenSource](#cancellationtokensource)
- [CancellationTokenRegistration](#cancellationtokenregistration)
- [OperationCanceledException](#operationcanceledexception)
- [Utility Functions](#utility-functions)
- [IHostEnvironment](#ihostenvironment)
- [HostEnvironment](#hostenvironment)
- [IHostApplicationLifetime](#ihostapplicationlifetime)
- [HostApplicationLifetime](#hostapplicationlifetime)
- [IDisposable](#idisposable)
- [IMiddleware](#imiddleware)
- [MiddlewarePipeline](#middlewarepipeline)
- [IChangeToken](#ichangetoken)
- [ConfigurationChangeToken](#configurationchangetoken)
- [FileChangeWatcher](#filechangewatcher)
- [CliRunnerService](#clirunnerservice)

---

## ApplicationBuilder

Container for service registrations. Entry point for configuring and building an application.

```python
class ApplicationBuilder:
    def __init__(self)
```

### Service Registration

| Method | Description |
|--------|-------------|
| `add(descriptor: ServiceDescriptor) -> ApplicationBuilder` | Add a raw service descriptor |
| `add_singleton(service_type, implementation_type=None) -> ApplicationBuilder` | Register a singleton service |
| `add_singleton_instance(service_type, instance) -> ApplicationBuilder` | Register a pre-built singleton instance |
| `add_singleton_factory(service_type, factory) -> ApplicationBuilder` | Register a singleton with factory `Callable[[ServiceProvider], T]` |
| `add_scoped(service_type, implementation_type=None) -> ApplicationBuilder` | Register a scoped service |
| `add_scoped_factory(service_type, factory) -> ApplicationBuilder` | Register a scoped service with factory |
| `add_transient(service_type, implementation_type=None) -> ApplicationBuilder` | Register a transient service |
| `add_transient_factory(service_type, factory) -> ApplicationBuilder` | Register a transient service with factory |

### Conditional Registration

| Method | Description |
|--------|-------------|
| `try_add_singleton(service_type, implementation_type=None) -> ApplicationBuilder` | Register singleton only if not already registered |
| `try_add_scoped(service_type, implementation_type=None) -> ApplicationBuilder` | Register scoped only if not already registered |
| `try_add_transient(service_type, implementation_type=None) -> ApplicationBuilder` | Register transient only if not already registered |
| `replace(descriptor: ServiceDescriptor) -> ApplicationBuilder` | Replace the first registration for the same service type |
| `remove_all(service_type: Type) -> ApplicationBuilder` | Remove all registrations for a service type |

### Keyed Services

| Method | Description |
|--------|-------------|
| `add_keyed_singleton(service_type, key, implementation_type=None) -> ApplicationBuilder` | Register a keyed singleton |
| `add_keyed_singleton_factory(service_type, key, factory) -> ApplicationBuilder` | Register a keyed singleton with factory |
| `add_keyed_scoped(service_type, key, implementation_type=None) -> ApplicationBuilder` | Register a keyed scoped service |
| `add_keyed_transient(service_type, key, implementation_type=None) -> ApplicationBuilder` | Register a keyed transient service |

### Decoration

| Method | Description |
|--------|-------------|
| `decorate(service_type, decorator_factory) -> ApplicationBuilder` | Wrap a service with `decorator_factory(provider, inner) -> decorated` |

### Options

| Method | Description |
|--------|-------------|
| `configure_options(options_type, section_key) -> ApplicationBuilder` | Register `IOptions`, `IOptionsSnapshot`, `IOptionsMonitor` for a type bound to a config section |

### Workers

| Method | Description |
|--------|-------------|
| `add_worker(implementation_type: Type[IWorker]) -> ApplicationBuilder` | Register a hosted worker service |

### Configuration

| Method | Description |
|--------|-------------|
| `add_configuration(configure_action: Callable[[ConfigurationBuilder], None]) -> ApplicationBuilder` | Configure via the `ConfigurationBuilder` |
| `add_configuration_dictionary(config_dict: Dict[str, Any]) -> ApplicationBuilder` | Add a nested dictionary (auto-flattened) |

### Validation

| Method | Description |
|--------|-------------|
| `set_validate_on_build(enabled=True) -> ApplicationBuilder` | Enable build-time dependency validation |
| `set_validate_scopes(enabled=True) -> ApplicationBuilder` | Prevent scoped resolution from root provider |

### Build & Run

| Method | Description |
|--------|-------------|
| `build(auto_start_hosted_services=True) -> ServiceProvider` | Build the provider; optionally start workers |
| `run() -> None` | Build, start workers, and block until SIGINT/SIGTERM |

---

## ServiceProvider

Resolves services from registered descriptors.

```python
class ServiceProvider:
    def __init__(self, descriptors, validate_scopes=False, decorators=None)
```

| Method | Description |
|--------|-------------|
| `get_service(service_type: Type[T]) -> Optional[T]` | Resolve a service, or `None` |
| `get_required_service(service_type: Type[T]) -> T` | Resolve or raise `KeyError` |
| `get_keyed_service(service_type: Type[T], key: str) -> Optional[T]` | Resolve a keyed service, or `None` |
| `get_required_keyed_service(service_type: Type[T], key: str) -> T` | Resolve keyed or raise `KeyError` |
| `get_services(service_type: Type[T]) -> List[T]` | Resolve all registrations for a type |
| `create_scope() -> ServiceScope` | Create a new service scope |
| `validate_all_registrations() -> None` | Check all dependencies can resolve; raises `ValueError` |
| `start_hosted_services() -> None` | Discover and start all workers |
| `stop_hosted_services() -> None` | Stop all running workers |

---

## ServiceScope

Extends `ServiceProvider` with scoped instance tracking and disposal.

```python
class ServiceScope(ServiceProvider):
    def __init__(self, root_provider: ServiceProvider)
```

| Method | Description |
|--------|-------------|
| `dispose() -> None` | Dispose all scoped `IDisposable` instances and clear the scope cache |

---

## ScopeFactory

Factory for creating scoped service providers.

```python
class ScopeFactory:
    def __init__(self, provider: ServiceProvider)
```

| Method | Description |
|--------|-------------|
| `create_scope_context() -> ScopeContext` | Returns a context manager that yields a `ServiceScope` and disposes it on exit |

### ScopeContext

```python
with scope_factory.create_scope_context() as scope:
    service = scope.get_required_service(MyService)
# scope.dispose() called automatically
```

---

## ServiceDescriptor

Describes a service registration.

```python
class ServiceDescriptor:
    def __init__(self,
                 service_type: Type,
                 implementation_type: Optional[Type] = None,
                 implementation_factory: Optional[Callable[[ServiceProvider], Any]] = None,
                 lifetime: ServiceLifetime = ServiceLifetime.SINGLETON,
                 instance: Any = None,
                 key: Optional[str] = None)
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `service_type` | `Type` | The interface or base type |
| `implementation_type` | `Type` | The concrete class (defaults to `service_type`) |
| `implementation_factory` | `Callable` | Factory function receiving `ServiceProvider` |
| `lifetime` | `ServiceLifetime` | Singleton, Scoped, or Transient |
| `instance` | `Any` | Pre-built instance (for `add_singleton_instance`) |
| `key` | `str` | Key for keyed services |

---

## ServiceLifetime

```python
class ServiceLifetime(Enum):
    SINGLETON = auto()   # One instance for the application
    SCOPED = auto()      # One instance per scope
    TRANSIENT = auto()   # New instance every time
```

---

## IConfiguration / IConfigurationSection

### IConfigurationSection (ABC)

| Property / Method | Return Type | Description |
|------------------|-------------|-------------|
| `key` | `str` | Section key |
| `path` | `str` | Full path |
| `value` | `Optional[str]` | Direct value |
| `get_section(key)` | `IConfigurationSection` | Get a child section |
| `get_children()` | `List[IConfigurationSection]` | Get immediate children |
| `get(key, default=None)` | `Any` | Get string value |
| `get_int(key, default=None)` | `Optional[int]` | Get integer |
| `get_float(key, default=None)` | `Optional[float]` | Get float |
| `get_bool(key, default=None)` | `Optional[bool]` | Get boolean |
| `get_dict(key, default=None)` | `Optional[Dict]` | Get dictionary (JSON) |
| `get_list(key, default=None)` | `Optional[List]` | Get list (JSON or comma-separated) |

### IConfiguration (extends IConfigurationSection)

| Method | Description |
|--------|-------------|
| `reload() -> None` | Reload configuration from all providers |

---

## Configuration

Implementation of `IConfiguration`.

```python
class Configuration(IConfiguration):
    def __init__(self, providers: List[ConfigurationProvider] = None)
```

Calls `reload()` on construction, merging all provider data.

---

## ConfigurationBuilder

Fluent builder for assembling configuration providers.

```python
class ConfigurationBuilder:
    def __init__(self)
```

| Method | Description |
|--------|-------------|
| `add_provider(provider: ConfigurationProvider) -> ConfigurationBuilder` | Add a custom provider |
| `add_environment_variables(prefix=None) -> ConfigurationBuilder` | Add environment variables provider |
| `add_json_file(file_path: str) -> ConfigurationBuilder` | Add JSON file provider |
| `add_in_memory_collection(initial_data: Dict[str, str]) -> ConfigurationBuilder` | Add in-memory provider |
| `add_command_line(args=None, switch_mappings=None) -> ConfigurationBuilder` | Add CLI args provider |
| `build() -> Configuration` | Build the configuration |

---

## ConfigurationSection

Implementation of `IConfigurationSection`. Provides a scoped view into a `Configuration`.

```python
class ConfigurationSection(IConfigurationSection):
    def __init__(self, configuration: Configuration, path: str, key: str = None)
```

---

## Configuration Providers

### ConfigurationProvider (ABC)

```python
class ConfigurationProvider(ABC):
    @abstractmethod
    def load(self) -> Dict[str, str]
```

### EnvironmentVariablesConfigurationProvider

```python
class EnvironmentVariablesConfigurationProvider(ConfigurationProvider):
    def __init__(self, prefix: str = None)
```

### JsonFileConfigurationProvider

```python
class JsonFileConfigurationProvider(ConfigurationProvider):
    def __init__(self, file_path: str)
```

### MemoryConfigurationProvider

```python
class MemoryConfigurationProvider(ConfigurationProvider):
    def __init__(self, initial_data: Dict[str, str] = None)
    def set(self, key: str, value: str) -> None
```

### CommandLineConfigurationProvider

```python
class CommandLineConfigurationProvider(ConfigurationProvider):
    def __init__(self, args: Optional[List[str]] = None, switch_mappings: Optional[Dict[str, str]] = None)
```

Supported formats: `--Key=Value`, `--Key Value`, `/Key=Value`, `/Key Value`.

---

## IOptions / IOptionsSnapshot / IOptionsMonitor

### IOptions (ABC) — Singleton

```python
class IOptions(ABC):
    @abstractmethod
    def get_value(self) -> T
```

Bound once at first access. Cached for application lifetime.

### IOptionsSnapshot (ABC) — Scoped

```python
class IOptionsSnapshot(ABC):
    @abstractmethod
    def get_value(self) -> T
```

Re-bound fresh each time within a scope.

### IOptionsMonitor (ABC) — Singleton, Always Fresh

```python
class IOptionsMonitor(ABC):
    @abstractmethod
    def get_current_value(self) -> T

    @abstractmethod
    def on_change(self, callback: Callable[[T], None]) -> CancellationTokenRegistration
```

---

## bind_configuration

```python
def bind_configuration(config_section: IConfigurationSection, target_type: Type[T]) -> T
```

Bind a configuration section to a dataclass or plain class. Values are coerced to annotated types.

---

## ILogger

ABC for logging services.

| Method | Description |
|--------|-------------|
| `trace(message, *args, **kwargs)` | Log at TRACE level |
| `debug(message, *args, **kwargs)` | Log at DEBUG level |
| `info(message, *args, **kwargs)` | Log at INFO level |
| `success(message, *args, **kwargs)` | Log at SUCCESS level |
| `warning(message, *args, **kwargs)` | Log at WARNING level |
| `error(message, *args, **kwargs)` | Log at ERROR level |
| `critical(message, *args, **kwargs)` | Log at CRITICAL level |
| `begin_scope(**properties) -> LogScope` | Begin enriched logging scope |

---

## LoguruLogger

Implementation of `ILogger` using loguru.

```python
class LoguruLogger(ILogger):
    def __init__(self, config: IConfiguration, context: str)
```

| Method | Description |
|--------|-------------|
| `with_context(context: str) -> LoguruLogger` | Create a new logger with a different context name |
| `begin_scope(**properties) -> LogScope` | Begin a scoped logging context |

---

## LogScope

Context manager that enriches log entries with additional properties.

```python
class LogScope:
    def __init__(self, logger_instance: LoguruLogger, properties: Dict[str, Any])
```

Usage:
```python
with logger.begin_scope(order_id="123"):
    logger.info("Processing")  # Log entry carries order_id binding
```

---

## Worker / IWorker

### IWorker (ABC)

```python
class IWorker(ABC):
    @abstractmethod
    def start(self) -> None

    @abstractmethod
    def stop(self) -> None
```

### Worker

```python
class Worker(IWorker):
    def __init__(self)
```

| Method | Description |
|--------|-------------|
| `start() -> None` | Start the worker in a background daemon thread |
| `stop() -> None` | Signal stop and wait up to 30 seconds |
| `execute() -> None` | **Abstract** — implement the work logic |
| `is_stopping() -> bool` | Check if stop has been signaled |
| `wait_for_stop(timeout_seconds=None) -> bool` | Wait for stop signal; returns `True` if signaled |

---

## TimedWorker

```python
class TimedWorker(Worker):
    def __init__(self, interval_seconds: float = 5)
```

| Method | Description |
|--------|-------------|
| `do_work() -> None` | **Abstract** — called at each interval |

`execute()` is implemented internally to call `do_work()` on the configured interval.

---

## WorkerState

```python
class WorkerState(Enum):
    CREATED = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    STOPPED = auto()
    FAILED = auto()
```

---

## WorkerManager

Manages the lifecycle of hosted workers. Created internally by `ServiceProvider`.

```python
class WorkerManager:
    def __init__(self, root_provider: ServiceProvider)
```

| Method | Description |
|--------|-------------|
| `add_service(service_type: Type[IWorker])` | Add and optionally start a worker |
| `start_all()` | Start all registered workers |
| `stop_all()` | Stop all workers in reverse order |

---

## JobManager

Runs cancellable jobs in managed threads with concurrency limits.

```python
class JobManager:
    def __init__(self, logger: ILogger, max_concurrent: int = 4)
```

| Method | Description |
|--------|-------------|
| `start_job(func, *args, name=None, daemon=False, kill_timeout=2.0, cancellation_token=None, provide_token=False, **kwargs) -> str` | Start a managed job; returns job ID |
| `wait(job_id, timeout=None) -> bool` | Wait for job completion |
| `is_running(job_id) -> bool` | Check if a job is still running |
| `cancel_job(job_id, wait=False, timeout=None) -> bool` | Cancel a job |
| `cancel_job_nowait(job_id) -> bool` | Cancel without waiting |
| `cancel_all(wait=False)` | Cancel all jobs |
| `list_jobs() -> Dict[str, Dict[str, Any]]` | List all jobs with status |
| `get_result(job_id) -> Optional[BaseException]` | Get the job's exception (or `None`) |
| `cleanup_finished() -> int` | Move finished jobs to completed map; returns count |

---

## CancellationToken

Signals cancellation to operations.

```python
class CancellationToken:
    def __init__(self, cancelled: bool = False)
```

| Property / Method | Description |
|------------------|-------------|
| `is_cancellation_requested -> bool` | Whether cancellation was requested |
| `can_be_cancelled -> bool` | Whether this token supports cancellation |
| `throw_if_cancellation_requested()` | Raise `OperationCanceledException` if cancelled |
| `register(callback) -> CancellationTokenRegistration` | Register a cancellation callback |
| `CancellationToken.none()` | Static — returns a never-cancelled token |

---

## CancellationTokenSource

Creates and controls a `CancellationToken`.

```python
class CancellationTokenSource:
    def __init__(self)
```

| Property / Method | Description |
|------------------|-------------|
| `token -> CancellationToken` | The associated token |
| `is_cancellation_requested -> bool` | Whether cancellation was requested |
| `cancel()` | Request cancellation |
| `cancel_after(delay_seconds: float)` | Schedule cancellation after delay |
| `dispose()` | Release resources |

Supports context manager (`with CancellationTokenSource() as source:`).

---

## CancellationTokenRegistration

Represents a callback registration with a cancellation token.

```python
class CancellationTokenRegistration:
    def __init__(self, unregister_func: Callable)
```

| Method | Description |
|--------|-------------|
| `dispose()` | Unregister the callback |

Supports context manager.

---

## OperationCanceledException

```python
class OperationCanceledException(Exception):
    def __init__(self, message="The operation was cancelled.", token=None)
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `token` | `Optional[CancellationToken]` | The token that triggered cancellation |

---

## Utility Functions

### validate_log_level

```python
def validate_log_level(level: str) -> str
```

Validate and return a log level string as a valid loguru level name. Handles case-insensitive matching. Raises `ValueError` for unrecognized level strings. Only accepts actual loguru levels: TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL.

### create_loguru_logger

```python
def create_loguru_logger(log_context: str, log_level: str, log_file: Optional[str])
```

Create a loguru logger bound to a specific context. Adds a stdout sink (with color) and optionally a file sink (10 MB rotation). The `log_level` is validated through `validate_log_level`.

### reset_logger_state

```python
def reset_logger_state() -> None
```

Reset loguru logger state by removing all sinks and re-initializing. Intended for use in tests to ensure clean state between test cases.

### create_linked_token

```python
def create_linked_token(*tokens: CancellationToken) -> CancellationTokenSource
```

Create a source that cancels when any of the provided tokens cancel.

### with_timeout

```python
def with_timeout(timeout_seconds: float) -> CancellationTokenSource
```

Create a source that auto-cancels after the specified timeout.

---

## IHostEnvironment

ABC providing hosting environment information.

| Property / Method | Return Type | Description |
|------------------|-------------|-------------|
| `environment_name` | `str` | Environment name (e.g., `"Production"`) |
| `application_name` | `str` | Application name |
| `content_root_path` | `str` | Content root directory |
| `is_development()` | `bool` | Check for `"development"` |
| `is_staging()` | `bool` | Check for `"staging"` |
| `is_production()` | `bool` | Check for `"production"` |

---

## HostEnvironment

Default implementation of `IHostEnvironment`.

```python
class HostEnvironment(IHostEnvironment):
    def __init__(self, environment_name="Production", application_name="Application", content_root_path=None)
```

---

## IHostApplicationLifetime

ABC for application lifetime events.

| Property / Method | Type | Description |
|------------------|------|-------------|
| `application_started` | `CancellationToken` | Signaled when app has started |
| `application_stopping` | `CancellationToken` | Signaled when shutdown begins |
| `application_stopped` | `CancellationToken` | Signaled when shutdown is complete |
| `stop_application()` | — | Request application termination |

---

## HostApplicationLifetime

Default implementation of `IHostApplicationLifetime`.

```python
class HostApplicationLifetime(IHostApplicationLifetime):
    def __init__(self)
```

Additional methods (used internally by the framework):

| Method | Description |
|--------|-------------|
| `notify_started()` | Signal that the application has started |
| `notify_stopping()` | Signal that the application is stopping |
| `notify_stopped()` | Signal that the application has stopped |

---

## IDisposable

ABC for objects holding resources that need cleanup.

```python
class IDisposable(ABC):
    @abstractmethod
    def dispose(self) -> None
```

Scoped services implementing `IDisposable` are automatically disposed when their scope ends.

---

## IMiddleware

ABC for middleware components.

```python
class IMiddleware(ABC):
    @abstractmethod
    def invoke(self, context: Dict[str, Any], next_middleware: Callable[[Dict[str, Any]], None]) -> None
```

---

## MiddlewarePipeline

Composable middleware chain.

```python
class MiddlewarePipeline:
    def __init__(self)
```

| Method | Description |
|--------|-------------|
| `use(middleware: IMiddleware) -> MiddlewarePipeline` | Add a middleware |
| `use_func(func: Callable) -> MiddlewarePipeline` | Add function-based middleware |
| `execute(context: Dict[str, Any])` | Run the pipeline |

---

## IChangeToken

ABC for change notification.

| Property / Method | Description |
|------------------|-------------|
| `has_changed -> bool` | Whether a change has occurred |
| `register_change_callback(callback) -> CancellationTokenRegistration` | Register callback |

---

## ConfigurationChangeToken

Implementation of `IChangeToken` backed by `threading.Event`.

```python
class ConfigurationChangeToken(IChangeToken):
    def __init__(self)
```

| Method | Description |
|--------|-------------|
| `signal()` | Fire the change event and invoke callbacks |
| `register_change_callback(callback) -> CancellationTokenRegistration` | Register callback |

---

## FileChangeWatcher

Monitors a file for modifications.

```python
class FileChangeWatcher:
    def __init__(self, file_path: str, poll_interval: float = 2.0)
```

| Property / Method | Description |
|------------------|-------------|
| `change_token -> ConfigurationChangeToken` | Current change token |
| `start()` | Start watching (daemon thread) |
| `stop()` | Stop watching |

---

## CliRunnerService

Runs external CLI commands under `JobManager` supervision.

```python
class CliRunnerService:
    def __init__(self, logger: ILogger, job_manager: JobManager)
```

| Method | Description |
|--------|-------------|
| `run(command, *, name=None, cwd=None, env=None, cancellation_token=None) -> str` | Run command; returns job ID. Blocks until complete. |
| `cancel(job_id, *, wait=True, timeout=10)` | Cancel a running CLI job |
