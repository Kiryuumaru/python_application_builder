# Samples Guide

The `samples/` directory contains 20 runnable example applications, each demonstrating a specific framework feature. Every sample follows the same structure:

```
samples/<name>/
├── main.py          # Entry point — run this file
├── interfaces.py    # ABC interfaces (if needed)
├── services.py      # Service implementations (if needed)
└── workers.py       # Worker implementations
```

## Running a Sample

```bash
cd src
python ../samples/<name>/main.py
```

All samples add `src/` to `sys.path` so they can import `application_builder`.

## Sample Index

| # | Sample | Primary Feature |
|---|--------|----------------|
| 1 | [build_runner](#build_runner) | Multi-binding with `List[T]` |
| 2 | [chat_room](#chat_room) | Scoped services + multi-binding |
| 3 | [cli_args](#cli_args) | Command-line argument parsing |
| 4 | [data_pipeline](#data_pipeline) | Multiple cooperating workers |
| 5 | [decorated_services](#decorated_services) | Service decoration |
| 6 | [disposable_scopes](#disposable_scopes) | `IDisposable` + scoped cleanup |
| 7 | [env_aware](#env_aware) | `IHostEnvironment` |
| 8 | [event_bus](#event_bus) | Mixed lifetime interplay |
| 9 | [health_dashboard](#health_dashboard) | `add_singleton_instance()` |
| 10 | [inventory_cli](#inventory_cli) | Singleton + scoped together |
| 11 | [job_scheduler](#job_scheduler) | `JobManager` |
| 12 | [keyed_services](#keyed_services) | Keyed/named services |
| 13 | [lifecycle_hooks](#lifecycle_hooks) | Application lifetime events |
| 14 | [middleware_demo](#middleware_demo) | `MiddlewarePipeline` |
| 15 | [multi_config](#multi_config) | Multi-source configuration |
| 16 | [plugin_system](#plugin_system) | Factory-based plugin selection |
| 17 | [service_collection](#service_collection) | `try_add`, `replace`, `remove_all` |
| 18 | [task_queue](#task_queue) | Concurrent task processing |
| 19 | [typed_options](#typed_options) | `configure_options()` with dataclasses |
| 20 | [validated_app](#validated_app) | Build-time + scope validation |

---

## Sample Details

### build_runner

**Feature:** Multi-binding with `List[IBuildStep]` injection

Registers multiple implementations of `IBuildStep` (compile, test, deploy) and injects them all as a `List[IBuildStep]` into a runner that executes them sequentially in a pipeline.

**Key APIs:**
- `add_singleton(IBuildStep, CompileStep)` (multiple registrations for same type)
- Constructor parameter `steps: List[IBuildStep]` for multi-binding injection

**Related Docs:** [Dependency Injection — Multi-Binding](dependency-injection.md#multi-binding)

---

### chat_room

**Feature:** Scoped services and multi-binding formatters

Demonstrates per-scope state isolation with scoped `ISessionContext` and multi-binding for `List[IMessageFormatter]` to apply multiple formatters to chat messages.

**Key APIs:**
- `add_scoped(ISessionContext, SessionContext)`
- `add_singleton(IMessageFormatter, ...)` (multiple)
- `ScopeFactory.create_scope_context()`

**Related Docs:** [Dependency Injection — Service Scoping](dependency-injection.md#service-scoping)

---

### cli_args

**Feature:** Command-line argument configuration

Shows how to parse CLI arguments via `add_command_line()` with configuration priority ordering. Defaults are overridden by command-line values.

**Key APIs:**
- `add_configuration(lambda b: b.add_command_line())`
- `add_configuration_dictionary(...)` for defaults
- Configuration priority (last provider wins)

**Related Docs:** [Configuration — Command-Line Arguments](configuration.md#command-line-arguments)

---

### data_pipeline

**Feature:** Multiple concurrent workers with producer/consumer pattern

Multiple workers coordinate through shared singleton services. Demonstrates how workers run in separate threads with independent scopes while sharing singleton state.

**Key APIs:**
- `add_worker(ProducerWorker)`
- `add_worker(ConsumerWorker)`
- `add_singleton(ISharedQueue, ...)`

**Related Docs:** [Workers — Patterns](workers.md#patterns)

---

### decorated_services

**Feature:** Service decoration with `decorate()` API

Layers multiple decorators (logging, caching) around a base service implementation. The decorators wrap the original service transparently.

**Key APIs:**
- `app.decorate(IService, lambda sp, inner: LoggingDecorator(inner, sp.get_required_service(ILogger)))`
- Multiple `decorate()` calls stack in registration order

**Related Docs:** [Dependency Injection — Service Decoration](dependency-injection.md#service-decoration)

---

### disposable_scopes

**Feature:** `IDisposable` cleanup in scopes

Shows how scoped services implementing `IDisposable` are automatically disposed when the scope ends. Resources are cleaned up deterministically within the `with` block.

**Key APIs:**
- `class MyResource(IDisposable):`
- `ScopeFactory.create_scope_context()` (auto-disposes on exit)
- `ServiceScope.dispose()`

**Related Docs:** [Dependency Injection — IDisposable](dependency-injection.md#idisposable)

---

### env_aware

**Feature:** `IHostEnvironment` for environment-aware configuration

Demonstrates injecting `IHostEnvironment` to access the current environment name, application name, and content root path. Shows conditional behavior based on `is_development()` / `is_production()`.

**Key APIs:**
- `IHostEnvironment.environment_name`
- `IHostEnvironment.is_development()`
- Configuration key `Environment` or `APP_ENVIRONMENT` env var

**Related Docs:** [Advanced Features — Host Environment](advanced-features.md#host-environment)

---

### event_bus

**Feature:** Mixed service lifetime interplay

Showcases all three lifetimes working together: a singleton metrics counter, scoped event logs, and transient handlers. Each scope creates fresh scoped and transient instances while sharing the singleton.

**Key APIs:**
- `add_singleton(IMetrics, MetricsCounter)`
- `add_scoped(IEventLog, EventLog)`
- `add_transient(IEventHandler, ...)`

**Related Docs:** [Dependency Injection — Service Lifetimes](dependency-injection.md#service-lifetimes)

---

### health_dashboard

**Feature:** Pre-built singleton via `add_singleton_instance()`

Registers an externally constructed object as a singleton instance rather than having the container create it. Useful for objects that require complex initialization outside the container.

**Key APIs:**
- `add_singleton_instance(IHealthDashboard, dashboard_instance)`

**Related Docs:** [Dependency Injection — Instance Registration](dependency-injection.md#instance-registration)

---

### inventory_cli

**Feature:** Singleton + scoped services together

A shared singleton inventory is accessible across all scopes, while each scope gets its own `ICommandLog` for per-operation audit trails. Demonstrates the interaction between singleton and scoped lifetimes.

**Key APIs:**
- `add_singleton(IInventory, Inventory)`
- `add_scoped(ICommandLog, CommandLog)`
- Scoped services access singleton dependencies

**Related Docs:** [Dependency Injection — Service Lifetimes](dependency-injection.md#service-lifetimes)

---

### job_scheduler

**Feature:** `JobManager` for background task execution

Shows how to use `JobManager` to run concurrent background tasks with lifecycle tracking. Includes a job runner that schedules work and a reporter that monitors job status.

**Key APIs:**
- `job_manager.start_job(func, ...)`
- `job_manager.list_jobs()`
- `job_manager.wait(job_id)`
- `job_manager.cancel_job(job_id)`

**Related Docs:** [Advanced Features — Job Management](advanced-features.md#job-management)

---

### keyed_services

**Feature:** Keyed/named service resolution

Registers multiple implementations of the same interface under different string keys. Resolves specific implementations by key at runtime.

**Key APIs:**
- `add_keyed_singleton(IDatabase, "primary", PostgresDatabase)`
- `provider.get_required_keyed_service(IDatabase, "primary")`

**Related Docs:** [Dependency Injection — Keyed Services](dependency-injection.md#keyed-services)

---

### lifecycle_hooks

**Feature:** `IHostApplicationLifetime` events

Hooks into application startup and shutdown sequences using `IHostApplicationLifetime` tokens. Registers callbacks for started, stopping, and stopped events.

**Key APIs:**
- `lifetime.application_started.register(callback)`
- `lifetime.application_stopping.register(callback)`
- `lifetime.application_stopped.register(callback)`

**Related Docs:** [Advanced Features — Host Application Lifetime](advanced-features.md#host-application-lifetime)

---

### middleware_demo

**Feature:** `MiddlewarePipeline` composition

Workers process requests through a configured middleware chain. Demonstrates both class-based (`IMiddleware`) and function-based middleware.

**Key APIs:**
- `pipeline.use(MyMiddleware())`
- `pipeline.use_func(lambda ctx, next_mw: ...)`
- `pipeline.execute(context)`

**Related Docs:** [Advanced Features — Middleware Pipeline](advanced-features.md#middleware-pipeline)

---

### multi_config

**Feature:** Multi-source configuration with priority ordering

Loads configuration from a JSON file, environment variables (with prefix), and an in-memory dictionary. Demonstrates that later providers override earlier ones.

**Key APIs:**
- `add_configuration(lambda b: b.add_json_file("appsettings.json"))`
- `add_configuration(lambda b: b.add_environment_variables("MYAPP_"))`
- `add_configuration_dictionary({...})`

**Related Docs:** [Configuration — Overview](configuration.md#overview)

---

### plugin_system

**Feature:** Factory-based dynamic plugin selection

Uses `add_singleton_factory()` to choose a plugin implementation at runtime based on configuration values. The factory inspects `IConfiguration` to decide which concrete class to instantiate.

**Key APIs:**
- `add_singleton_factory(IPlugin, create_plugin)`
- Factory reads config to select implementation

**Related Docs:** [Dependency Injection — Factory Registration](dependency-injection.md#factory-registration)

---

### service_collection

**Feature:** Service collection manipulation APIs

Demonstrates `try_add_singleton()` (conditional), multi-binding (multiple registrations), `replace()` (swap implementation), and `remove_all()` (clear registrations).

**Key APIs:**
- `try_add_singleton(IService, Impl)` — only if not registered
- `replace(ServiceDescriptor(...))` — swap first match
- `remove_all(IService)` — clear all

**Related Docs:** [Dependency Injection — Conditional Registration](dependency-injection.md#conditional-registration), [Replace and Remove](dependency-injection.md#replace-and-remove)

---

### task_queue

**Feature:** Concurrent task processing with concurrency limits

A task queue where a producer enqueues work items and consumer workers process them concurrently. Demonstrates configurable concurrency through `JobManager`.

**Key APIs:**
- `job_manager.start_job(process_task, ...)`
- Concurrency limits via `max_concurrent` parameter

**Related Docs:** [Advanced Features — Job Management](advanced-features.md#job-management)

---

### typed_options

**Feature:** `configure_options()` for typed dataclass binding

Binds a configuration section to a Python dataclass. Registers `IOptions` (singleton), `IOptionsSnapshot` (scoped), and `IOptionsMonitor` (always-fresh) automatically.

**Key APIs:**
- `@dataclass class AppOptions: ...`
- `app.configure_options(AppOptions, "App")`
- `provider.get_required_service(IOptions).get_value()`

**Related Docs:** [Configuration — Typed Options Pattern](configuration.md#typed-options-pattern)

---

### validated_app

**Feature:** Build-time and scope validation

Enables `set_validate_on_build()` to catch missing dependencies at build time and `set_validate_scopes()` to prevent scoped services from being resolved outside a scope. Shows how validation errors are surfaced.

**Key APIs:**
- `app.set_validate_on_build(True)`
- `app.set_validate_scopes(True)`
- `provider.validate_all_registrations()`

**Related Docs:** [Dependency Injection — Validation](dependency-injection.md#validation)

---

## Feature Cross-Reference

| Framework Feature | Samples |
|------------------|---------|
| `add_singleton` | Most samples |
| `add_scoped` | chat_room, disposable_scopes, inventory_cli, event_bus |
| `add_transient` | event_bus |
| `add_singleton_instance` | health_dashboard |
| `add_singleton_factory` | plugin_system |
| `add_keyed_singleton` | keyed_services |
| `add_worker` | All samples (primary entry point) |
| `add_configuration_dictionary` | Most samples |
| `add_configuration` (JSON) | multi_config |
| `add_configuration` (CLI) | cli_args |
| `add_configuration` (env) | multi_config, env_aware |
| `configure_options` | typed_options |
| `decorate` | decorated_services |
| `try_add_singleton` | service_collection |
| `replace` | service_collection |
| `remove_all` | service_collection |
| `set_validate_on_build` | validated_app |
| `set_validate_scopes` | validated_app |
| `ScopeFactory` | chat_room, disposable_scopes, inventory_cli, event_bus |
| `IDisposable` | disposable_scopes |
| `IHostEnvironment` | env_aware |
| `IHostApplicationLifetime` | lifecycle_hooks |
| `JobManager` | job_scheduler, task_queue |
| `MiddlewarePipeline` | middleware_demo |
| `List[T]` multi-binding | build_runner, chat_room, service_collection |
