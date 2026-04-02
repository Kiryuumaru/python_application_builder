# Dependency Injection

Python Application Builder provides a full IoC (Inversion of Control) container with automatic constructor injection, multiple service lifetimes, scoping, keyed services, multi-binding, service decoration, and validation.

## Table of Contents

- [Service Registration](#service-registration)
- [Service Lifetimes](#service-lifetimes)
- [Constructor Injection](#constructor-injection)
- [Interface-Based Design](#interface-based-design)
- [Factory Registration](#factory-registration)
- [Instance Registration](#instance-registration)
- [Service Scoping](#service-scoping)
- [Multi-Binding](#multi-binding)
- [Keyed Services](#keyed-services)
- [Service Decoration](#service-decoration)
- [Conditional Registration](#conditional-registration)
- [Replace and Remove](#replace-and-remove)
- [Validation](#validation)

## Service Registration

All registration goes through `ApplicationBuilder`. Every `add_*` method returns the builder for fluent chaining:

```python
app = ApplicationBuilder()
app.add_singleton(IUserRepository, SqlUserRepository)
app.add_scoped(IRequestContext, RequestContext)
app.add_transient(EmailMessage)
```

### Registration Styles

| Style | Method | Use When |
|-------|--------|----------|
| Concrete type | `add_singleton(MyService)` | No interface needed |
| Interface + implementation | `add_singleton(IMyService, MyService)` | Coding against an abstraction |
| Factory function | `add_singleton_factory(IMyService, factory_fn)` | Complex creation logic |
| Pre-built instance | `add_singleton_instance(IMyService, instance)` | Externally created object |

## Service Lifetimes

### Singleton

One instance for the entire application lifetime. Created on first resolution and reused for all subsequent requests.

```python
app.add_singleton(DatabaseConnectionPool)
app.add_singleton(ICacheService, RedisCacheService)
app.add_singleton_factory(IStorage, create_storage)
app.add_singleton_instance(IConfig, my_config)
```

Rules:
- Must be thread-safe (shared across all threads)
- Must not hold references to scoped or transient services
- Good for: stateless services, shared state, connection pools, configuration

### Scoped

One instance per scope. A new instance is created for each `ServiceScope` and shared within that scope.

```python
app.add_scoped(IUnitOfWork, UnitOfWork)
app.add_scoped(IRequestContext, RequestContext)
app.add_scoped_factory(ISession, create_session)
```

Rules:
- Must be used within a scope context (see [Service Scoping](#service-scoping))
- Can depend on singleton services
- Must not depend on transient services
- Good for: request-level state, unit of work, database sessions

### Transient

A new instance is created every time the service is requested.

```python
app.add_transient(EmailMessage)
app.add_transient(IValidator, DataValidator)
app.add_transient_factory(ICommand, create_command)
```

Rules:
- Must be lightweight
- Must not hold expensive resources
- Good for: stateless utilities, messages, short-lived objects, commands

## Constructor Injection

The framework inspects `__init__` type hints to resolve dependencies automatically:

```python
class OrderService:
    def __init__(self, repo: IOrderRepository, logger: ILogger, email: IEmailService):
        self.repo = repo
        self.logger = logger
        self.email = email
```

When `OrderService` is resolved:
1. The container reads the type hints: `IOrderRepository`, `ILogger`, `IEmailService`
2. Each dependency is resolved from the container
3. The instance is created with all dependencies injected

If a required parameter cannot be resolved (no registration and no default value), `ValueError` is raised.

### Optional Dependencies

Use default values for optional dependencies:

```python
class MyService:
    def __init__(self, logger: ILogger, cache: ICacheService = None):
        self.logger = logger
        self.cache = cache  # None if ICacheService is not registered
```

## Interface-Based Design

Define contracts using ABCs and register implementations:

```python
from abc import ABC, abstractmethod

class INotificationService(ABC):
    @abstractmethod
    def notify(self, user_id: str, message: str) -> None:
        pass

class EmailNotificationService(INotificationService):
    def __init__(self, config: IConfiguration, logger: ILogger):
        self.smtp_host = config.get("Email:SmtpHost")
        self.logger = logger

    def notify(self, user_id: str, message: str) -> None:
        self.logger.info(f"Sending email to {user_id}: {message}")

# Registration
app.add_singleton(INotificationService, EmailNotificationService)

# Resolution — consumer only knows the interface
class OrderWorker(Worker):
    def __init__(self, notifications: INotificationService, logger: ILogger):
        super().__init__()
        self.notifications = notifications
        self.logger = logger

    def execute(self):
        self.notifications.notify("user-123", "Order confirmed")
```

## Factory Registration

For complex creation logic, register a factory function that receives the `ServiceProvider`:

```python
def create_storage(provider: ServiceProvider) -> IStorageService:
    config = provider.get_required_service(IConfiguration)
    storage_type = config.get("Storage:Type", "local")

    if storage_type == "s3":
        return S3StorageService(config)
    return LocalStorageService(config)

app.add_singleton_factory(IStorageService, create_storage)
```

Factory methods are available for all lifetimes:
- `add_singleton_factory(service_type, factory)`
- `add_scoped_factory(service_type, factory)`
- `add_transient_factory(service_type, factory)`

## Instance Registration

Register a pre-constructed object as a singleton:

```python
external_client = ExternalApiClient(api_key="abc123")
app.add_singleton_instance(IExternalClient, external_client)
```

The instance is used as-is — no constructor injection occurs.

## Service Scoping

Scopes create an isolated resolution context. Scoped services are unique per scope and disposed when the scope ends.

### Creating Scopes

```python
provider = app.build()
scope_factory = provider.get_required_service(ScopeFactory)

with scope_factory.create_scope_context() as scope:
    # Scoped services are created fresh for this scope
    unit_of_work = scope.get_required_service(IUnitOfWork)
    context = scope.get_required_service(IRequestContext)
    
    # Same instance within the scope
    same_uow = scope.get_required_service(IUnitOfWork)
    assert unit_of_work is same_uow

# Scope disposed — all IDisposable scoped services are cleaned up
```

### Scope Validation

Enable scope validation to prevent accidental resolution of scoped services from the root provider:

```python
app.set_validate_scopes(True)
provider = app.build()

# This raises RuntimeError:
provider.get_required_service(IScopedService)

# This works:
with scope_factory.create_scope_context() as scope:
    scope.get_required_service(IScopedService)
```

### IDisposable

Scoped services implementing `IDisposable` are automatically disposed when the scope ends:

```python
class DatabaseSession(IDisposable):
    def __init__(self, logger: ILogger):
        self.logger = logger

    def dispose(self) -> None:
        self.logger.info("Session closed")

app.add_scoped(DatabaseSession)

with scope_factory.create_scope_context() as scope:
    session = scope.get_required_service(DatabaseSession)
    # Use session...
# session.dispose() is called automatically here
```

## Multi-Binding

Register multiple implementations of the same interface and resolve them all with `List[T]`:

```python
class IBuildStep(ABC):
    @abstractmethod
    def execute(self) -> None: pass

class CompileStep(IBuildStep):
    def execute(self): print("Compiling...")

class TestStep(IBuildStep):
    def execute(self): print("Testing...")

class DeployStep(IBuildStep):
    def execute(self): print("Deploying...")

# Register multiple implementations
app.add_singleton(IBuildStep, CompileStep)
app.add_singleton(IBuildStep, TestStep)
app.add_singleton(IBuildStep, DeployStep)

# Inject all of them via List[T]
class BuildRunner:
    def __init__(self, steps: List[IBuildStep], logger: ILogger):
        self.steps = steps
        self.logger = logger

    def run_all(self):
        for step in self.steps:
            step.execute()
```

You can also resolve manually:

```python
all_steps = provider.get_services(IBuildStep)  # Returns List[IBuildStep]
```

## Keyed Services

Register and resolve services by a string key:

```python
app.add_keyed_singleton(IDatabase, "primary", PostgresDatabase)
app.add_keyed_singleton(IDatabase, "analytics", ClickHouseDatabase)

# Also supports factories:
app.add_keyed_singleton_factory(IDatabase, "cache", create_redis)

# Resolution
provider = app.build()
primary_db = provider.get_required_keyed_service(IDatabase, "primary")
analytics_db = provider.get_required_keyed_service(IDatabase, "analytics")

# Optional resolution (returns None if not found)
cache_db = provider.get_keyed_service(IDatabase, "cache")
```

Available keyed registration methods:
- `add_keyed_singleton(service_type, key, impl_type)`
- `add_keyed_singleton_factory(service_type, key, factory)`
- `add_keyed_scoped(service_type, key, impl_type)`
- `add_keyed_transient(service_type, key, impl_type)`

## Service Decoration

Wrap an existing service registration with additional behavior:

```python
class LoggingDecorator(IMyService):
    def __init__(self, inner: IMyService, logger: ILogger):
        self.inner = inner
        self.logger = logger

    def do_work(self):
        self.logger.info("Before work")
        self.inner.do_work()
        self.logger.info("After work")

app.add_singleton(IMyService, MyServiceImpl)
app.decorate(IMyService, lambda sp, inner: LoggingDecorator(inner, sp.get_required_service(ILogger)))
```

The decorator factory receives `(provider, inner_instance)` and returns the decorated instance. Multiple decorators can be stacked — they apply in registration order.

## Conditional Registration

### try_add

Register only if no registration exists for that service type:

```python
app.add_singleton(IMyService, PrimaryImpl)
app.try_add_singleton(IMyService, FallbackImpl)  # Skipped — IMyService already registered
```

Available for all lifetimes:
- `try_add_singleton(service_type, impl_type)`
- `try_add_scoped(service_type, impl_type)`
- `try_add_transient(service_type, impl_type)`

## Replace and Remove

### replace

Replace the first registration for a service type:

```python
app.add_singleton(IMyService, OriginalImpl)
app.replace(ServiceDescriptor(
    service_type=IMyService,
    implementation_type=ReplacementImpl,
    lifetime=ServiceLifetime.SINGLETON
))
```

### remove_all

Remove all registrations for a service type:

```python
app.remove_all(IMyService)
```

## Validation

### Build-Time Validation

Enable validation to check that all constructor parameters can be resolved at build time:

```python
app.set_validate_on_build(True)
provider = app.build()  # Raises ValueError if any dependency is unresolvable
```

This walks every registration and verifies that all `__init__` type-hinted parameters have matching registrations. Factory and instance registrations are skipped (their dependencies cannot be statically analyzed).

### Scope Validation

Prevent scoped services from being resolved outside a scope:

```python
app.set_validate_scopes(True)
```

When enabled, `get_service()` and `get_required_service()` raise `RuntimeError` if a scoped service is requested from the root provider.

## Built-In Services

These services are registered automatically by the framework during `build()`:

| Service Type | Lifetime | Description |
|-------------|----------|-------------|
| `IConfiguration` | Singleton (instance) | Application configuration |
| `ILogger` | Transient (factory) | Logger with automatic context naming |
| `JobManager` | Singleton | Background job runner |
| `CliRunnerService` | Singleton | External CLI process runner |
| `IHostEnvironment` | Singleton (instance) | Environment info (Development/Production) |
| `IHostApplicationLifetime` | Singleton (instance) | Application lifetime events |
| `MiddlewarePipeline` | Singleton (instance) | Middleware chain |
| `ScopeFactory` | Singleton (instance) | Creates service scopes |
| `ServiceProvider` | Singleton (instance) | The provider itself |

## Resolution Order

When multiple registrations exist for the same service type:
- `get_service()` and `get_required_service()` return the **last** registered implementation
- `get_services()` returns **all** registrations in registration order
