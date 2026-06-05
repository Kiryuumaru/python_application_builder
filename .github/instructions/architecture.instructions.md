---
applyTo: '**'
---
# Architecture Rules

## Repository Structure

The framework itself is a single-module library. Applications built on top of the framework MUST follow a layered structure.

### Framework Source (this repository)

```
src/
+-- application_builder.py    <- Core framework (single module)
+-- requirements.txt          <- Dependencies
```

### Application Project Structure (consumers of the framework)

```
src/
+-- domain/                           <- Pure business logic, no I/O
|   +-- shared/
|   |   +-- interfaces/               <- IUnitOfWork, IAggregateRoot, IDomainEvent
|   |   +-- models/                   <- Entity, AggregateRoot, ValueObject bases
|   |   +-- constants/
|   |   +-- exceptions/
|   +-- {feature}/
|       +-- entities/
|       +-- value_objects/
|       +-- models/                   <- Non-entity domain types
|       +-- enums/
|       +-- events/                   <- Domain events
|       +-- interfaces/               <- I{Feature}Repository, I{Feature}UnitOfWork
|       +-- constants/
|       +-- services/                 <- Pure domain services
|       +-- exceptions/
+-- application/                      <- Orchestration, references domain only
|   +-- shared/
|   |   +-- interfaces/               <- IDomainEventDispatcher, IDomainEventHandler
|   |   +-- models/
|   |   +-- primitives/
|   |   +-- services/                 <- DomainEventDispatcher
|   +-- {feature}/
|       +-- interfaces/               <- I{Feature}Service, I{External}Provider
|       +-- services/                 <- Service implementations
|       +-- models/
|       +-- workers/                  <- Background workers
|       +-- validators/
|       +-- event_handlers/
+-- infrastructure/                   <- External integrations, references application/domain
|   +-- {provider}/                   <- e.g. sqlite, http
|   |   +-- adapters/                 <- Implements application interfaces
|   |   +-- repositories/             <- Implements domain repository interfaces
|   |   +-- configurations/
|   |   +-- services/
|   |   +-- models/
|   +-- {provider}_{feature}/         <- e.g. sqlite_identity
|       +-- adapters/
|       +-- repositories/
|       +-- configurations/
+-- presentation/                     <- Drives the app (composition root)
|   +-- main.py                       <- ONLY module that imports infrastructure
|   +-- commands/                     <- CLI commands
|   +-- controllers/                  <- HTTP/incoming adapters
|   +-- workers/                      <- Presentation-only workers
|   +-- models/
+-- requirements.txt
```

Each folder is a Python package containing `__init__.py` plus one file per public type (see `One Type Per File`). See `Layer Folder Structures` below for per-layer details.

---

## Framework Subsystems

```
+-------------------------------------------+
|           APPLICATION LAYER               |
|  Workers, Services, Entry Points          |
|  Uses: ApplicationBuilder, ILogger,       |
|        IConfiguration, Worker             |
+-------------------------------------------+
                    |
                    v
+-------------------------------------------+
|        DEPENDENCY INJECTION CORE          |
|  ServiceProvider, ServiceDescriptor       |
|  ServiceScope, ScopeFactory               |
|  Lifetimes: Singleton, Scoped, Transient  |
+-------------------------------------------+
                    |
                    v
+-------------------------------------------+
|         INFRASTRUCTURE SERVICES           |
|  Configuration, Logging, JobManager       |
|  CancellationToken, CliRunnerService      |
+-------------------------------------------+
```

---

## Subsystem Dependencies

```
ApplicationBuilder
  ^
ServiceProvider / ServiceScope
  ^
Configuration    Logging    JobManager    CancellationToken
  ^                ^            ^               ^
Providers      LoguruLogger  ThreadJobs    TokenSource
```

---

## Adding a New Feature

```
1. Define abstractions
   ├── Create ABC interface in application_builder.py
   └── Define abstract methods
            ↓
2. Implement concrete class
   ├── Implement the ABC
   └── Accept dependencies via __init__ type hints
            ↓
3. Register with ApplicationBuilder
   ├── app.add_singleton(IService, ConcreteService)
   ├── app.add_scoped(IService, ConcreteService)
   └── app.add_transient(IService, ConcreteService)
            ↓
4. Consume via constructor injection
   ├── Declare typed parameter in __init__
   └── Framework resolves automatically
```

Key principle: **ABCs define contracts → Concrete classes implement → ApplicationBuilder wires → ServiceProvider resolves**

---

## Adding a New Feature (Layered)

When adding a feature in an application built on the framework:

```
1. domain/{feature}/
   ├── Create entities/, value_objects/, models/, enums/, events/
   ├── Define I{Feature}Repository in interfaces/i_{feature}_repository.py
   └── Define I{Feature}UnitOfWork in interfaces/i_{feature}_unit_of_work.py
            ↓
2. application/{feature}/
   ├── Define I{Feature}Service in interfaces/i_{feature}_service.py
   ├── Define I{External}Provider in interfaces/i_{external}_provider.py (if needed)
   ├── Implement service in services/{feature}_service.py
   └── Expose register(builder) function in __init__.py
            ↓
3. infrastructure/{provider}_{feature}/
   ├── Implement repository in repositories/{provider}_{feature}_repository.py
   ├── Implement adapter in adapters/{provider}_{external}_adapter.py
   └── Expose register(builder) function in __init__.py
            ↓
4. presentation/
   ├── Add command/controller (incoming adapter)
   └── Call layer register(builder) functions in main.py
```

Key principle: **Domain defines contracts → Application orchestrates → Infrastructure implements → Presentation drives**

---

## Service Lifetime Rules

Singleton:
- MUST be thread-safe
- MUST NOT hold references to scoped or transient services
- USE for stateless services, configuration, shared state

Scoped:
- MUST be used within a scope context
- MAY hold references to singleton services
- MUST NOT hold references to transient services
- USE for request-level or unit-of-work services

Transient:
- MUST be lightweight
- MUST NOT hold expensive resources
- USE for stateless utilities, messages, short-lived objects
- MUST propagate `CancellationToken` to all downstream blocking calls when accepted

---

## Worker Rules

- MUST extend `Worker` or `TimedWorker`
- MUST implement `execute()` or `do_work()` respectively
- MUST check `self.is_stopping()` in loops
- MUST use `self.wait_for_stop(timeout)` instead of `time.sleep()`
- MUST handle exceptions within the execute loop
- MUST NOT block indefinitely without checking stop signal
- MUST pass `self.stopping_token` to any downstream blocking calls (e.g. HTTP clients, DB queries, sleeps)

---

## Interface Design Rules

- MUST use `ABC` and `@abstractmethod` for service contracts
- MUST define interfaces before implementations
- MUST accept dependencies via `__init__` parameters with type hints
- MUST NOT use global state or module-level singletons for service resolution
- MUST NOT import concrete implementations in consumer code when an interface exists

---

## Configuration Key Convention

- USE colon-delimited hierarchical keys: `Section:SubSection:Key`
- USE `get_section()` for grouping related settings
- Environment variables USE double-underscore `__` or single-underscore `_` as separator
- Later configuration providers override earlier ones
- PREFER YAML files (`appsettings.yaml`) over JSON for file-based configuration
- USE JSON only when YAML is not supported by the toolchain

YAML is the preferred configuration format because it supports inline comments (`#`) and is designed as a human-authored configuration language. JSON forbids comments by spec, which forces operators to either remove documentation before edits or maintain parallel documentation outside the config file. For settings that humans read and edit, comment support is a hard requirement.

| Source | Priority | Example |
|--------|----------|---------|
| Environment variables | Low (loaded first) | `APP_DATABASE__HOST` |
| YAML file (preferred) | Medium | `appsettings.yaml` |
| JSON file (fallback) | Medium | `appsettings.json` |
| In-memory dictionary | High (loaded last) | `add_configuration_dictionary()` |

---

## Layered Architecture (Application Code)

The framework itself is a single module (`src/application_builder.py`). Applications built on top of the framework MUST follow a layered architecture inspired by Clean Architecture / Hexagonal Architecture.

```
+-------------------------------------------+
|            PRESENTATION                   |
|  References: application, domain          |
|  Infrastructure: main.py only             |
+-------------------------------------------+
                    |
                    v
+-------------------------------------------+
|           INFRASTRUCTURE                  |
|  References: application, domain          |
|  Implements: Domain repository interfaces |
|  Implements: Application interfaces       |
+-------------------------------------------+
                    |
                    v
+-------------------------------------------+
|            APPLICATION                    |
|  References: domain only                  |
|  Defines: interfaces, services, workers   |
+-------------------------------------------+
                    |
                    v
+-------------------------------------------+
|              DOMAIN                       |
|  References: nothing (pure logic)         |
|  Contains: entities, value objects, models|
+-------------------------------------------+
```

---

## Layer Reference Rules

Domain:
- MUST NOT import from `application`, `infrastructure`, or `presentation`
- MUST NOT depend on third-party frameworks (loguru, yaml, http clients, ORMs)
- MAY depend only on the standard library and `application_builder` ABCs/utilities
- MUST NOT use framework decorators on entities or value objects

Application:
- MUST only import from `domain`
- MUST NOT import from `infrastructure` or `presentation`
- MAY import `application_builder` abstractions (`ILogger`, `IConfiguration`, `Worker`, `CancellationToken`)
- Defines interfaces, services, and workers

Infrastructure:
- MUST import from `application` and `domain`
- MUST NOT import from `presentation`
- Implements repository interfaces declared in Domain
- Implements interfaces declared in Application that require external resources

Presentation:
- MUST import from `application` and `domain`
- MUST NOT import from `infrastructure` except in `main.py`
- Drives the application by calling Application interfaces

---

## Layer Folder Structures

### Domain Layer

```
domain/
+-- __init__.py
+-- shared/
|   +-- __init__.py
|   +-- interfaces/                  <- IAggregateRoot, IDomainEvent, IEntity, IUnitOfWork
|   +-- models/                      <- Entity, AggregateRoot, ValueObject, DomainEvent
|   +-- constants/                   <- EmptyCollections and shared constants
|   +-- exceptions/                  <- DomainException, ValidationException
+-- {feature}/
    +-- __init__.py
    +-- entities/
    +-- value_objects/
    +-- models/                      <- Plain dataclasses / records
    +-- enums/
    +-- events/
    +-- interfaces/                  <- I{Feature}Repository, I{Feature}UnitOfWork
    +-- constants/
    +-- services/                    <- Pure domain services
    +-- exceptions/
```

### Application Layer

```
application/
+-- __init__.py
+-- shared/
|   +-- __init__.py
|   +-- interfaces/                  <- Shared application interfaces (IDomainEventDispatcher, IDomainEventHandler)
|   +-- models/
|   +-- primitives/                  <- Instantiable utility classes
|   +-- services/                    <- e.g. DomainEventDispatcher
+-- {feature}/
    +-- __init__.py
    +-- interfaces/                  <- I{Feature}Service, I{External}Provider, internal abstractions
    +-- services/                    <- Service implementations
    +-- models/
    +-- workers/                     <- Background workers
    +-- validators/
    +-- event_handlers/
```

### Infrastructure Layer

```
infrastructure/
+-- __init__.py
+-- {provider}/                      <- e.g. sqlite, postgres, http
|   +-- __init__.py
|   +-- adapters/                    <- Adapter implementations of application interfaces
|   +-- repositories/                <- Domain repository implementations
|   +-- services/
|   +-- configurations/
|   +-- models/
+-- {provider}_{feature}/            <- e.g. sqlite_identity
    +-- __init__.py
    +-- adapters/
    +-- repositories/
    +-- configurations/
```

### Presentation Layer

```
presentation/
+-- __init__.py
+-- main.py                          <- Composition root
+-- commands/                        <- CLI commands
+-- controllers/                     <- HTTP/incoming adapters
+-- workers/                         <- Presentation-only workers
+-- middleware/
+-- models/
```

---

## Interfaces

All ABC interfaces for a feature live in a single `interfaces/` package per layer. There is no inbound/outbound split.

Interface rules:
- Domain interfaces (repositories, UnitOfWork, marker interfaces) live in `domain/{feature}/interfaces/` and `domain/shared/interfaces/`
- Application interfaces (services, providers, internal abstractions) live in `application/{feature}/interfaces/` and `application/shared/interfaces/`
- Domain interfaces MUST be implemented in Infrastructure
- Application interfaces MAY be implemented in Application or Infrastructure
- Any layer MAY call any interface; access is controlled by import direction, not by interface kind
- One ABC per file; file name is the snake_case form of the interface name (e.g. `IOrderService` -> `i_order_service.py`)
- Examples: `IOrderService`, `IIdentityService`, `IEmailSender`, `IDateTimeProvider`, `IDomainEventDispatcher`, `IOrderRepository`

### Interface Access

| Interface Location | Who Can Use | Who Implements |
|--------------------|-------------|----------------|
| `domain/shared/interfaces/` | Application, Infrastructure, Presentation | Infrastructure (except markers) |
| `domain/{feature}/interfaces/` | Application, Infrastructure, Presentation | Infrastructure |
| `application/shared/interfaces/` | Application, Infrastructure, Presentation | Application or Infrastructure |
| `application/{feature}/interfaces/` | Application, Infrastructure, Presentation | Application or Infrastructure |

---

## Adapters

Incoming adapters:
- Drive the application
- Live in Presentation layer
- Call Application interfaces (services)
- Examples: CLI commands, controllers, presentation workers

Outgoing adapters:
- Are driven by the application
- Live in Infrastructure layer
- Implement Domain interfaces (repositories, UnitOfWork)
- Implement Application interfaces that require external resources
- Examples: Repositories, HTTP clients, message bus clients

---

## Entry Points vs Services

Entry points:
- Are initiators of application logic
- Create scopes and resolve services
- MUST NOT be injected
- Examples: CLI commands, controllers, application workers, queue consumers

Services:
- Are injectable units of logic
- Are registered with a DI lifetime
- Are resolved by entry points or other services
- Examples: domain services, application services, repositories

---

## Service Categories

### Domain Services

Domain services:
- Contain pure business logic
- MUST NOT perform I/O
- MUST NOT depend on repositories, HTTP clients, or external services
- MUST receive all required data as method parameters
- MUST be registered as Singleton
- Are stateless calculators

### Application Services

Application services:
- Orchestrate business operations
- Implement application interfaces
- Call Domain services for business logic
- Call infrastructure interfaces for I/O
- Coordinate transactions through `IUnitOfWork`
- Are typically Scoped when depending on UnitOfWork or per-request state
- MAY be Singleton if stateless

---

## Unit of Work Pattern

Unit of Work coordinates persistence and domain event dispatch within an atomicity boundary.

### Base Interface

`IUnitOfWork` lives in `domain/shared/interfaces/i_unit_of_work.py`:
- Defines the `commit()` contract
- MUST NOT be implemented directly by Infrastructure
- MUST be inherited by feature-specific UnitOfWork interfaces

### Feature-Specific Interfaces

Each feature defines its own UnitOfWork interface in `domain/{feature}/interfaces/`:
- MUST inherit from `IUnitOfWork`
- Declares the atomicity boundary for that feature
- Is implemented by Infrastructure adapters

```python
# domain/shared/interfaces/i_unit_of_work.py
class IUnitOfWork(ABC):
    @abstractmethod
    def commit(self, cancellation_token: CancellationToken) -> None: ...

# domain/trading/interfaces/i_trading_unit_of_work.py
class ITradingUnitOfWork(IUnitOfWork): ...

# domain/identity/interfaces/i_identity_unit_of_work.py
class IIdentityUnitOfWork(IUnitOfWork): ...
```

### UnitOfWork Rules

- Base `IUnitOfWork` MUST be in `domain/shared/interfaces/`
- Feature UnitOfWork interfaces MUST be in `domain/{feature}/interfaces/`
- Feature UnitOfWork interfaces MUST inherit from `IUnitOfWork`
- Infrastructure MUST implement feature-specific interfaces, NOT base `IUnitOfWork`
- Services MUST inject feature-specific interfaces, NOT base `IUnitOfWork`
- Each UnitOfWork defines one atomicity boundary

---

## Cancellation Discipline

All methods performing I/O, network calls, file operations, sleeps, or any blocking/waiting logic MUST accept a `CancellationToken` parameter and propagate it to every downstream blocking call.

### Required Approach

- MUST accept `cancellation_token: CancellationToken` on any method that performs I/O, network calls, file operations, sleeps, or blocking waits
- MUST propagate the token to every downstream blocking call: `await asyncio.sleep(..., cancel=token)`, `requests.get(..., timeout=token)`, DB queries, HTTP calls, file reads/writes
- MUST pass the token through the entire call chain — a service calling a repository MUST accept a token and forward it
- MUST check `token.is_cancellation_requested` before entering blocking operations and after each await point
- MUST use `token.register(callback)` for cleanup on cancellation

### Prohibited

- NEVER call blocking I/O without a cancellation token
- NEVER fire-and-forget a loop without checking `is_cancellation_requested`
- NEVER swallow `CancelledError` or `CancellationRequested` without cleanup
- NEVER create a `CancellationToken.none()` as a default to silence the requirement — if a caller cannot provide a token, the method is not cancellable and MUST be documented as such

---

## Domain Base Models

Base models live in `domain/shared/models/`.

| Base Class | Purpose |
|------------|---------|
| `Entity` | Identity with `id` and `rev_id` |
| `AggregateRoot` | Entity with domain events |
| `AuditableEntity` | Entity with `created` / `last_modified` |
| `ValueObject` | Immutable equality-by-components |

---

## Domain Type Categories

| Category | Location | Characteristics |
|----------|----------|-----------------|
| Entities | `entities/` | Extends `Entity`/`AggregateRoot`, has identity, mutable state |
| ValueObjects | `value_objects/` | Extends `ValueObject`, immutable, equality by components |
| Models | `models/` | Plain classes/dataclasses, no base class required |

When to use each:
- USE Entities for things with unique identity tracked over time
- USE ValueObjects for immutable concepts defined by their attributes
- USE Models for domain data structures that fit neither

---

## Entity Implementation

Entities use a private `__init__` and classmethod factories for creation.

```python
class OrderEntity(AggregateRoot):
    def __init__(self, id: UUID, customer_name: str, total: Decimal):
        super().__init__(id)
        self._customer_name = customer_name
        self._total = total

    @property
    def customer_name(self) -> str:
        return self._customer_name

    @property
    def total(self) -> Decimal:
        return self._total

    @classmethod
    def create(cls, customer_name: str, total: Decimal) -> "OrderEntity":
        if not customer_name:
            raise ValueError("customer_name required")
        entity = cls(uuid4(), customer_name, total)
        entity.add_domain_event(OrderCreatedEvent(entity.id))
        return entity
```

Entity implementation rules:
- MUST extend `Entity`, `AggregateRoot`, or `AuditableEntity`
- MUST expose state as read-only `@property`
- MUST store state in private attributes (`_attribute`)
- MUST use `classmethod` factories for business creation
- MUST raise domain events only in factory methods
- MUST validate parameters only in factory methods
- MUST NOT add logic in `__init__` beyond assignment

---

## ValueObject Implementation

```python
class Money(ValueObject):
    def __init__(self, amount: Decimal, currency: str):
        self._amount = amount
        self._currency = currency

    @property
    def amount(self) -> Decimal:
        return self._amount

    @property
    def currency(self) -> str:
        return self._currency

    @classmethod
    def create(cls, amount: Decimal, currency: str) -> "Money":
        if not currency:
            raise ValueError("currency cannot be empty")
        return cls(amount, currency)

    def _equality_components(self) -> Tuple[object, ...]:
        return (self._amount, self._currency)
```

ValueObject implementation rules:
- MUST extend `ValueObject`
- MUST override `_equality_components()`
- MUST expose state via read-only `@property`
- MUST use `classmethod` factories for business creation
- MUST validate parameters only in factory methods
- MUST NOT have an `id` attribute
- MUST NOT have mutable state

---

## Domain Exceptions

```
Exception
  └── DomainException             <- Base for all domain errors
        ├── EntityNotFoundException
        └── ValidationException
```

| Exception | Purpose | Attributes |
|-----------|---------|------------|
| `DomainException` | Base domain error | - |
| `EntityNotFoundException` | Entity not found | `entity_type`, `entity_identifier` |
| `ValidationException` | Validation failure | `property_name` |

Exception placement:
- Base exceptions MUST be in `domain/shared/exceptions/`
- Feature exceptions MUST be in `domain/{feature}/exceptions/`
- Feature exceptions MUST extend `DomainException`

---

## Domain Events

### Event Flow

1. Entity raises event via `add_domain_event()`
2. UnitOfWork commit triggers dispatch (post-commit)
3. Handlers execute independently

### Domain Layer Types

- `IDomainEvent` — marker interface in `domain/shared/interfaces/`
- `IAggregateRoot` — marker interface in `domain/shared/interfaces/`
- `DomainEvent` — base class in `domain/shared/models/`
- `AggregateRoot` — base entity with `add_domain_event()` in `domain/shared/models/`
- `{Entity}{Action}Event` — concrete events in `domain/{feature}/events/`

### Application Layer Types

- `IDomainEventHandler` — interface in `application/shared/interfaces/`
- `IDomainEventDispatcher` — interface in `application/shared/interfaces/`
- `DomainEventHandler` — base in `application/shared/models/`
- `DomainEventDispatcher` — default implementation in `application/shared/services/`
- `{Action}Handler` — concrete handlers in `application/{feature}/event_handlers/`

### Naming Conventions

| Type | Pattern | Example |
|------|---------|---------|
| Event | `{Entity}{Action}Event` | `UserCreatedEvent`, `OrderPlacedEvent` |
| Handler | `{Action}Handler` | `SendWelcomeEmailHandler` |

### Handler Rules

- MUST extend `DomainEventHandler[TEvent]`
- MUST be registered as `IDomainEventHandler` in DI
- MUST be independent (no order dependencies)
- MUST handle their own exceptions

### Dispatch Rules

- MUST occur after successful commit (post-commit)
- MUST dispatch all events from all modified aggregates
- MUST clear events from entities after dispatch

---

## Workers by Layer

Application workers:
- Are background tasks for business logic
- ACT by calling services; are not called by services
- Own their loop and decide WHEN to run
- MUST extend `Worker` or `TimedWorker`
- MUST be registered via `add_hosted_service()`
- Live in `application/{feature}/workers/`

Presentation workers:
- Are rare — only for UI-only background tasks
- MUST NOT contain business logic
- Live in `presentation/workers/`
- PREFER application workers for business-related work

---

## Composition Root

Infrastructure wiring occurs only in `main.py` of an executable application.

```
presentation/main.py            <- Composition root
```

Services, commands, controllers, and workers MUST NOT import from `infrastructure.*`.

---

## Service Accessibility

Python uses naming conventions instead of `public`/`internal`:

- Interfaces are public — exported via `__all__`
- Implementations are module-private — prefix with `_` OR not listed in `__all__`
- Domain types (entities, value objects, events) are public

```python
# application/identity/interfaces/i_identity_service.py
__all__ = ["IIdentityService"]

class IIdentityService(ABC): ...

# application/identity/services/_identity_service.py
__all__ = []  # implementations resolved via DI, not imported by name

class _IdentityService(IIdentityService): ...
```

---

## DI Lifetime Rules by Layer

Domain services:
- MUST be Singleton
- Are stateless pure-logic calculators

Application services:
- MAY be Singleton, Scoped, or Transient
- MUST be Scoped when depending on UnitOfWork or per-request state

Infrastructure adapters:
- Are typically Scoped (DB connections, HTTP clients)
- MAY be Singleton when stateless (clocks, factories)

Captive dependency rule:
- Singleton MUST NOT inject Scoped or Transient
- Scoped MUST NOT inject Transient
- Transient MAY inject anything

---

## Infrastructure Separation

Providers (sqlite, postgres) and Features (identity, trading) are independent.

- Provider modules MAY reference Application and Domain
- Provider modules MUST NOT reference Feature modules
- Feature modules MAY reference Application and Domain
- Feature modules MUST NOT reference Provider modules
- Composition happens only in `main.py`

Naming convention: `infrastructure/{provider}_{feature}/` for provider-feature combinations.

---

## Ignorance Principles

- Application and Domain MUST have no knowledge of databases, HTTP, or message brokers
- Application MUST have no knowledge of specific external services (Stripe, Binance, etc.)
- Domain MUST use a single entity with a Type enum, not separate entities per variant
- Application class names MUST NOT reference infrastructure (e.g. `LocalStoreTokenStorage`, not `SqliteTokenStorage`)
- Infrastructure class names MUST reflect the implementation (e.g. `SqliteOrderRepository`)

---

## Search Before Create

Before creating any type, utility, or pattern:

1. MUST search the codebase for existing types with similar purpose
2. MUST check these locations in order:
   - `domain/shared/` for domain primitives and interfaces
   - `domain/{feature}/value_objects/` for value types
   - `domain/{feature}/services/` for domain logic
   - `application/shared/models/` for shared DTOs
   - `application/{feature}/models/` for feature DTOs
3. If found: MUST use or extend it
4. If not found: MUST create in the appropriate shared location

---

## One Concept, One Type, One Location

- If the same type is defined in multiple files, MUST keep one and delete others
- If the same concept has different names, MUST consolidate to a single canonical name
- If a private type could be shared, MUST move to a shared module
- If an anonymous dict is used for a known concept, MUST use the existing named type

---

## No Duplication

MUST extract when:
- Same logic appears 2+ times
- Same pattern emerges across files
- Same constant value used in multiple locations
- Same error handling repeated

---

## Consolidation Workflow

When duplicates are discovered:

1. MUST identify canonical location based on layer rules (prefer `domain/shared/`, `domain/{feature}/value_objects/`, or `application/shared/models/`)
2. MUST keep the most complete definition
3. MUST update all references to use the shared type
4. MUST delete duplicate definitions
5. MUST run tests to verify nothing is broken

---

## Constants Over Magic Values

Every literal value used more than once MUST be a named constant.

- Shared constants MUST be in `domain/shared/constants/`
- Domain feature constants MUST be in `domain/{feature}/constants/`
- Application settings MUST come from `IConfiguration` or `application/{feature}/constants/`
- Test values MUST be in a test base class or constants module

### EmptyCollections Pattern

`domain/shared/constants/` provides shared empty collection instances:

```python
class EmptyCollections:
    STRING_LIST: Tuple[str, ...] = ()
    UUID_LIST: Tuple[UUID, ...] = ()
    STRING_DICT: Mapping[str, str] = MappingProxyType({})
```

---

## One Type Per File

Each `.py` file contains exactly one public type. File name is the `snake_case` form of the type name.

| Type Kind | File Name Format | Example |
|-----------|------------------|---------|
| Class | `{snake_case_name}.py` | `user_service.py` for `UserService` |
| ABC interface | `i_{snake_case_name}.py` | `i_user_service.py` for `IUserService` |
| Dataclass / record | `{snake_case_name}.py` | `login_request.py` for `LoginRequest` |
| Enum | `{snake_case_name}.py` | `order_status.py` for `OrderStatus` |

Exceptions:
- Module-private helpers (prefix `_`) MAY live in the file that uses them
- Small private nested classes MAY live in the file of the enclosing public type
- A package's `__init__.py` MAY re-export the package's public types

Each folder MUST contain `__init__.py`. Re-exporting from `__init__.py` is encouraged for public APIs:

```python
# application/identity/interfaces/__init__.py
from application.identity.interfaces.i_identity_service import IIdentityService
from application.identity.interfaces.i_token_provider import ITokenProvider

__all__ = ["IIdentityService", "ITokenProvider"]
```

---

## Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Modules / packages | `snake_case` | `event_bus`, `identity` |
| Classes | `PascalCase` | `UserService`, `LoginRequest` |
| Interfaces (ABCs) | `I` + `PascalCase` | `IUserService`, `ITokenProvider` |
| Functions / methods | `snake_case` | `get_user`, `validate_token` |
| Properties | `snake_case` | `user_id`, `is_authenticated` |
| Private attributes | `_snake_case` | `_user_service`, `_logger` |
| Local variables | `snake_case` | `user_id`, `token_result` |
| Constants | `UPPER_SNAKE_CASE` | `DEFAULT_TIMEOUT`, `MAX_RETRIES` |
| Async functions | suffix `_async` only when sync sibling exists | `get_user_async` |

---

## Member Ordering Within Classes

1. Class constants
2. Class fields
3. `__init__`
4. Other dunder methods
5. Properties
6. Public methods
7. Private methods (`_method`)

---

## Module Path Convention

Import path mirrors folder path from `src/`.

```
src/application/identity/services/identity_service.py
-> import path: application.identity.services.identity_service
```

---

## Type Placement by Folder

Files MUST be placed in the folder matching their type kind. If the folder does not exist within the feature, create it.

| Type Kind | Required Folder | Example |
|-----------|-----------------|---------|
| ABC interface | `interfaces/` | `interfaces/i_order_service.py` |
| Enum | `enums/` | `enums/order_status.py` |
| Dataclass / DTO / model | `models/` | `models/login_request.py` |
| Service class | `services/` | `services/user_service.py` |
| Adapter class | `adapters/` | `adapters/binance_exchange_adapter.py` |
| Repository class | `repositories/` | `repositories/order_repository.py` |
| Entity | `entities/` | `entities/user.py` |
| Value object | `value_objects/` | `value_objects/email.py` |
| Exception | `exceptions/` | `exceptions/user_not_found_exception.py` |
| Validator | `validators/` | `validators/login_request_validator.py` |
| Configuration class | `configurations/` | `configurations/user_configuration.py` |
| Constants class / values | `constants/` | `constants/error_messages.py` |
| Worker (application) | `workers/` | `workers/trade_filler.py` |
| Worker (presentation) | `workers/` | `workers/trade_filler_host.py` |
| Domain event | `events/` | `events/order_placed_event.py` |
| Event handler | `event_handlers/` | `event_handlers/send_welcome_email_handler.py` |
| Primitive class | `primitives/` | `primitives/gate_keeper.py` |
| Utility module | `utils/` | `utils/network_utils.py` |
| Helper module | `helpers/` | `helpers/string_helpers.py` |
| Middleware | `middleware/` | `middleware/auth_middleware.py` |
| Command | `commands/` | `commands/main_command.py` |
| Controller | `controllers/` | `controllers/orders_controller.py` |
| Composition root | project root | `main.py` |

Verification:
- Before creating a file, identify its type kind
- Place in the corresponding folder within the feature
- If folder does not exist, create it with an `__init__.py`

---

## Primitives, Utilities, and Helpers

### Primitives

Primitives are instantiable utility classes:
- Created with the class constructor
- Hold state or manage resources
- Live in `primitives/` folder within the relevant package
- Examples: `GateKeeper`, `AsyncManualResetEvent`, `LazyValue`

### Utilities

Utilities are stateless module-level functions:
- Called directly without instantiation
- Live in `utils/` folder within the relevant package
- Examples: `network_utils`, `random_helpers`, `task_utils`

### Helpers

Module-level helper functions that operate on existing types:
- Live in `helpers/` folder within the relevant package
- Replace C# extension methods (Python has no extension method syntax)

---

## FAQ / Edge Cases

**Q: Where do I put a service used by multiple features?**
A: `application/shared/services/`.

**Q: What if Infrastructure needs to call another Infrastructure adapter?**
A: Define an interface in `application/{feature}/interfaces/` and have both adapters depend on it. Infrastructure adapters MUST NOT import each other directly.

**Q: Where do constants shared across all layers go?**
A: `domain/shared/constants/`. Domain is referenced by all layers.

**Q: Can a domain service call a repository?**
A: No. Domain services are pure logic with no I/O. Pass data as method parameters. Application services orchestrate repositories and domain services.

**Q: Where do DTOs for external API responses go?**
A: `infrastructure/{provider}/models/`. They are implementation details of the adapter.

**Q: Can Presentation call any application interface?**
A: Yes. There is no inbound/outbound split; access is governed by import direction. Presentation imports from `application` and `domain` only.

---

## Layer Registration

Python lacks C# extension methods and assembly-discovery patterns. Each layer MUST expose a single module-level `register` function that wires its services into an `ApplicationBuilder`.

Layer registration rules:
- Each layer / feature module MUST expose `register(builder: ApplicationBuilder) -> None`
- `main.py` MUST call each layer's `register` in dependency order: domain → application → infrastructure → presentation
- `register` MUST only call `builder.add_singleton`, `add_scoped`, `add_transient`, or `add_hosted_service`
- `register` MUST NOT instantiate services directly
- `register` MUST NOT perform I/O
- Feature `register` functions MAY be composed inside a layer-level `register`

```python
# application/identity/__init__.py
from application_builder import ApplicationBuilder
from application.identity.interfaces.i_identity_service import IIdentityService
from application.identity.services.identity_service import _IdentityService

def register(builder: ApplicationBuilder) -> None:
    builder.add_scoped(IIdentityService, _IdentityService)

# presentation/main.py
import domain, application, infrastructure, presentation

builder = ApplicationBuilder()
domain.register(builder)
application.register(builder)
infrastructure.register(builder)
presentation.register(builder)
provider = builder.build()
```

---

## Composition Root Module

`presentation/main.py` is the only module permitted to import from `infrastructure.*`.

Composition root rules:
- MUST live at `presentation/main.py`
- MUST be the only entry point that imports infrastructure modules
- MUST construct `ApplicationBuilder` and call `register` for each layer
- MUST start workers via `WorkerManager` after `build()`
- MUST NOT contain business logic
- MUST handle top-level cancellation with `CancellationTokenSource`

---

## CLI Command Exit

One-shot CLI commands MUST cancel the application's `CancellationTokenSource` after completing their work to allow graceful exit.

Long-running commands (servers, watchers, daemons) MUST NOT cancel manually and MUST allow external cancellation (Ctrl+C, signal) to stop the application.

```python
def run(provider: ServiceProvider, cts: CancellationTokenSource) -> None:
    service = provider.get_service(IExportService)
    service.export()
    cts.cancel()
```

---

## Prohibited Patterns

- NEVER import from `infrastructure.*` outside `main.py`
- NEVER use concrete types in `__init__` parameter type hints when an ABC exists
- NEVER use third-party framework decorators on domain entities
- NEVER use module-level singletons or service locators for service resolution
- NEVER hardcode connection strings or URLs in Application
- NEVER use `if isinstance(x, ConcreteType)` branching in Application layer
- NEVER implement application interfaces in Domain
- NEVER call Infrastructure directly from Presentation (except DI registration in `main.py`)
- NEVER place business logic in Presentation workers
- NEVER place I/O in Domain services
- NEVER expose internal interfaces to Presentation
- NEVER inject workers into other services
- NEVER copy-paste with minor variations
- NEVER use inline magic values
- NEVER duplicate validation logic
- NEVER use `dict` when a named dataclass exists for the concept
- NEVER place shared types in feature-specific modules when multiple features need them
- NEVER use file names that do not match the module's content category
- NEVER place a type in a module that does not match its kind
- NEVER use `public set` equivalents (mutable public attributes) on entities
- NEVER use parameterless factory creation (entity must be created via `classmethod` factory)
- NEVER raise domain events in `__init__`
- NEVER validate in `__init__`
- NEVER add logic in `__init__` beyond assignment
- NEVER add `id` to a `ValueObject`
- NEVER use mutable attributes in `ValueObject`
- NEVER use bare `except:` — always specify exception type
- NEVER use `time.sleep()` in workers — use `wait_for_stop()`
- NEVER perform blocking I/O, network calls, file operations, or sleeps without a `CancellationToken`
