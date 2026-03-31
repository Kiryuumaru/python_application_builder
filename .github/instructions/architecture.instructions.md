---
applyTo: '**'
---
# Architecture Rules

## Module Structure

```
src/
├── application_builder.py    <- Core framework (single module)
├── main.py                   <- Entry point / demo
├── requirements.txt          <- Dependencies
└── python_application_builder.pyproj  <- VS project file
```

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

---

## Worker Rules

- MUST extend `Worker` or `TimedWorker`
- MUST implement `execute()` or `do_work()` respectively
- MUST check `self.is_stopping()` in loops
- MUST use `self.wait_for_stop(timeout)` instead of `time.sleep()`
- MUST handle exceptions within the execute loop
- MUST NOT block indefinitely without checking stop signal

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

| Source | Priority | Example |
|--------|----------|---------|
| Environment variables | Low (loaded first) | `APP_DATABASE__HOST` |
| JSON file | Medium | `appsettings.json` |
| In-memory dictionary | High (loaded last) | `add_configuration_dictionary()` |
