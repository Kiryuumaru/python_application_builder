# Configuration System

The configuration system supports multiple sources with a layered override model. Later providers override earlier ones, and all values are stored as flattened colon-delimited key-value pairs.

## Table of Contents

- [Overview](#overview)
- [Configuration Providers](#configuration-providers)
- [Hierarchical Keys](#hierarchical-keys)
- [Typed Access](#typed-access)
- [Configuration Sections](#configuration-sections)
- [Typed Options Pattern](#typed-options-pattern)
- [Configuration Reload](#configuration-reload)
- [Change Tokens](#change-tokens)
- [File Watching](#file-watching)
- [Configuration Binding](#configuration-binding)

## Overview

Configuration is built from one or more providers. Each provider loads key-value pairs, and later providers override earlier ones:

```python
app = ApplicationBuilder()

# Priority: low → high (last wins)
app.add_configuration(lambda b: b.add_environment_variables("MYAPP_"))    # 1st (lowest)
app.add_configuration(lambda b: b.add_json_file("appsettings.json"))      # 2nd
app.add_configuration(lambda b: b.add_command_line())                      # 3rd
app.add_configuration_dictionary({"App": {"Debug": "true"}})              # 4th (highest)
```

Environment variables are loaded by default (with no prefix filter) when `ApplicationBuilder` is constructed.

## Configuration Providers

### Environment Variables

Reads from `os.environ`. An optional prefix filters which variables are included — the prefix is stripped from the key.

```python
app.add_configuration(lambda b: b.add_environment_variables("MYAPP_"))
```

Naming conventions:
- `__` (double underscore) maps to `:` (section separator)
- `_` (single underscore) also maps to `:`

| Environment Variable | Configuration Key |
|---------------------|-------------------|
| `MYAPP_DATABASE__HOST` | `DATABASE:HOST` |
| `MYAPP_LOGGING_LEVEL` | `LOGGING:LEVEL` |

### JSON File

Reads from a JSON file. Nested objects are flattened with `:` separators. Arrays are stored as JSON strings.

```python
app.add_configuration(lambda b: b.add_json_file("appsettings.json"))
```

Given `appsettings.json`:
```json
{
    "Database": {
        "Host": "localhost",
        "Port": 5432
    },
    "AllowedOrigins": ["http://localhost:3000", "https://example.com"]
}
```

Produces:
| Key | Value |
|-----|-------|
| `Database:Host` | `localhost` |
| `Database:Port` | `5432` |
| `AllowedOrigins` | `["http://localhost:3000", "https://example.com"]` |

If the file does not exist, an empty configuration is returned (no error).

### Command-Line Arguments

Parses command-line arguments in several formats:

```python
app.add_configuration(lambda b: b.add_command_line())
```

Supported formats:
```
--Key=Value
--Key Value
/Key=Value
/Key Value
```

A flag without a value is stored as `"true"`:
```
--Verbose  →  Verbose = "true"
```

Switch mappings allow short aliases:
```python
app.add_configuration(lambda b: b.add_command_line(
    switch_mappings={"--env": "Environment", "--db": "Database:Host"}
))
```

Underscores and double-underscores in keys are normalized to `:`.

### In-Memory Dictionary

Adds a flat dictionary of pre-set values:

```python
app.add_configuration(lambda b: b.add_in_memory_collection({
    "App:Name": "My App",
    "App:Version": "1.0.0"
}))
```

### Convenience: add_configuration_dictionary

A shortcut that flattens a nested dictionary and adds it as an in-memory provider:

```python
app.add_configuration_dictionary({
    "Database": {
        "Host": "localhost",
        "Port": 5432
    },
    "Logging": {
        "Level": "INFO"
    }
})
```

This produces the same flat keys as `Database:Host`, `Database:Port`, `Logging:Level`.

## Hierarchical Keys

All configuration values are stored as flat key-value pairs using `:` as the section delimiter:

```
Database:Host = localhost
Database:Port = 5432
Logging:Level = INFO
Logging:File = app.log
```

This hierarchy can be navigated with `get_section()` and `get_children()`.

## Typed Access

`IConfiguration` and `IConfigurationSection` provide typed getters:

```python
class MyService:
    def __init__(self, config: IConfiguration):
        # String (default)
        host = config.get("Database:Host", "localhost")

        # Integer
        port = config.get_int("Database:Port", 5432)

        # Float
        timeout = config.get_float("Service:Timeout", 30.0)

        # Boolean — recognizes: true/yes/1/on, false/no/0/off
        debug = config.get_bool("App:Debug", False)

        # List — parses JSON array or comma-separated values
        hosts = config.get_list("Security:AllowedHosts", [])

        # Dictionary — parses JSON object
        features = config.get_dict("Features", {})
```

All typed getters return the `default` if the key is missing or the value cannot be parsed.

## Configuration Sections

Sections provide a scoped view into a subtree of the configuration:

```python
class DatabaseService:
    def __init__(self, config: IConfiguration):
        db = config.get_section("Database")

        # Access keys relative to the section
        host = db.get("Host")          # Reads "Database:Host"
        port = db.get_int("Port", 5432)  # Reads "Database:Port"

        # Nested sections
        primary = db.get_section("Primary")
        conn = primary.get("ConnectionString")  # Reads "Database:Primary:ConnectionString"

        # Enumerate children
        for child in db.get_children():
            print(f"{child.key} = {child.value}")
```

### Section Properties

| Property | Description |
|----------|-------------|
| `key` | The section's own key (e.g., `"Database"`) |
| `path` | The full path (e.g., `"Database:Primary"`) |
| `value` | The section's direct value, or `None` if it has children |

## Typed Options Pattern

Bind a configuration section to a dataclass for strongly typed access. Three interfaces provide different update behaviors:

### IOptions (Singleton)

Bound once at first access. Values are cached for the application lifetime.

```python
from dataclasses import dataclass
from application_builder import ApplicationBuilder, IOptions

@dataclass
class DatabaseOptions:
    host: str = "localhost"
    port: int = 5432
    max_connections: int = 10

app = ApplicationBuilder()
app.add_configuration_dictionary({"Database": {"host": "db.example.com", "port": 3306}})
app.configure_options(DatabaseOptions, "Database")

provider = app.build()
options = provider.get_required_service(IOptions)
db_config = options.get_value()
# db_config.host == "db.example.com"
# db_config.port == 3306
# db_config.max_connections == 10  (default)
```

### IOptionsSnapshot (Scoped)

Re-bound on each scope creation. Values reflect configuration at scope creation time.

```python
scope_factory = provider.get_required_service(ScopeFactory)
with scope_factory.create_scope_context() as scope:
    snapshot = scope.get_required_service(IOptionsSnapshot)
    db_config = snapshot.get_value()
```

### IOptionsMonitor (Singleton, Always Fresh)

Always re-reads configuration on each call. Supports change callbacks.

```python
monitor = provider.get_required_service(IOptionsMonitor)
current = monitor.get_current_value()  # Always reads latest config

registration = monitor.on_change(lambda new_value: print(f"Config changed: {new_value}"))
# Later: registration.dispose() to unregister
```

### Type Coercion

The options binding system coerces string configuration values to the annotated types:

| Annotated Type | Coercion |
|---------------|----------|
| `str` | No conversion |
| `int` | `int(value)` |
| `float` | `float(value)` |
| `bool` | `true/yes/1/on` → `True`, otherwise `False` |
| `List[T]` | JSON parse, or comma-separated split |
| `Dict[K, V]` | JSON parse |

## Configuration Reload

Reload configuration from all providers at runtime:

```python
class ConfigurableService:
    def __init__(self, config: IConfiguration):
        self.config = config
        self.refresh()

    def refresh(self):
        self.config.reload()
        self.timeout = self.config.get_int("Service:Timeout", 5000)
```

`reload()` re-executes all providers in order and rebuilds the merged data. Later providers still override earlier ones.

## Change Tokens

`IChangeToken` signals that a data source has changed:

```python
from application_builder import ConfigurationChangeToken

token = ConfigurationChangeToken()

registration = token.register_change_callback(lambda: print("Configuration changed!"))

# Later, signal the change
token.signal()  # Triggers all registered callbacks

# Cleanup
registration.dispose()
```

| Property / Method | Description |
|------------------|-------------|
| `has_changed` | `True` after `signal()` is called |
| `register_change_callback(callback)` | Register a callback; returns a `CancellationTokenRegistration` for cleanup |
| `signal()` | Fire the change event and invoke all callbacks |

## File Watching

`FileChangeWatcher` monitors a file for modifications and signals a change token:

```python
from application_builder import FileChangeWatcher

watcher = FileChangeWatcher("appsettings.json", poll_interval=2.0)

registration = watcher.change_token.register_change_callback(
    lambda: print("File modified!")
)

watcher.start()
# ... later
watcher.stop()
registration.dispose()
```

The watcher polls the file's modification time at the specified interval using a background daemon thread. When a change is detected, the current `ConfigurationChangeToken` is signaled and replaced with a fresh one.

## Configuration Binding

The `bind_configuration()` function binds a configuration section to a class:

```python
from application_builder import bind_configuration

@dataclass
class SmtpSettings:
    host: str = "localhost"
    port: int = 25
    use_ssl: bool = False

section = config.get_section("Smtp")
settings = bind_configuration(section, SmtpSettings)
```

Works with both dataclasses and plain classes (reads `__init__` type hints).

## Custom Configuration Provider

Implement `ConfigurationProvider` to create a custom source:

```python
from application_builder import ConfigurationProvider

class VaultConfigurationProvider(ConfigurationProvider):
    def __init__(self, vault_url: str, token: str):
        self.vault_url = vault_url
        self.token = token

    def load(self) -> Dict[str, str]:
        # Fetch secrets from vault and return as flat key-value pairs
        return {
            "Database:Password": "secret123",
            "Api:Key": "abc-def-ghi"
        }

app.add_configuration(lambda b: b.add_provider(VaultConfigurationProvider(
    vault_url="https://vault.example.com",
    token="s.xyz"
)))
```

The `load()` method must return a `Dict[str, str]` of flattened key-value pairs.
