# Logging

The framework provides structured logging through `ILogger`, backed by [loguru](https://github.com/Delgan/loguru). Loggers are automatically contextualized with the consuming class name when injected.

## Table of Contents

- [ILogger Interface](#ilogger-interface)
- [Log Levels](#log-levels)
  - [Level Aliases](#level-aliases)
- [Automatic Context](#automatic-context)
- [Log Scopes](#log-scopes)
- [Configuration](#configuration)
- [Custom Logger Context](#custom-logger-context)

## ILogger Interface

`ILogger` is the logging abstraction injected into services:

```python
class MyService:
    def __init__(self, logger: ILogger):
        self.logger = logger

    def process(self):
        self.logger.info("Processing started")
        self.logger.debug("Detailed info for debugging")
        self.logger.warning("Something looks off")
        self.logger.error("An error occurred")
```

### Methods

| Method | Level | Use For |
|--------|-------|---------|
| `trace(message, *args, **kwargs)` | TRACE | Very detailed diagnostic info |
| `debug(message, *args, **kwargs)` | DEBUG | Debugging information |
| `info(message, *args, **kwargs)` | INFO | General informational messages |
| `success(message, *args, **kwargs)` | SUCCESS | Successful operation confirmations |
| `warning(message, *args, **kwargs)` | WARNING | Potentially harmful situations |
| `error(message, *args, **kwargs)` | ERROR | Error events, application continues |
| `critical(message, *args, **kwargs)` | CRITICAL | Severe errors, possible application failure |
| `begin_scope(**properties)` | — | Create enriched logging context |

## Log Levels

Levels from lowest to highest severity:

| Level | Color | Purpose |
|-------|-------|---------|
| TRACE | Dark gray | Fine-grained diagnostic events |
| DEBUG | Gray | Development-time debugging |
| INFO | White | Runtime informational messages |
| SUCCESS | Green | Operation completed successfully |
| WARNING | Yellow | Unexpected behavior, not an error |
| ERROR | Red | Failure in a specific operation |
| CRITICAL | Magenta | System-level failure |

The minimum log level is controlled by the `Logging:Level` configuration key. Only messages at or above the configured level are emitted.

### Level Aliases

Common alternate names from other logging frameworks are accepted wherever a log level string is expected (configuration values, `create_loguru_logger`, etc.). Matching is case-insensitive.

| Alias | Resolves To |
|-------|-------------|
| `WARN` | WARNING |
| `FATAL` | CRITICAL |
| `VERBOSE` | TRACE |
| `INFORMATION` | INFO |

For example, setting `Logging:Level` to `WARN` is equivalent to `WARNING`. An invalid level string raises `ValueError` with a message listing valid levels and aliases.

## Automatic Context

When `ILogger` is injected into a class, the framework automatically creates a logger with the consuming class name as context:

```python
class OrderService:
    def __init__(self, logger: ILogger):
        self.logger = logger

    def place_order(self):
        self.logger.info("Order placed")
        # Output: 2025-01-15 10:30:00 INFO     - [OrderService] Order placed

class PaymentService:
    def __init__(self, logger: ILogger):
        self.logger = logger

    def charge(self):
        self.logger.info("Payment charged")
        # Output: 2025-01-15 10:30:01 INFO     - [PaymentService] Payment charged
```

Each class gets its own logger context automatically — no manual setup needed. The context appears in brackets in the log output.

## Log Scopes

Scopes temporarily enrich all log entries with additional properties:

```python
class OrderProcessor:
    def __init__(self, logger: ILogger):
        self.logger = logger

    def process(self, order_id: str):
        with self.logger.begin_scope(order_id=order_id, customer="acme"):
            self.logger.info("Processing order")
            self.validate()
            self.logger.info("Order complete")
```

`begin_scope()` returns a context manager (`LogScope`). While inside the scope, all log entries from that logger carry the enriched properties bound via loguru's `bind()`.

Scopes can be nested — inner scopes add to the outer scope's properties.

## Configuration

Configure logging through the `Logging` configuration section:

```python
app.add_configuration_dictionary({
    "Logging": {
        "Level": "INFO",
        "File": "logs/app.log"
    }
})
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `Logging:Level` | string | `TRACE` | Minimum log level (accepts aliases — see [Level Aliases](#level-aliases)) |
| `Logging:File` | string | `None` | File path for log output (optional) |

When `Logging:File` is set, logs are written to both stdout and the file. File rotation is automatic at 10 MB.

### Log Output Format

```
{timestamp} {level:<8} - [{context}] {message}
```

Example:
```
2025-01-15 10:30:00.123 INFO     - [OrderService] Order placed successfully
2025-01-15 10:30:00.234 WARNING  - [PaymentService] Retry attempt 2/3
2025-01-15 10:30:00.345 ERROR    - [DataWorker] Connection timeout
```

## Custom Logger Context

Create a new logger instance with a different context name:

```python
class MyService:
    def __init__(self, logger: ILogger):
        self.logger = logger
        self.audit_logger = logger.with_context("Audit")

    def process(self):
        self.logger.info("Processing")              # [MyService] Processing
        self.audit_logger.info("Audit event logged") # [Audit] Audit event logged
```

`with_context(name)` returns a new `LoguruLogger` instance with the specified context string. The original logger is not modified.

## Implementation: LoguruLogger

`LoguruLogger` is the default `ILogger` implementation. It is registered as a transient service with a factory that auto-sets the context:

```python
# Registered automatically during build():
app.add_transient_factory(ILogger, lambda provider: LoguruLogger(
    provider.get_required_service(IConfiguration),
    "Default"
))
```

When injected via constructor injection, the framework detects `ILogger` and calls `with_context(ClassName)` to set the appropriate context.

### Direct Construction

For testing or manual use, create a logger directly:

```python
from application_builder import LoguruLogger, Configuration

config = Configuration()
logger = LoguruLogger(config, "MyContext")
logger.info("Hello")  # [MyContext] Hello
```
