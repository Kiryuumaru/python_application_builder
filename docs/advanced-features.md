# Advanced Features

This document covers cancellation tokens, job management, the middleware pipeline, the CLI runner service, and host lifetime events.

## Table of Contents

- [Cancellation Tokens](#cancellation-tokens)
- [Job Management](#job-management)
- [Middleware Pipeline](#middleware-pipeline)
- [CLI Runner Service](#cli-runner-service)
- [Host Environment](#host-environment)
- [Host Application Lifetime](#host-application-lifetime)

## Cancellation Tokens

Cooperative cancellation using the `CancellationToken` / `CancellationTokenSource` pattern. A source controls the token; consumers check the token for cancellation.

### CancellationTokenSource

The source creates and controls a token:

```python
from application_builder import CancellationTokenSource

source = CancellationTokenSource()
token = source.token

# Check cancellation
print(token.is_cancellation_requested)  # False

# Cancel
source.cancel()
print(token.is_cancellation_requested)  # True

# Cleanup
source.dispose()
```

#### Timed Cancellation

```python
source = CancellationTokenSource()
source.cancel_after(10.0)  # Auto-cancels after 10 seconds
```

Or use the convenience function:

```python
from application_builder import with_timeout

source = with_timeout(10.0)
token = source.token
# token.is_cancellation_requested becomes True after 10 seconds
```

#### Context Manager

```python
with CancellationTokenSource() as source:
    token = source.token
    # ... use token
# source.dispose() called automatically
```

### CancellationToken

The token is the read-only side consumers use:

```python
def do_work(token: CancellationToken):
    while not token.is_cancellation_requested:
        process_next_item()

    # Or throw:
    token.throw_if_cancellation_requested()
    # Raises OperationCanceledException
```

| Property / Method | Description |
|------------------|-------------|
| `is_cancellation_requested` | `True` when cancellation has been signaled |
| `can_be_cancelled` | `True` for normal tokens, `False` for `CancellationToken.none()` |
| `throw_if_cancellation_requested()` | Raises `OperationCanceledException` if cancelled |
| `register(callback)` | Register a callback; returns `CancellationTokenRegistration` |
| `CancellationToken.none()` | Static — returns a token that can never be cancelled |

### Callbacks

Register callbacks that fire when cancellation is requested:

```python
def on_cancel():
    print("Cancellation requested — cleaning up")

registration = token.register(on_cancel)

# Later, unregister:
registration.dispose()
```

If the token is already cancelled when `register()` is called, the callback executes immediately.

### Linked Tokens

Create a source that cancels when any of several tokens cancel:

```python
from application_builder import create_linked_token

linked_source = create_linked_token(token_a, token_b)
linked_token = linked_source.token

# If either token_a or token_b cancels, linked_token also cancels
```

### OperationCanceledException

Thrown by `throw_if_cancellation_requested()`:

```python
from application_builder import OperationCanceledException

try:
    token.throw_if_cancellation_requested()
except OperationCanceledException as e:
    print(f"Cancelled: {e}")
    print(f"Token: {e.token}")
```

### CancellationTokenRegistration

Represents a callback registration. Supports context manager:

```python
with token.register(my_callback) as registration:
    do_work()
# Callback unregistered automatically
```

## Job Management

`JobManager` runs functions in managed threads with concurrency limits, cooperative cancellation, and lifecycle tracking.

### Getting the JobManager

It is registered as a singleton automatically:

```python
provider = app.build()
job_manager = provider.get_required_service(JobManager)
```

Or inject it via constructor:

```python
class MyService:
    def __init__(self, job_manager: JobManager, logger: ILogger):
        self.jobs = job_manager
        self.logger = logger
```

### Starting Jobs

```python
def heavy_computation(x, y):
    return x + y

job_id = job_manager.start_job(heavy_computation, 10, 20, name="add-numbers")
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `func` | `Callable` | required | The function to execute |
| `*args` | positional | — | Arguments passed to `func` |
| `name` | `str` | auto-generated | Human-readable job name |
| `daemon` | `bool` | `False` | Whether the thread is a daemon |
| `kill_timeout` | `float` | `2.0` | Timeout when force-cancelling |
| `cancellation_token` | `CancellationToken` | `None` | External token to link to |
| `provide_token` | `bool` | `False` | If `True`, inject a `CancellationToken` as the first argument |
| `**kwargs` | keyword | — | Keyword arguments passed to `func` |

#### Token Injection

When `provide_token=True`, the job's internal cancellation token is passed as the first argument:

```python
def cancellable_work(token: CancellationToken, data):
    while not token.is_cancellation_requested:
        process(data)

job_id = job_manager.start_job(cancellable_work, my_data, provide_token=True)
```

### Waiting and Querying

```python
# Wait for completion (blocking)
completed = job_manager.wait(job_id, timeout=30.0)

# Check if running
running = job_manager.is_running(job_id)

# Get result (exception or None)
exception = job_manager.get_result(job_id)
if exception:
    print(f"Job failed: {exception}")

# List all jobs
for jid, info in job_manager.list_jobs().items():
    print(f"{info['name']}: {info['status']}")
```

### Cancellation

```python
# Cancel and wait for completion
job_manager.cancel_job(job_id, wait=True, timeout=5.0)

# Cancel without waiting
job_manager.cancel_job_nowait(job_id)

# Cancel all jobs
job_manager.cancel_all(wait=True)
```

Cancellation signals the job's internal `CancellationTokenSource`. The job function must check `token.is_cancellation_requested` (when `provide_token=True`) to respond cooperatively.

### Concurrency Control

The `max_concurrent` parameter limits how many jobs run simultaneously:

```python
# Constructor injection with default limit
class JobManager:
    def __init__(self, logger: ILogger, max_concurrent: int = 4):
        ...
```

When the limit is reached, `start_job()` blocks until a slot becomes available.

### Cleanup

```python
cleaned = job_manager.cleanup_finished()
print(f"Cleaned up {cleaned} finished jobs")
```

## Middleware Pipeline

A composable pipeline for processing context dictionaries through a chain of middleware:

### IMiddleware Interface

```python
from abc import ABC, abstractmethod
from application_builder import IMiddleware

class LoggingMiddleware(IMiddleware):
    def invoke(self, context: dict, next_middleware):
        print(f"Before: {context}")
        next_middleware(context)
        print(f"After: {context}")

class AuthMiddleware(IMiddleware):
    def invoke(self, context: dict, next_middleware):
        if context.get("authenticated"):
            next_middleware(context)
        else:
            context["error"] = "Unauthorized"
```

### Building a Pipeline

```python
from application_builder import MiddlewarePipeline

pipeline = MiddlewarePipeline()
pipeline.use(LoggingMiddleware())
pipeline.use(AuthMiddleware())
pipeline.use(BusinessLogicMiddleware())

# Execute
context = {"request": "data", "authenticated": True}
pipeline.execute(context)
```

### Function-Based Middleware

For quick middleware without a class:

```python
pipeline.use_func(lambda ctx, next_mw: (
    ctx.update({"timestamp": time.time()}) or next_mw(ctx)
))
```

### Pipeline Behavior

- Middleware executes in registration order
- Each middleware decides whether to call `next_middleware(context)` to continue the chain
- If a middleware doesn't call `next_middleware`, the pipeline short-circuits
- `use()` and `use_func()` return the pipeline for fluent chaining

### Framework Integration

A `MiddlewarePipeline` is registered automatically as a singleton:

```python
class MyWorker(Worker):
    def __init__(self, pipeline: MiddlewarePipeline, logger: ILogger):
        super().__init__()
        self.pipeline = pipeline
        self.logger = logger

    def execute(self):
        self.pipeline.use(MyMiddleware())
        self.pipeline.execute({"data": "value"})
```

## CLI Runner Service

`CliRunnerService` runs external commands as managed jobs with cancellation support:

```python
class DeployWorker(Worker):
    def __init__(self, cli: CliRunnerService, logger: ILogger):
        super().__init__()
        self.cli = cli
        self.logger = logger

    def execute(self):
        job_id = self.cli.run(
            command=["python", "-m", "pytest"],
            name="run-tests",
            cwd="/path/to/project"
        )
        self.logger.info(f"Tests completed: {job_id}")
```

### run() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `command` | `Sequence[str]` | required | Command and arguments |
| `name` | `str` | `None` | Job name |
| `cwd` | `str` | `None` | Working directory |
| `env` | `Dict[str, str]` | `None` | Environment variables |
| `cancellation_token` | `CancellationToken` | `None` | Token for external cancellation |

### Cancellation

```python
from application_builder import CancellationTokenSource

source = CancellationTokenSource()
source.cancel_after(60.0)  # 60-second timeout

try:
    cli.run(["long-running-command"], cancellation_token=source.token)
except OperationCanceledException:
    print("Command was cancelled")
```

When cancelled, the process receives SIGTERM first, then SIGKILL after a 5-second grace period.

### Cancel a Running Job

```python
job_id = cli.run(["sleep", "3600"], name="long-job")

# In another thread:
cli.cancel(job_id, wait=True, timeout=10)
```

## Host Environment

`IHostEnvironment` provides information about the hosting environment:

```python
class MyService:
    def __init__(self, env: IHostEnvironment, logger: ILogger):
        self.env = env
        self.logger = logger

    def configure(self):
        if self.env.is_development():
            self.logger.info("Running in development mode")
        elif self.env.is_production():
            self.logger.info("Running in production mode")

        self.logger.info(f"App: {self.env.application_name}")
        self.logger.info(f"Root: {self.env.content_root_path}")
```

### Properties

| Property | Default | Configuration Key |
|----------|---------|-------------------|
| `environment_name` | `"Production"` | `Environment` or `APP_ENVIRONMENT` env var |
| `application_name` | `"Application"` | `ApplicationName` |
| `content_root_path` | `os.getcwd()` | `ContentRoot` |

### Helper Methods

| Method | Returns `True` when |
|--------|-------------------|
| `is_development()` | `environment_name` is `"development"` (case-insensitive) |
| `is_staging()` | `environment_name` is `"staging"` |
| `is_production()` | `environment_name` is `"production"` |

## Host Application Lifetime

`IHostApplicationLifetime` provides tokens that signal application lifecycle events:

```python
class StartupService:
    def __init__(self, lifetime: IHostApplicationLifetime, logger: ILogger):
        self.lifetime = lifetime
        self.logger = logger

        lifetime.application_started.register(lambda: self.logger.info("App started"))
        lifetime.application_stopping.register(lambda: self.logger.info("App stopping"))
        lifetime.application_stopped.register(lambda: self.logger.info("App stopped"))
```

### Lifetime Tokens

| Property | Signaled When |
|----------|---------------|
| `application_started` | `CancellationToken` signaled after `build()` completes and workers start |
| `application_stopping` | `CancellationToken` signaled when shutdown begins (SIGINT/SIGTERM) |
| `application_stopped` | `CancellationToken` signaled after all workers have stopped |

### Programmatic Shutdown

Request application shutdown from anywhere:

```python
class ShutdownWorker(Worker):
    def __init__(self, lifetime: IHostApplicationLifetime, logger: ILogger):
        super().__init__()
        self.lifetime = lifetime
        self.logger = logger

    def execute(self):
        self.do_work()
        self.logger.info("Work complete, requesting shutdown")
        self.lifetime.stop_application()
```

`stop_application()` signals the stopping token, which triggers the shutdown sequence in `run()`.
