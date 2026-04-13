# Worker System

Workers are background services that run in dedicated threads. The framework manages their lifecycle — starting them after the service provider is built and stopping them gracefully on shutdown.

## Table of Contents

- [Worker Base Class](#worker-base-class)
- [TimedWorker](#timedworker)
- [Registration](#registration)
- [Lifecycle](#lifecycle)
- [Worker States](#worker-states)
- [Graceful Shutdown](#graceful-shutdown)
- [Error Handling](#error-handling)
- [Workers and Scoping](#workers-and-scoping)
- [Patterns](#patterns)

## Worker Base Class

Extend `Worker` and implement `execute()`:

```python
from application_builder import Worker, ILogger

class DataProcessingWorker(Worker):
    def __init__(self, logger: ILogger):
        super().__init__()
        self.logger = logger

    def execute(self):
        self.logger.info("Worker started")

        while not self.is_stopping():
            try:
                self.process_batch()
            except Exception as e:
                self.logger.error(f"Error: {e}")
                self.wait_for_stop(10.0)

    def process_batch(self):
        # Work logic here
        self.wait_for_stop(1.0)  # Pause between batches
```

### Key Methods

| Method | Description |
|--------|-------------|
| `execute()` | Abstract — implement your work loop here |
| `is_stopping()` | Returns `True` when the worker has been asked to stop |
| `stopping_token` | Property — `CancellationToken` cancelled when the worker stops |
| `wait_for_stop(timeout)` | Blocks up to `timeout` seconds; returns `True` if stop was signaled |
| `start()` | Starts the worker in a background thread (called by framework) |
| `stop()` | Signals the worker to stop and waits up to 30 seconds for completion |

### Rules

- Always call `super().__init__()` in your constructor
- Check `is_stopping()` in loops to respond to shutdown
- Use `wait_for_stop(timeout)` instead of `time.sleep()` — it returns immediately when stop is signaled
- Catch exceptions inside your work loop to prevent thread death
- Never block indefinitely without checking the stop signal

## TimedWorker

For work that runs on a fixed interval, extend `TimedWorker` and implement `do_work()`:

```python
from application_builder import TimedWorker, ILogger

class HealthCheckWorker(TimedWorker):
    def __init__(self, logger: ILogger):
        super().__init__(interval_seconds=30)
        self.logger = logger

    def do_work(self):
        self.logger.info("Running health check")
        # Check health...
```

`TimedWorker` handles the loop and timing automatically:
1. Calls `do_work()`
2. Calculates how long `do_work()` took
3. Waits the remaining interval (or immediately re-executes if `do_work()` took longer than the interval)
4. Repeats until stopped

If `do_work()` raises an exception, it is logged and the timer continues.

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `interval_seconds` | `float` | `5` | Seconds between executions |

## Registration

Register workers with `add_worker()`:

```python
app = ApplicationBuilder()
app.add_worker(DataProcessingWorker)
app.add_worker(HealthCheckWorker)
app.run()
```

`add_worker()` performs two registrations:
1. The worker type as a singleton: `add_singleton(WorkerType)`
2. The worker type under `IWorker`: `add_singleton(IWorker, WorkerType)`

Each worker is created in its own scope, so constructor dependencies are resolved independently per worker.

## Lifecycle

### Startup

When `build(auto_start_hosted_services=True)` is called (the default):

1. The `ServiceProvider` discovers all `IWorker` registrations
2. A `WorkerManager` creates a scope for each worker and resolves it
3. Each worker's `start()` is called, launching a daemon thread
4. The thread calls `execute()` (or `TimedWorker`'s internal loop)

### Shutdown

When `run()` receives SIGINT/SIGTERM, or when `stop_hosted_services()` is called:

1. Each worker's `stop()` is called (in reverse registration order)
2. `stop()` sets the stop event and waits up to 30 seconds for the thread to finish
3. Workers that check `is_stopping()` exit their loops
4. If a worker doesn't stop within 30 seconds, the framework proceeds (daemon threads are killed on process exit)

### Manual Control

```python
provider = app.build(auto_start_hosted_services=False)

# Start when ready
provider.start_hosted_services()

# ... later
provider.stop_hosted_services()
```

## Worker States

Workers progress through these states:

```
CREATED → STARTING → RUNNING → STOPPING → STOPPED
                         ↓
                       FAILED
```

| State | Description |
|-------|-------------|
| `CREATED` | Worker constructed, not yet started |
| `STARTING` | `start()` called, thread launching |
| `RUNNING` | `execute()` is running |
| `STOPPING` | `stop()` called, waiting for `execute()` to return |
| `STOPPED` | `execute()` returned normally |
| `FAILED` | `execute()` threw an unhandled exception |

## Graceful Shutdown

The `wait_for_stop()` / `is_stopping()` pattern enables cooperative shutdown:

```python
class GracefulWorker(Worker):
    def __init__(self, logger: ILogger):
        super().__init__()
        self.logger = logger

    def execute(self):
        while not self.is_stopping():
            item = self.get_next_item()

            if item is None:
                # No work — wait, but wake up for stop signal
                self.wait_for_stop(5.0)
                continue

            # Process the item (check stopping between expensive steps)
            self.step_one(item)
            if self.is_stopping():
                break
            self.step_two(item)

        self.logger.info("Worker shut down gracefully")
```

### wait_for_stop vs time.sleep

| `wait_for_stop(5.0)` | `time.sleep(5.0)` |
|----------------------|-------------------|
| Returns immediately when stop is signaled | Always sleeps the full 5 seconds |
| Returns `True` if stop was signaled | No stop awareness |
| Enables fast shutdown | Delays shutdown by up to 5 seconds |

## Error Handling

Always catch exceptions inside your work loop:

```python
class RobustWorker(Worker):
    def __init__(self, service: IMyService, logger: ILogger):
        super().__init__()
        self.service = service
        self.logger = logger

    def execute(self):
        while not self.is_stopping():
            try:
                self.service.process()
            except Exception as e:
                self.logger.error(f"Processing failed: {e}")
                # Back off before retrying
                self.wait_for_stop(10.0)
```

If an unhandled exception escapes `execute()`, the worker transitions to `FAILED` state and the exception is logged by the framework. The thread dies, but other workers continue.

For `TimedWorker`, exceptions in `do_work()` are caught and logged automatically — the timer continues running.

## Workers and Scoping

Each worker is resolved in its own scope. This means:
- Workers get their own instances of scoped services
- Workers can safely hold references to singleton services
- Two workers that depend on the same scoped service get different instances

```python
class WorkerA(Worker):
    def __init__(self, context: IScopedContext, logger: ILogger):
        super().__init__()
        self.context = context  # Unique to WorkerA's scope
        self.logger = logger

    def execute(self):
        self.logger.info(f"Context ID: {self.context.id}")

class WorkerB(Worker):
    def __init__(self, context: IScopedContext, logger: ILogger):
        super().__init__()
        self.context = context  # Different instance — WorkerB's scope
        self.logger = logger

    def execute(self):
        self.logger.info(f"Context ID: {self.context.id}")
```

## Patterns

### Long-Running Processing Loop

```python
class QueueWorker(Worker):
    def __init__(self, queue: IMessageQueue, logger: ILogger):
        super().__init__()
        self.queue = queue
        self.logger = logger

    def execute(self):
        while not self.is_stopping():
            message = self.queue.dequeue(timeout=1.0)
            if message:
                try:
                    self.process(message)
                    self.queue.acknowledge(message)
                except Exception as e:
                    self.logger.error(f"Failed to process {message.id}: {e}")
                    self.queue.reject(message)
            else:
                self.wait_for_stop(0.5)
```

### One-Shot Worker

A worker that does its work and exits:

```python
class MigrationWorker(Worker):
    def __init__(self, db: IDatabaseService, logger: ILogger):
        super().__init__()
        self.db = db
        self.logger = logger

    def execute(self):
        self.logger.info("Running migrations...")
        self.db.run_migrations()
        self.logger.info("Migrations complete")
        # execute() returns — worker transitions to STOPPED
```

### Multiple Cooperating Workers

```python
class ProducerWorker(Worker):
    def __init__(self, queue: ISharedQueue, logger: ILogger):
        super().__init__()
        self.queue = queue
        self.logger = logger

    def execute(self):
        while not self.is_stopping():
            data = self.fetch_data()
            if data:
                self.queue.enqueue(data)
            self.wait_for_stop(1.0)

class ConsumerWorker(Worker):
    def __init__(self, queue: ISharedQueue, processor: IDataProcessor, logger: ILogger):
        super().__init__()
        self.queue = queue
        self.processor = processor
        self.logger = logger

    def execute(self):
        while not self.is_stopping():
            item = self.queue.dequeue(timeout=0.5)
            if item:
                self.processor.process(item)

app.add_singleton(ISharedQueue, ThreadSafeQueue)
app.add_worker(ProducerWorker)
app.add_worker(ConsumerWorker)
```

### Cascading Cancellation to Child Jobs

Use `stopping_token` to propagate the worker's stop signal to jobs started via `JobManager`:

```python
class FrameStreamWorker(TimedWorker):
    def __init__(self, job_manager: JobManager, logger: ILogger):
        super().__init__(interval_seconds=2)
        self.job_manager = job_manager
        self.logger = logger

    def do_work(self):
        self.job_manager.start_job(
            self._process_frame,
            cancellation_token=self.stopping_token,
        )

    def _process_frame(self):
        self.logger.info("Processing frame...")
```

When the worker stops, `stopping_token` is cancelled, which auto-cancels all child jobs that were given the token.
