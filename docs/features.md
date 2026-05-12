# Features

| Feature | Description |
|---------|-------------|
| **Dependency Injection** | IoC container with automatic constructor injection and multi-binding support |
| **Service Lifetimes** | Singleton, Scoped, and Transient lifetimes with scope validation |
| **Configuration** | Multi-source: environment variables, JSON files, YAML files, command-line args, in-memory dictionaries |
| **Typed Options** | Bind configuration sections to dataclasses via `IOptions` / `IOptionsSnapshot` / `IOptionsMonitor` |
| **Background Workers** | `Worker` (free-running) and `TimedWorker` (interval-based) with graceful shutdown |
| **Structured Logging** | Contextual logging backed by loguru with scoped enrichment |
| **Job Management** | `JobManager` for concurrent background tasks with cancellation and concurrency limits |
| **Cancellation Tokens** | Cooperative cancellation via `CancellationToken` / `CancellationTokenSource` |
| **Middleware Pipeline** | Composable `MiddlewarePipeline` for request/context processing |
| **Keyed Services** | Named service registrations resolved by key |
| **Service Decoration** | Wrap existing registrations with decorator factories |
| **Host Lifetime** | `IHostApplicationLifetime` events for started/stopping/stopped hooks |
| **CLI Runner** | `CliRunnerService` for running external processes under job management |
