---
applyTo: '**'
---
# Project Context

## Project Status

This project is new and unreleased. No production users or deployments exist.

---

## Breaking Changes Policy

- Backward compatibility is not a concern
- Rename, restructure, or rewrite freely
- API contracts and serialization formats may change without migration
- No deprecation warnings required

---

## Design Priority

Prioritize clean, correct design over compatibility. Fix naming mistakes and structural issues immediately. Reset test data as needed.

---

## Project Description

Python Application Builder is a lightweight dependency injection framework for Python built on clean architecture principles. Single-module architecture providing:

- IoC container with automatic constructor injection
- Service lifetimes: Singleton, Scoped, Transient
- Multi-source configuration (environment, JSON, in-memory)
- Background workers with lifecycle management
- Structured logging via loguru
- Cooperative cancellation tokens
- Job management with concurrency control
- CLI runner service

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.8+ |
| Logging | loguru 0.7.3 |
| Threading | stdlib `threading` |
| Configuration | stdlib `json`, `os.environ` |
| IDE | Visual Studio (Python Tools) |
| Project format | `.sln` + `.pyproj` |

---

## Terminology

| Term | Definition |
|------|------------|
| "rules", "the rules" | Files in `.github/instructions/` |
| "check the rules" | Read relevant `.github/instructions/*.md` files |
| "add to rules" | Create or update file in `.github/instructions/` |
| "service" | A class registered in the DI container |
| "worker" | A background service extending `Worker` or `TimedWorker` |
| "scope" | An isolated service resolution context for scoped lifetimes |
| "provider" | A `ServiceProvider` instance that resolves services |
| "descriptor" | A `ServiceDescriptor` defining a service registration |

---

## Documentation Index

| Need | Location |
|------|----------|
| API reference and usage | `README.md` |
| Architecture patterns | `.github/instructions/architecture.instructions.md` |
| Build commands | `.github/instructions/workflow.instructions.md` |
| Code conventions | `.github/instructions/code-quality.instructions.md` |
| Documentation rules | `.github/instructions/documentation.instructions.md` |
