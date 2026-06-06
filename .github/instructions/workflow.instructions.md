---
applyTo: '**'
---
# Workflow Rules

## Session Start — Data Flow Mapping

Before any planning, coding, or implementation begins in a session, map the data flow first. The AI must understand how data moves through the codebase before making any changes.

1. **Identify the source** — Where does the data originate? (user input, database, external API, event stream)
2. **Identify the destination** — Where does the data end up? (database, response, external API, event bus)
3. **Identify the transformations** — What changes the data? (validation, business logic, enrichment, filtering)
4. **Identify the state owners** — Which layer owns the mutable state at each step?
5. **Check layer boundaries** — Does the flow respect the existing architecture rules? (Domain → Application → Infrastructure → Presentation)

Output a brief outline (bullet points, not a doc) before generating code. Paste this map into the prompt to give the AI rails:

> "Here's the exact data flow, generate code that strictly follows it. Don't try to introduce new entities, state, or flows unless I ask."

Structure comes first. Speed is second. Real speed is not generating 500 lines in 10 seconds, it's not spending the next 3 hours deleting them.

---

## Build Commands

### Python Commands

| Command | Description |
|---------|-------------|
| `pip install -r requirements.txt` | Install dependencies |
| `python main.py` | Run application |
| `python -m pytest` | Run all tests |
| `python -m pytest -k "test_name"` | Run specific test |
| `python -m pytest -v` | Run tests with verbose output |
| `python -m pytest --tb=short` | Run tests with short traceback |

### Linting Commands

| Command | Description |
|---------|-------------|
| `python -m py_compile application_builder.py` | Check syntax |
| `python -m py_compile main.py` | Check syntax |

### Working Directory

All commands MUST be run from `src/` directory.

```
cd src
```

---

## First-Time Setup

1. `pip install -r requirements.txt` - Install dependencies
2. `python main.py` - Verify application runs

---

## Dependency Management

- MUST pin dependency versions in `requirements.txt`
- MUST use `pip install -r requirements.txt` for reproducible installs
- MUST NOT install packages globally without explicit reason

| Dependency | Purpose |
|------------|---------|
| `loguru==0.7.3` | Structured logging |
| `pyyaml==6.0.3` | YAML configuration support |

---

## Pre-Commit Verification

Before every commit:

| Check | Command | Required Result |
|-------|---------|-----------------|
| Syntax | `python -m py_compile application_builder.py` | No errors |
| Run | `python main.py` | Runs without crash |
| Tests | `python -m pytest` | 100% pass |

### Manual Review Checklist

- MUST follow architecture rules in `architecture.instructions.md`
- MUST follow code quality rules in `code-quality.instructions.md`
- MUST verify type hints on all public method signatures
- MUST verify ABC interfaces have `@abstractmethod` decorators
- MUST update documentation per `documentation.instructions.md`
