---
applyTo: '**'
---
# Workflow Rules

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
