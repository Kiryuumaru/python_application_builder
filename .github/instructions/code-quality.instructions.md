---
applyTo: '**'
---
# Code Quality Rules

## Fix Hygiene

**Failed fix = full revert.**

- NEVER leave non-working code from failed fix attempts
- NEVER apply fix B on top of failed fix A
- NEVER comment out failed code instead of removing

---

## Zero Warnings

Code MUST pass linting with 0 warnings and 0 errors. Every warning is a potential bug.

---

## None Handling

- For possibly None values, USE explicit checks: `if x is None: raise ValueError()`
- For optional parameters, DECLARE with `Optional[T]` and provide defaults
- For required dependencies, RAISE `ValueError` if resolution fails
- MUST NOT use bare `except:` — always specify exception type
- MUST NOT silently swallow None — check and handle explicitly

---

## Type Hints

- MUST use type hints on all function signatures
- MUST use type hints on `__init__` parameters for DI resolution
- MUST use `Optional[T]` for nullable parameters
- MUST use `List[T]` for multi-service injection
- MUST import from `typing` module: `Any`, `Callable`, `Dict`, `List`, `Optional`, `Type`, `TypeVar`

---

## Reliability Principles

- MUST validate at boundaries by checking all inputs at system edges
- MUST fail fast by raising early rather than propagating bad state
- MUST be explicit and never rely on implicit behavior or defaults
- MUST use `ABC` and `@abstractmethod` to enforce interface contracts

---

## Verify Before Claiming

- NEVER claim code is broken without running it to confirm
- NEVER assume a fix is needed without reproducing the issue
- NEVER guess at behavior without checking the actual implementation
- MUST read existing code before claiming it needs changes
- MUST run tests or the application before stating something fails

---

## Code Longevity

- MUST use stable, documented APIs
- MUST avoid implementation-specific tricks
- MUST document reasoning, not mechanics

---

## Commenting

### Core Principle

Comments exist for future readers with no conversation context. The codebase stands alone.

### Prohibited Comments

- NEVER use `# TODO:` comments
- NEVER use `# HACK:` comments
- NEVER use `# FIXME:` comments
- NEVER use `# type: ignore` without justification comment
- NEVER use `# noqa` without justification comment
- NEVER write comments referencing prompts: "As requested", "Per instruction", "Added because asked"
- NEVER write comments referencing conversation: "As discussed", "Per our conversation", "Following the plan"
- NEVER write meta-comments: "NEW:", "CHANGED:", "This is the fix"
- NEVER write obvious descriptions: "Loop through list", "Return result", "Check if None"
- NEVER write comments that would not make sense 2 years later without conversation context

### Required Comments

- MUST document non-obvious behavior: security implications, order dependencies
- MUST document external requirements: RFC references, spec compliance
- MUST document edge cases: intentional empty returns, fail-safe defaults
- MUST document reasoning: why this approach, not what it does

### Comment Decision

1. Is this obvious from the code? -> No comment
2. Does this reference conversation? -> No comment
3. Would future reader need this context? -> Comment
4. Is there non-obvious reasoning? -> Comment

---

## Docstrings

- MUST use docstrings on public ABC interfaces and their methods
- MUST use docstrings on `ApplicationBuilder` public methods
- MUST NOT add docstrings to trivial internal helper methods where the name is self-explanatory
- USE triple double quotes `"""` for all docstrings
- USE imperative mood: "Get the service" not "Gets the service" or "Returns the service"

---

## Exception Handling

- MUST use specific exception types: `ValueError`, `KeyError`, `TypeError`, `RuntimeError`
- MUST NOT use bare `except:` or `except Exception:` without re-raising or logging
- MUST NOT silently catch and ignore exceptions
- MUST use `raise ... from e` for exception chaining when wrapping exceptions
- Workers MUST catch exceptions inside the work loop to prevent thread death

---

## Thread Safety

- MUST use `threading.Lock` or `threading.RLock` for shared mutable state
- MUST use `threading.Event` for signaling between threads
- MUST NOT share mutable state between threads without synchronization
- MUST use `daemon=True` for background threads that should not prevent shutdown
