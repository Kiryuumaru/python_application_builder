---
applyTo: '**'
---
# Code Quality Rules

## Fix Hygiene

### Fix Procedure

1. **Find and analyze the problem** — MUST understand what is actually broken before touching code
2. **Use proper diagnostic tools** — MUST use available tools (tracing, structured logs, debuggers, profilers, network tools, build diagnostics) to produce evidence of the root cause. MUST NOT guess based on reading code alone
3. **Fix one thing at a time** — MUST make ONE focused change per attempt
4. **Verify** — MUST confirm the fix works (build, test, or runtime check). If confirmed working, proceed to next issue
5. **Undo on failure** — If the fix does NOT work, MUST undo the specific code changes from that attempt. MUST NOT use blanket `git revert` or `git checkout` that could destroy ongoing uncommitted work. MUST restore files to their pre-attempt state before trying a different approach

### Prohibited

- NEVER leave non-working code from failed fix attempts
- NEVER apply fix B on top of failed fix A
- NEVER comment out failed code instead of removing
- NEVER guess at root cause without diagnostic evidence
- NEVER make multiple unrelated changes in one fix attempt

---

## Zero Warnings

Build MUST complete with 0 warnings and 0 errors. Every warning is a potential bug.

---

## None Handling

Use `Optional[T]` (or `T | None`) to mark values that may be `None`. Type-checker escape hatches (`# type: ignore`, `cast(T, value)`, `assert value is not None`) are prohibited except when ALL conditions are met:
1. Value is proven non-`None` at that point
2. Type checker cannot infer this due to analysis limitations
3. Comment explains why it is safe
4. No reasonable restructuring alternative exists

None handling patterns:
- For required value that may be `None`, USE `if value is None: raise ValueError(...)`
- For `None` check, USE `if value is None: return`
- For optional value, DECLARE as `Optional[T]` and provide `None` default explicitly
- For constructor parameter, USE `if param is None: raise ValueError("param required")` at the top of `__init__`
- For default mutable arguments, USE `param: Optional[List[X]] = None` then `param = param if param is not None else []`
- MUST NEVER use mutable default arguments (`def f(x=[])`)
- MUST NEVER use bare `except:` — always specify the exception type
- MUST USE `is` / `is not` for `None` comparisons, never `==` / `!=`

---

## Reliability Principles

- MUST make illegal states unrepresentable using the type system
- MUST validate at boundaries by checking all inputs at system edges
- MUST fail fast by throwing early rather than propagating bad state
- MUST be explicit and never rely on implicit behavior or defaults

---

## Verify Before Claiming

- NEVER claim code is broken without running it to confirm
- NEVER assume a fix is needed without reproducing the issue
- NEVER guess at behavior without checking the actual implementation
- MUST read existing code before claiming it needs changes
- MUST run tests or build before stating something fails

---

## Code Longevity

- MUST use stable, documented APIs
- MUST avoid implementation-specific tricks
- MUST document reasoning, not mechanics

---

## Constructor Simplicity

When a constructor only stores parameters, keep it minimal — assign parameters directly to instance attributes without additional logic.

Simple constructor (preferred):
```python
class DomainEventDispatcher:
    def __init__(self, handlers: List[IDomainEventHandler]):
        self.handlers = handlers

    def do_work(self):
        for handler in self.handlers:
            handler.handle()
```

Constructor simplicity rules:
- USE simple assignment when constructor only stores parameters
- KEEP validation or transformation logic in the constructor when needed
- MUST accept dependencies via `__init__` parameters with type hints

---

## Commenting

### Core Principle

Comments exist for future readers with no conversation context. The codebase stands alone.

### Prohibited Comments

- NEVER use `# TODO:` comments
- NEVER use `# HACK:` comments
- NEVER use `# FIXME:` comments
- NEVER use `# type: ignore` without a justification comment on the same line
- NEVER use `# noqa` without a specific rule code AND justification (e.g. `# noqa: E501 - URL must stay on one line`)
- NEVER use `# pylint: disable=...` without justification
- NEVER use `cast(T, value)` without a justification comment
- NEVER use `assert value is not None` to silence a type checker without justification
- NEVER write comments referencing prompts: "As requested", "Per instruction", "Added because asked"
- NEVER write comments referencing conversation: "As discussed", "Per our conversation", "Following the plan"
- NEVER write meta-comments: "NEW:", "CHANGED:", "This is the fix"
- NEVER write obvious descriptions: "Loop through list", "Return result", "Check if None"
- NEVER write comments that would not make sense 2 years later without conversation context
- NEVER write docstrings on internal implementation classes (prefix `_` or not in `__all__`)
- NEVER describe what code does when code is self-explanatory
- NEVER ignore linter or type-checker warnings
- NEVER suppress warnings instead of fixing root cause

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

Docstrings (PEP 257 triple-quoted strings) are required for public types and functions shared across layers. Internal implementations do not need docstrings.

Public cross-layer items MUST have docstrings:
- ABC interfaces (services, providers, repositories, internal abstractions)
- Abstract methods on those interfaces
- Domain entities, value objects, events, enums
- Application DTOs / models used by other layers
- Module-level public APIs exported via `__all__`

Internal items MUST NOT have docstrings:
- Interface implementations resolved via DI (prefix `_` or not in `__all__`)
- Infrastructure adapters / repositories
- Layer-internal helper classes and module-private functions

Docstring style:
- USE triple double-quotes `"""..."""`
- USE one-line summary for single-purpose items
- USE multi-line with `Args:` / `Returns:` / `Raises:` sections for non-trivial signatures
- MUST NOT restate the function name or repeat type hints
- MUST document `Raises:` for any exception that is part of the contract

Quick rule:
- public ABC / cross-layer type = docstring required
- module-private (`_name`) or internal implementation = no docstring
