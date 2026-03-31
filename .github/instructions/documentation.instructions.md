---
applyTo: '**'
---
# Documentation Rules

## Check Documentation First

Before implementing, modifying, or asking questions, check the relevant documentation.

| Task | Check |
|------|-------|
| New feature | `README.md` for similar patterns |
| Modify framework | `README.md` API Reference section |
| Write tests | `.github/instructions/code-quality.instructions.md` |
| Architecture question | `.github/instructions/architecture.instructions.md` |

---

## Documentation Locations

| Content | Location |
|---------|----------|
| API reference | `README.md` |
| Architecture patterns | `.github/instructions/architecture.instructions.md` |
| Code conventions | `.github/instructions/code-quality.instructions.md` |
| Build commands | `.github/instructions/workflow.instructions.md` |

---

## Required Updates

### API Changes

| Change | Update |
|--------|--------|
| New public method on ApplicationBuilder | Update API Reference in README.md |
| New service lifetime or registration pattern | Update Core Concepts in README.md |
| New configuration provider | Update Configuration System in README.md |
| New worker base class | Update Worker System in README.md |
| Changed method signature | Update all code examples |

### Test Changes

| Change | Update |
|--------|--------|
| New test patterns | Update Testing section in README.md |
| New test utilities | Document in README.md |

### Architecture Changes

| Change | Update |
|--------|--------|
| New subsystem | Update architecture.instructions.md |
| New ABC interface | Update architecture.instructions.md |
| Changed dependency flow | Update architecture.instructions.md diagrams |

---

## Pre-Commit Documentation Check

- Does this change any public API? -> Update README.md
- Does this add a new subsystem? -> Update architecture.instructions.md
- Does this change configuration behavior? -> Update README.md Configuration section
- Does this change worker lifecycle? -> Update README.md Worker section
