# Initialize Python Application Builder Project

You are initializing a Python project that uses the **Python Application Builder** framework — a lightweight dependency injection framework built on clean architecture principles. It ships as a single module (`application_builder.py`) providing an IoC container with automatic constructor injection, multi-source configuration, background workers, structured logging, cooperative cancellation, job management, and a middleware pipeline.

Refer to the [python_application_builder](https://github.com/Kiryuumaru/python_application_builder) repository for more detailed information and guidelines.

---

## Step 1: Install the Rules and Documentation First

Before creating any application file or running any command, install the project's rule set and framework documentation. These rules dictate folder structure, naming, layering, lifetimes, and prohibited patterns. Every step that follows depends on them.

Clone the framework repository into a temporary directory:

```bash
python -c "import subprocess, tempfile, os; subprocess.run(['git', 'clone', 'https://github.com/Kiryuumaru/python_application_builder.git', os.path.join(tempfile.gettempdir(), 'pab_source')])"
```

Copy the agent instruction files into the project:

```bash
python -c "import shutil, tempfile, os; src=os.path.join(tempfile.gettempdir(), 'pab_source', '.github', 'instructions'); dst=os.path.join('.github', 'instructions'); os.makedirs(dst, exist_ok=True); [shutil.copy2(os.path.join(src, f), os.path.join(dst, f)) for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]"
```

Copy the framework documentation:

```bash
python -c "import shutil, tempfile, os; src=os.path.join(tempfile.gettempdir(), 'pab_source', 'docs'); dst='docs/application_builder'; os.makedirs(dst, exist_ok=True); [shutil.copy2(os.path.join(src, f), os.path.join(dst, f)) for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]"
```

---

## Step 2: Read All Rules End-to-End

Read every file in `.github/instructions/` before writing any code. Treat them as MUST/NEVER constraints, not suggestions:

- `project-context.instructions.md` — terminology, project status, breaking-change policy
- `rule-style.instructions.md` — how rules are written
- `architecture.instructions.md` — layering, folder structure, DI lifetimes, workers, ports/adapters, naming, prohibited patterns
- `code-quality.instructions.md` — None handling, docstrings, comments, constructor simplicity
- `documentation.instructions.md` — when to update docs
- `workflow.instructions.md` — build/test commands, pre-commit checks
- `agent-terminal.instructions.md` — terminal usage rules (no `&&`, `|`, `;`, redirections)
- `ui-test-practices.instructions.md` — pytest conventions, no `time.sleep`, DI test doubles

Also skim `docs/application_builder/` for the public API surface.

---

## Step 3: Plan the Structure From the Rules

Using the rules from Step 2, design the project layout BEFORE writing files. The architecture rules require a layered structure for application code:

```
<project_root>/
├── .github/
│   └── instructions/                  <- already copied in Step 1
├── docs/
│   └── application_builder/           <- already copied in Step 1
├── src/
│   ├── application_builder.py         <- framework module (copied in Step 4)
│   ├── domain/                        <- pure business logic, no I/O
│   │   ├── __init__.py
│   │   └── shared/
│   ├── application/                   <- application interfaces, services, workers
│   │   ├── __init__.py
│   │   └── shared/
│   ├── infrastructure/                <- adapters, repositories
│   │   └── __init__.py
│   ├── presentation/                  <- composition root, CLI, controllers
│   │   ├── __init__.py
│   │   └── main.py                    <- ONLY module that imports from infrastructure
│   └── requirements.txt
└── tests/
    ├── conftest.py
    └── (test files mirroring src/)
```

For a minimal starter, only `presentation/main.py` and `src/application_builder.py` are required. Domain / Application / Infrastructure modules are added as features arrive, following the "Adding a New Feature (Layered)" workflow in `architecture.instructions.md`.

---

## Step 4: Install the Framework Module

```bash
python -c "import shutil, tempfile, os; shutil.copy2(os.path.join(tempfile.gettempdir(), 'pab_source', 'src', 'application_builder.py'), os.path.join('src', 'application_builder.py'))"
```

---

## Step 5: Create requirements.txt

Create `src/requirements.txt` with pinned dependencies:

```
loguru==0.7.3
pyyaml==6.0.3
```

---

## Step 6: Install Dependencies

```bash
cd src
pip install -r requirements.txt
```

---

## Step 7: Create the Composition Root

Create `src/presentation/__init__.py` (empty) and `src/presentation/main.py`:

```python
from application_builder import ApplicationBuilder, IConfiguration, ILogger, Worker


class MainWorker(Worker):
    def __init__(self, config: IConfiguration, logger: ILogger):
        super().__init__()
        self._config = config
        self._logger = logger

    def execute(self) -> None:
        self._logger.info("Application started")
        while not self.is_stopping():
            self.wait_for_stop(1.0)


def main() -> None:
    app = ApplicationBuilder()
    app.add_configuration_dictionary({
        "App": {
            "Name": "MyApplication",
        },
    })
    app.add_worker(MainWorker)
    app.run()


if __name__ == "__main__":
    main()
```

`presentation/main.py` is the only module permitted to import from `infrastructure.*`. As features are added, register each layer here in dependency order: domain → application → infrastructure → presentation.

---

## Step 8: Verify the Setup

The project structure MUST look like this:

```
<project_root>/
├── .github/
│   └── instructions/
│       ├── agent-terminal.instructions.md
│       ├── architecture.instructions.md
│       ├── code-quality.instructions.md
│       ├── documentation.instructions.md
│       ├── project-context.instructions.md
│       ├── rule-style.instructions.md
│       ├── ui-test-practices.instructions.md
│       └── workflow.instructions.md
├── docs/
│   └── application_builder/
│       ├── advanced-features.md
│       ├── api-reference.md
│       ├── configuration.md
│       ├── dependency-injection.md
│       ├── features.md
│       ├── getting-started.md
│       ├── logging.md
│       ├── samples.md
│       └── workers.md
├── src/
│   ├── application_builder.py
│   ├── presentation/
│   │   ├── __init__.py
│   │   └── main.py
│   └── requirements.txt
└── tests/
```

Verify by running:

```bash
cd src
python -m presentation.main
```

Expected: the application starts, prints a structured log message via loguru, and blocks until Ctrl+C is pressed.

---

## Step 9: Pre-Commit Verification

Per `workflow.instructions.md`:

```bash
cd src
python -m py_compile application_builder.py
```

```bash
cd src
python -m py_compile presentation/main.py
```

Both MUST exit with code 0.

---

## Reminder: Rules Take Precedence

When adding any feature beyond this scaffold, MUST consult the rules in `.github/instructions/` first. The architecture rules govern:

- Where each type of file lives (entities, value objects, services, adapters, etc.)
- Which layer may import from which other layer
- Which lifetime each service uses
- Where each ABC interface lives (a single `interfaces/` package per feature)
- Naming conventions for modules, classes, and ABCs
- Prohibited patterns that MUST NEVER be introduced

Do not improvise structure. The rules are the contract.
