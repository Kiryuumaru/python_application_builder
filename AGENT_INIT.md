# Initialize Python Application Builder Project

You are initializing a Python project that uses the **Python Application Builder** framework — a lightweight dependency injection framework built on clean architecture principles. It ships as a single module (`application_builder.py`) providing an IoC container with automatic constructor injection, multi-source configuration, background workers, structured logging, cooperative cancellation, job management, and a middleware pipeline.

Follow these steps to set up the project in the current directory.

---

## Step 1: Create Project Structure

Create the following directory layout:

```
<project_root>/
├── docs/
│   └── application_builder/    <- Framework documentation (copied from the repository)
├── src/
│   ├── application_builder.py   <- Framework module (copied from the repository)
│   ├── main.py                  <- Application entry point
│   └── requirements.txt         <- Dependencies
└── tests/
    └── (test files)
```

---

## Step 2: Install the Framework

Clone the framework repository into a temporary directory and copy the core module into your project. The following commands use Python for cross-platform compatibility:

```bash
python -c "import subprocess, tempfile, os; subprocess.run(['git', 'clone', 'https://github.com/Kiryuumaru/python_application_builder.git', os.path.join(tempfile.gettempdir(), 'pab_source')])"
```

Then copy the framework module and documentation into your project:

```bash
python -c "import shutil, tempfile, os; shutil.copy2(os.path.join(tempfile.gettempdir(), 'pab_source', 'src', 'application_builder.py'), os.path.join('src', 'application_builder.py'))"
```

```bash
python -c "import shutil, tempfile, os; src=os.path.join(tempfile.gettempdir(), 'pab_source', 'docs'); dst='docs/application_builder'; os.makedirs(dst, exist_ok=True); [shutil.copy2(os.path.join(src, f), os.path.join(dst, f)) for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]"
```

---

## Step 3: Create requirements.txt

Create `src/requirements.txt` with the following pinned dependencies:

```
loguru==0.7.3
pyyaml==6.0.3
```

---

## Step 4: Install Dependencies

```bash
cd src
pip install -r requirements.txt
```

---

## Step 5: Create the Entry Point

Create `src/main.py` with the following starter template:

```python
from application_builder import ApplicationBuilder, IConfiguration, ILogger, Worker


class MainWorker(Worker):
    def __init__(self, config: IConfiguration, logger: ILogger):
        super().__init__()
        self.config = config
        self.logger = logger

    def execute(self):
        self.logger.info("Application started")
        while not self.is_stopping():
            self.wait_for_stop(1.0)


app = ApplicationBuilder()
app.add_configuration_dictionary({
    "App": {
        "Name": "MyApplication"
    }
})
app.add_worker(MainWorker)
app.run()
```

---

## Step 6: Verify the Setup

After completing all steps, the project structure MUST look like this:

```
<project_root>/
├── docs/
│   └── application_builder/
│       ├── advanced-features.md
│       ├── api-reference.md
│       ├── configuration.md
│       ├── dependency-injection.md
│       ├── getting-started.md
│       ├── logging.md
│       ├── samples.md
│       └── workers.md
├── src/
│   ├── application_builder.py   <- Framework module
│   ├── main.py                  <- Entry point with MainWorker
│   └── requirements.txt         <- loguru + pyyaml pinned
└── tests/
    └── (empty, ready for test files)
```

Verify by running:

```bash
cd src
python main.py
```

Expected: the application starts, prints a structured log message via loguru, and blocks until Ctrl+C is pressed.

---

## Architecture Rules

- Define ABC interfaces for service contracts before implementations
- Accept dependencies via `__init__` parameters with type hints
- Never use global state or module-level singletons for service resolution
- Singleton services must be thread-safe and must not hold references to scoped or transient services
- Workers must check `self.is_stopping()` in loops and use `self.wait_for_stop(timeout)` instead of `time.sleep()`
- Workers must handle exceptions within the execute loop
- Refer to `docs/application_builder/` for detailed framework documentation
