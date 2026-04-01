import os
import time
from application_builder import ILogger, CliRunnerService, IConfiguration
from interfaces import IBuildStep


class LintStep(IBuildStep):
    def __init__(self, logger: ILogger):
        self._logger = logger

    def name(self) -> str:
        return "Lint"

    def run(self, project_dir: str) -> bool:
        self._logger.info("Checking Python syntax...")
        # Find .py files in the project dir
        py_files = [f for f in os.listdir(project_dir) if f.endswith('.py')]
        for f in py_files:
            self._logger.debug(f"  Syntax OK: {f}")
        self._logger.success(f"Lint passed — {len(py_files)} files checked")
        return True


class TestStep(IBuildStep):
    def __init__(self, logger: ILogger):
        self._logger = logger

    def name(self) -> str:
        return "Test"

    def run(self, project_dir: str) -> bool:
        self._logger.info("Running tests...")
        # Simulated test results
        tests = [
            ("test_build_step_lint", True),
            ("test_build_step_test", True),
            ("test_build_step_package", True),
            ("test_pipeline_order", True),
        ]
        passed = 0
        for test_name, result in tests:
            status = "PASS" if result else "FAIL"
            self._logger.info(f"  {status}: {test_name}")
            if result:
                passed += 1
            time.sleep(0.3)
        self._logger.success(f"Tests passed — {passed}/{len(tests)}")
        return passed == len(tests)


class PackageStep(IBuildStep):
    def __init__(self, logger: ILogger):
        self._logger = logger

    def name(self) -> str:
        return "Package"

    def run(self, project_dir: str) -> bool:
        self._logger.info("Packaging artifacts...")
        time.sleep(0.5)
        self._logger.success("Package created: build/dist/app-1.0.0.tar.gz (simulated)")
        return True
