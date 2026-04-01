import time
from typing import List
from application_builder import Worker, ILogger, IConfiguration, CancellationTokenSource
from interfaces import IBuildStep


class BuildPipelineWorker(Worker):
    """Runs build steps sequentially. Stops the pipeline on first failure."""

    def __init__(self, steps: List[IBuildStep],
                 config: IConfiguration,
                 logger: ILogger):
        super().__init__()
        self._steps = steps
        self._project_dir = config.get("Build:ProjectDir", ".")
        self._timeout = config.get_float("Build:TimeoutSeconds", 30.0)
        self._logger = logger

    def execute(self) -> None:
        self._logger.info(f"=== Build Pipeline ({len(self._steps)} steps) ===")
        self._logger.info(f"Project: {self._project_dir}")

        overall_start = time.time()
        passed = 0
        failed = 0

        for i, step in enumerate(self._steps, 1):
            if self.is_stopping():
                self._logger.warning("Build cancelled")
                break

            self._logger.info(f"\n--- Step {i}/{len(self._steps)}: {step.name()} ---")
            step_start = time.time()

            try:
                success = step.run(self._project_dir)
                elapsed = time.time() - step_start

                if success:
                    passed += 1
                    self._logger.success(f"Step '{step.name()}' passed ({elapsed:.1f}s)")
                else:
                    failed += 1
                    self._logger.error(f"Step '{step.name()}' failed ({elapsed:.1f}s)")
                    self._logger.error("Pipeline aborted — fix errors and retry")
                    break
            except Exception as e:
                failed += 1
                self._logger.error(f"Step '{step.name()}' crashed: {e}")
                break

        total_time = time.time() - overall_start
        self._logger.info(f"\n=== Build {'SUCCEEDED' if failed == 0 else 'FAILED'} ===")
        self._logger.info(f"Steps: {passed} passed, {failed} failed, {total_time:.1f}s total")
