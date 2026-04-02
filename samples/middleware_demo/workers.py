from application_builder import Worker, ILogger, MiddlewarePipeline
from middleware import TimingMiddleware, AuthMiddleware, LoggingMiddleware


class PipelineWorker(Worker):
    def __init__(self, logger: ILogger) -> None:
        super().__init__()
        self._logger = logger

    def execute(self) -> None:
        self._logger.info("=== Middleware Pipeline Demo ===")

        pipeline = MiddlewarePipeline()
        pipeline.use(TimingMiddleware(self._logger))
        pipeline.use(AuthMiddleware(self._logger))
        pipeline.use(LoggingMiddleware(self._logger))

        # Add a terminal handler via use_func
        pipeline.use_func(lambda ctx, _next: self._logger.info(
            f"[Handler] Processing request for '{ctx.get('path', '/')}' by '{ctx.get('user', 'anon')}'"
        ))

        # Request 1: authenticated user
        self._logger.info("--- Request 1: Authenticated ---")
        pipeline.execute({"user": "alice", "path": "/api/data"})

        # Request 2: no user — AuthMiddleware short-circuits
        self._logger.info("--- Request 2: Unauthenticated ---")
        ctx2: dict = {"path": "/admin/settings"}
        pipeline.execute(ctx2)
        if ctx2.get("blocked"):
            self._logger.info("Request was blocked by auth middleware")

        self._logger.info("Demo complete.")
