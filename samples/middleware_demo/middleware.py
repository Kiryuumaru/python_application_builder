import time
from typing import Callable, Dict, Any
from application_builder import IMiddleware, ILogger


class TimingMiddleware(IMiddleware):
    """Measure how long the pipeline takes."""

    def __init__(self, logger: ILogger) -> None:
        self._logger = logger

    def invoke(self, context: Dict[str, Any], next_middleware: Callable[[Dict[str, Any]], None]) -> None:
        start = time.monotonic()
        context["timing_start"] = start
        self._logger.info("[Timing] Pipeline started")

        next_middleware(context)

        elapsed = (time.monotonic() - start) * 1000
        self._logger.info(f"[Timing] Pipeline completed in {elapsed:.1f}ms")


class AuthMiddleware(IMiddleware):
    """Check if user is authenticated before proceeding."""

    def __init__(self, logger: ILogger) -> None:
        self._logger = logger

    def invoke(self, context: Dict[str, Any], next_middleware: Callable[[Dict[str, Any]], None]) -> None:
        user = context.get("user")
        if not user:
            self._logger.warning("[Auth] No user in context — request blocked")
            context["blocked"] = True
            return  # Short-circuit: do not call next

        self._logger.info(f"[Auth] User '{user}' authenticated")
        next_middleware(context)


class LoggingMiddleware(IMiddleware):
    """Log request entry and exit."""

    def __init__(self, logger: ILogger) -> None:
        self._logger = logger

    def invoke(self, context: Dict[str, Any], next_middleware: Callable[[Dict[str, Any]], None]) -> None:
        path = context.get("path", "/")
        self._logger.info(f"[Log] >> Request to {path}")
        next_middleware(context)
        self._logger.info(f"[Log] << Response for {path}")
