import io
import loguru
import pytest

from application_builder import (
    ApplicationBuilder,
    LoguruLogger,
    create_loguru_logger,
    validate_log_level,
    reset_logger_state,
)


def _reset_loguru_state():
    """Reset loguru global state so each test starts clean."""
    reset_logger_state()


def _capture_logger_output(log_context: str, log_level: str) -> tuple:
    """Create a logger writing to a StringIO and return (logger, buffer)."""
    _reset_loguru_state()

    # create_loguru_logger triggers initialization internally
    create_loguru_logger(log_context, log_level, None)

    buf = io.StringIO()
    resolved = validate_log_level(log_level)
    level_no = loguru.logger.level(resolved).no

    def context_filter(record):
        return (
            record["extra"].get("context") == log_context
            and record["level"].no >= level_no
        )

    log_format = "{level} - [{extra[context]}] {message}"
    loguru.logger.add(buf, colorize=False, format=log_format, level="TRACE", filter=context_filter)
    bound = loguru.logger.bind(context=log_context)
    return bound, buf


# ─── validate_log_level ──────────────────────────────────────────────


class TestValidateLogLevel:
    """Tests for the validate_log_level function."""

    @pytest.mark.parametrize("input_level,expected", [
        ("TRACE", "TRACE"),
        ("DEBUG", "DEBUG"),
        ("INFO", "INFO"),
        ("SUCCESS", "SUCCESS"),
        ("WARNING", "WARNING"),
        ("ERROR", "ERROR"),
        ("CRITICAL", "CRITICAL"),
    ])
    def test_valid_levels_pass_through(self, input_level: str, expected: str):
        assert validate_log_level(input_level) == expected

    @pytest.mark.parametrize("input_level,expected", [
        ("trace", "TRACE"),
        ("debug", "DEBUG"),
        ("info", "INFO"),
        ("Info", "INFO"),
        ("Warning", "WARNING"),
        ("error", "ERROR"),
        ("Error", "ERROR"),
        ("critical", "CRITICAL"),
    ])
    def test_case_insensitive(self, input_level: str, expected: str):
        assert validate_log_level(input_level) == expected

    @pytest.mark.parametrize("invalid", ["", "BANANA", "NONE", "OFF", "ALL", "123", "WARN", "FATAL", "VERBOSE", "INFORMATION"])
    def test_invalid_levels_raise(self, invalid: str):
        with pytest.raises(ValueError, match="Invalid log level"):
            validate_log_level(invalid)


# ─── Level filtering ─────────────────────────────────────────────────


class TestLoggerLevelFiltering:
    """Tests that each log level properly filters lower-severity messages."""

    ALL_LEVELS = ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]
    LEVEL_ORDER = {name: idx for idx, name in enumerate(ALL_LEVELS)}

    def _log_all_levels(self, bound_logger):
        bound_logger.trace("msg_TRACE")
        bound_logger.debug("msg_DEBUG")
        bound_logger.info("msg_INFO")
        bound_logger.success("msg_SUCCESS")
        bound_logger.warning("msg_WARNING")
        bound_logger.error("msg_ERROR")
        bound_logger.critical("msg_CRITICAL")

    @pytest.mark.parametrize("min_level", ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"])
    def test_level_filtering(self, min_level: str):
        bound, buf = _capture_logger_output("FilterTest", min_level)
        self._log_all_levels(bound)
        output = buf.getvalue()

        min_idx = self.LEVEL_ORDER[min_level]
        for level_name in self.ALL_LEVELS:
            marker = f"msg_{level_name}"
            if self.LEVEL_ORDER[level_name] >= min_idx:
                assert marker in output, f"{level_name} should appear when min_level={min_level}"
            else:
                assert marker not in output, f"{level_name} should NOT appear when min_level={min_level}"


# ─── Per-logger isolation ────────────────────────────────────────────


class TestLoggerLevelIsolation:
    """Tests that each logger respects its own level independently."""

    def test_two_loggers_different_levels(self):
        _reset_loguru_state()

        # Trigger loguru initialization via a throwaway logger
        create_loguru_logger("_init", "TRACE", None)

        buf_a = io.StringIO()
        buf_b = io.StringIO()

        error_no = loguru.logger.level("ERROR").no
        trace_no = loguru.logger.level("TRACE").no

        log_format = "{level} - [{extra[context]}] {message}"

        def filter_a(record):
            return record["extra"].get("context") == "ServiceA" and record["level"].no >= error_no

        def filter_b(record):
            return record["extra"].get("context") == "ServiceB" and record["level"].no >= trace_no

        loguru.logger.add(buf_a, colorize=False, format=log_format, level="TRACE", filter=filter_a)
        loguru.logger.add(buf_b, colorize=False, format=log_format, level="TRACE", filter=filter_b)

        logger_a = loguru.logger.bind(context="ServiceA")
        logger_b = loguru.logger.bind(context="ServiceB")

        logger_a.trace("a_trace")
        logger_a.info("a_info")
        logger_a.error("a_error")
        logger_a.critical("a_critical")

        logger_b.trace("b_trace")
        logger_b.info("b_info")
        logger_b.error("b_error")

        output_a = buf_a.getvalue()
        output_b = buf_b.getvalue()

        assert "a_trace" not in output_a
        assert "a_info" not in output_a
        assert "a_error" in output_a
        assert "a_critical" in output_a

        assert "b_trace" in output_b
        assert "b_info" in output_b
        assert "b_error" in output_b

        assert "b_" not in output_a, "ServiceB messages must not appear in ServiceA output"
        assert "a_" not in output_b, "ServiceA messages must not appear in ServiceB output"

    def test_create_loguru_logger_isolation(self):
        """Verify create_loguru_logger produces isolated loggers."""
        _reset_loguru_state()

        buf = io.StringIO()
        log_format = "{level} - [{extra[context]}] {message}"

        logger_err = create_loguru_logger("SvcError", "ERROR", None)
        logger_trace = create_loguru_logger("SvcTrace", "TRACE", None)

        error_no = loguru.logger.level("ERROR").no
        trace_no = loguru.logger.level("TRACE").no

        def filter_err(record):
            return record["extra"].get("context") == "SvcError" and record["level"].no >= error_no

        def filter_trace(record):
            return record["extra"].get("context") == "SvcTrace" and record["level"].no >= trace_no

        loguru.logger.add(buf, colorize=False, format=log_format, level="TRACE", filter=filter_err)
        loguru.logger.add(buf, colorize=False, format=log_format, level="TRACE", filter=filter_trace)

        logger_err.info("err_info")
        logger_err.error("err_error")
        logger_trace.trace("trace_trace")
        logger_trace.info("trace_info")

        output = buf.getvalue()

        assert "err_info" not in output
        assert "err_error" in output
        assert "trace_trace" in output
        assert "trace_info" in output


# ─── ILogger interface methods ───────────────────────────────────────


class TestILoggerMethods:
    """Tests that LoguruLogger correctly delegates to all level methods."""

    def _make_logger(self, level: str = "TRACE") -> tuple:
        """Create a LoguruLogger with a captured StringIO buffer."""
        _reset_loguru_state()

        config = ApplicationBuilder()
        config.add_configuration_dictionary({"Logging": {"Level": level}})
        configuration = config._configuration_builder.build()

        loguru_logger = LoguruLogger(configuration, "TestCtx")

        buf = io.StringIO()
        log_format = "{level} - [{extra[context]}] {message}"
        resolved = validate_log_level(level)
        level_no = loguru.logger.level(resolved).no

        def filt(record):
            return record["extra"].get("context") == "TestCtx" and record["level"].no >= level_no

        loguru.logger.add(buf, colorize=False, format=log_format, level="TRACE", filter=filt)
        return loguru_logger, buf

    def test_trace_method(self):
        log, buf = self._make_logger("TRACE")
        log.trace("hello trace")
        assert "hello trace" in buf.getvalue()

    def test_debug_method(self):
        log, buf = self._make_logger("DEBUG")
        log.debug("hello debug")
        assert "hello debug" in buf.getvalue()

    def test_info_method(self):
        log, buf = self._make_logger("INFO")
        log.info("hello info")
        assert "hello info" in buf.getvalue()

    def test_success_method(self):
        log, buf = self._make_logger("SUCCESS")
        log.success("hello success")
        assert "hello success" in buf.getvalue()

    def test_warning_method(self):
        log, buf = self._make_logger("WARNING")
        log.warning("hello warning")
        assert "hello warning" in buf.getvalue()

    def test_error_method(self):
        log, buf = self._make_logger("ERROR")
        log.error("hello error")
        assert "hello error" in buf.getvalue()

    def test_critical_method(self):
        log, buf = self._make_logger("CRITICAL")
        log.critical("hello critical")
        assert "hello critical" in buf.getvalue()


# ─── create_loguru_logger validation ─────────────────────────────────


class TestCreateLoguruLoggerValidation:
    """Tests that create_loguru_logger validates log levels."""

    def test_invalid_level_raises(self):
        _reset_loguru_state()
        with pytest.raises(ValueError, match="Invalid log level"):
            create_loguru_logger("BadTest", "BANANA", None)
