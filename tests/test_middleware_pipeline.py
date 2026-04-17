from typing import Any, Callable, Dict, List

import pytest

from application_builder import IMiddleware, MiddlewarePipeline


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

class LoggingMiddleware(IMiddleware):
    """Records its own label into context["log"] and calls next."""

    def __init__(self, label: str):
        self._label = label

    def invoke(self, context: Dict[str, Any], next_middleware: Callable[[Dict[str, Any]], None]) -> None:
        context.setdefault("log", []).append(self._label)
        next_middleware(context)


class ModifyMiddleware(IMiddleware):
    """Sets a key on the context and calls next."""

    def __init__(self, key: str, value: Any):
        self._key = key
        self._value = value

    def invoke(self, context: Dict[str, Any], next_middleware: Callable[[Dict[str, Any]], None]) -> None:
        context[self._key] = self._value
        next_middleware(context)


class ShortCircuitMiddleware(IMiddleware):
    """Records itself but never calls next, stopping the chain."""

    def __init__(self, label: str = "short"):
        self._label = label

    def invoke(self, context: Dict[str, Any], next_middleware: Callable[[Dict[str, Any]], None]) -> None:
        context.setdefault("log", []).append(self._label)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEmptyPipeline:

    def test_execute_with_no_middlewares_does_nothing(self):
        pipeline = MiddlewarePipeline()
        context: Dict[str, Any] = {}
        pipeline.execute(context)
        assert context == {}

    def test_execute_empty_pipeline_does_not_raise(self):
        pipeline = MiddlewarePipeline()
        pipeline.execute({"key": "value"})


class TestSingleClassMiddleware:

    def test_single_middleware_is_invoked(self):
        pipeline = MiddlewarePipeline()
        pipeline.use(LoggingMiddleware("A"))
        context: Dict[str, Any] = {}
        pipeline.execute(context)
        assert context["log"] == ["A"]

    def test_single_middleware_can_modify_context(self):
        pipeline = MiddlewarePipeline()
        pipeline.use(ModifyMiddleware("added", 42))
        context: Dict[str, Any] = {"original": True}
        pipeline.execute(context)
        assert context["added"] == 42
        assert context["original"] is True


class TestSingleFunctionMiddleware:

    def test_use_func_is_invoked(self):
        def mw(ctx: Dict[str, Any], nxt: Callable) -> None:
            ctx.setdefault("log", []).append("func")
            nxt(ctx)

        pipeline = MiddlewarePipeline()
        pipeline.use_func(mw)
        context: Dict[str, Any] = {}
        pipeline.execute(context)
        assert context["log"] == ["func"]

    def test_use_func_can_modify_context(self):
        def mw(ctx: Dict[str, Any], nxt: Callable) -> None:
            ctx["touched"] = True
            nxt(ctx)

        pipeline = MiddlewarePipeline()
        pipeline.use_func(mw)
        context: Dict[str, Any] = {}
        pipeline.execute(context)
        assert context["touched"] is True


class TestChaining:

    def test_use_returns_pipeline(self):
        pipeline = MiddlewarePipeline()
        result = pipeline.use(LoggingMiddleware("A"))
        assert result is pipeline

    def test_use_func_returns_pipeline(self):
        pipeline = MiddlewarePipeline()
        result = pipeline.use_func(lambda ctx, nxt: nxt(ctx))
        assert result is pipeline

    def test_fluent_chaining(self):
        pipeline = (
            MiddlewarePipeline()
            .use(LoggingMiddleware("A"))
            .use(LoggingMiddleware("B"))
            .use_func(lambda ctx, nxt: (ctx.setdefault("log", []).append("C"), nxt(ctx)))
        )
        context: Dict[str, Any] = {}
        pipeline.execute(context)
        assert context["log"] == ["A", "B", "C"]


class TestOrdering:

    def test_middlewares_execute_in_order_added(self):
        pipeline = MiddlewarePipeline()
        pipeline.use(LoggingMiddleware("first"))
        pipeline.use(LoggingMiddleware("second"))
        pipeline.use(LoggingMiddleware("third"))
        context: Dict[str, Any] = {}
        pipeline.execute(context)
        assert context["log"] == ["first", "second", "third"]

    def test_each_middleware_calls_next(self):
        call_order: List[int] = []

        class OrderMiddleware(IMiddleware):
            def __init__(self, index: int):
                self._index = index

            def invoke(self, context, next_middleware):
                call_order.append(self._index)
                next_middleware(context)

        pipeline = MiddlewarePipeline()
        for i in range(5):
            pipeline.use(OrderMiddleware(i))
        pipeline.execute({})
        assert call_order == [0, 1, 2, 3, 4]


class TestContextModification:

    def test_subsequent_middleware_sees_changes(self):
        pipeline = MiddlewarePipeline()
        pipeline.use(ModifyMiddleware("step", 1))

        class AssertMiddleware(IMiddleware):
            def __init__(self):
                self.saw_value = None

            def invoke(self, context, next_middleware):
                self.saw_value = context.get("step")
                next_middleware(context)

        checker = AssertMiddleware()
        pipeline.use(checker)
        pipeline.execute({})
        assert checker.saw_value == 1

    def test_context_accumulates_changes(self):
        pipeline = MiddlewarePipeline()
        pipeline.use(ModifyMiddleware("a", 1))
        pipeline.use(ModifyMiddleware("b", 2))
        pipeline.use(ModifyMiddleware("c", 3))
        context: Dict[str, Any] = {}
        pipeline.execute(context)
        assert context == {"a": 1, "b": 2, "c": 3}


class TestShortCircuit:

    def test_middleware_not_calling_next_stops_chain(self):
        pipeline = MiddlewarePipeline()
        pipeline.use(LoggingMiddleware("before"))
        pipeline.use(ShortCircuitMiddleware("stop"))
        pipeline.use(LoggingMiddleware("after"))
        context: Dict[str, Any] = {}
        pipeline.execute(context)
        assert context["log"] == ["before", "stop"]

    def test_short_circuit_at_first_position(self):
        pipeline = MiddlewarePipeline()
        pipeline.use(ShortCircuitMiddleware("only"))
        pipeline.use(LoggingMiddleware("never"))
        context: Dict[str, Any] = {}
        pipeline.execute(context)
        assert context["log"] == ["only"]


class TestMultipleMiddlewares:

    def test_three_middlewares_in_order(self):
        pipeline = MiddlewarePipeline()
        pipeline.use(LoggingMiddleware("1"))
        pipeline.use(LoggingMiddleware("2"))
        pipeline.use(LoggingMiddleware("3"))
        context: Dict[str, Any] = {}
        pipeline.execute(context)
        assert context["log"] == ["1", "2", "3"]

    def test_five_middlewares_in_order(self):
        pipeline = MiddlewarePipeline()
        for i in range(1, 6):
            pipeline.use(LoggingMiddleware(str(i)))
        context: Dict[str, Any] = {}
        pipeline.execute(context)
        assert context["log"] == ["1", "2", "3", "4", "5"]


class TestMixedClassAndFunction:

    def test_class_then_function(self):
        def func_mw(ctx: Dict[str, Any], nxt: Callable) -> None:
            ctx.setdefault("log", []).append("func")
            nxt(ctx)

        pipeline = MiddlewarePipeline()
        pipeline.use(LoggingMiddleware("class"))
        pipeline.use_func(func_mw)
        context: Dict[str, Any] = {}
        pipeline.execute(context)
        assert context["log"] == ["class", "func"]

    def test_function_then_class(self):
        def func_mw(ctx: Dict[str, Any], nxt: Callable) -> None:
            ctx.setdefault("log", []).append("func")
            nxt(ctx)

        pipeline = MiddlewarePipeline()
        pipeline.use_func(func_mw)
        pipeline.use(LoggingMiddleware("class"))
        context: Dict[str, Any] = {}
        pipeline.execute(context)
        assert context["log"] == ["func", "class"]

    def test_interleaved_class_and_function(self):
        def make_func(label: str) -> Callable:
            def mw(ctx: Dict[str, Any], nxt: Callable) -> None:
                ctx.setdefault("log", []).append(label)
                nxt(ctx)
            return mw

        pipeline = MiddlewarePipeline()
        pipeline.use(LoggingMiddleware("C1"))
        pipeline.use_func(make_func("F1"))
        pipeline.use(LoggingMiddleware("C2"))
        pipeline.use_func(make_func("F2"))
        context: Dict[str, Any] = {}
        pipeline.execute(context)
        assert context["log"] == ["C1", "F1", "C2", "F2"]

    def test_mixed_with_short_circuit(self):
        def func_mw(ctx: Dict[str, Any], nxt: Callable) -> None:
            ctx.setdefault("log", []).append("func")
            nxt(ctx)

        pipeline = MiddlewarePipeline()
        pipeline.use_func(func_mw)
        pipeline.use(ShortCircuitMiddleware("stop"))
        pipeline.use(LoggingMiddleware("skipped"))
        context: Dict[str, Any] = {}
        pipeline.execute(context)
        assert context["log"] == ["func", "stop"]
