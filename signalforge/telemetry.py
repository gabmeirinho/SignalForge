"""Query metrics and opt-in, content-free OpenTelemetry traces."""

import base64
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import lru_cache
from time import perf_counter
from typing import Any, Iterator

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from signalforge.config import RuntimeConfig


_active: ContextVar["QueryTelemetry | None"] = ContextVar("query_telemetry", default=None)


def current_telemetry() -> "QueryTelemetry | None":
    return _active.get()


@lru_cache(maxsize=4)
def _tracer(endpoint: str, public_key: str, secret_key: str):
    credentials = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    exporter = OTLPSpanExporter(
        endpoint=endpoint,
        headers={
            "Authorization": f"Basic {credentials}",
            "x-langfuse-ingestion-version": "4",
        },
    )
    provider = TracerProvider(resource=Resource.create({"service.name": "signalforge"}))
    provider.add_span_processor(BatchSpanProcessor(exporter))
    return provider.get_tracer("signalforge.query", "1.0")


def _span(config: RuntimeConfig, name: str):
    if not config.tracing_enabled:
        return nullcontext(None)
    if not (
        config.otlp_traces_endpoint and config.langfuse_public_key and config.langfuse_secret_key
    ):
        raise ValueError("Tracing requires an OTLP traces endpoint and Langfuse keys")
    return _tracer(
        config.otlp_traces_endpoint, config.langfuse_public_key, config.langfuse_secret_key
    ).start_as_current_span(name, record_exception=False, set_status_on_exception=False)


@dataclass
class QueryTelemetry:
    config: RuntimeConfig = field(default_factory=RuntimeConfig)
    run_id: int | None = None
    started: float = field(default_factory=perf_counter)
    stage_ms: dict[str, float | None] = field(
        default_factory=lambda: {"planning": None, "retrieval": None, "answer": None}
    )
    attempts: dict[str, int] = field(default_factory=lambda: {"planning": 0, "answer": 0})
    reported_attempts: dict[str, int] = field(default_factory=lambda: {"planning": 0, "answer": 0})
    usage: dict[str, dict[str, Any]] = field(default_factory=dict)
    planner_provider: str = "local"
    answer_provider: str = "local"
    planner_model: str | None = None
    answer_model: str | None = None
    planner_prompt_version: str | None = None
    answer_prompt_version: str | None = None
    fallback_count: int = 0
    total_ms: float | None = None
    error_category: str | None = None
    status: str = "running"

    @contextmanager
    def activate(self) -> Iterator[None]:
        token = _active.set(self)
        try:
            with _span(self.config, "signalforge.query") as span:
                if span is not None and self.run_id is not None:
                    span.set_attribute("signalforge.query_run.id", self.run_id)
                try:
                    yield
                finally:
                    if span is not None:
                        self._span_attributes(span)
        finally:
            _active.reset(token)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = perf_counter()
        with _span(self.config, f"signalforge.{name}") as span:
            try:
                yield
            finally:
                duration = round((perf_counter() - started) * 1000, 3)
                self.stage_ms[name] = duration
                if span is not None:
                    span.set_attribute("signalforge.stage", name)
                    span.set_attribute("signalforge.latency_ms", duration)

    def record_attempt(self, stage: str) -> None:
        self.attempts[stage] += 1

    def record_usage(self, stage: str, response: Any) -> None:
        usage = getattr(response, "usage", None)
        prompt = getattr(usage, "prompt_tokens", None)
        completion = getattr(usage, "completion_tokens", None)
        if prompt is None or completion is None:
            if stage not in self.usage:
                self.usage[stage] = {"status": "unavailable"}
            return
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", None)
        try:
            prompt = int(prompt)
            completion = int(completion)
            cached = int(cached) if cached is not None else None
            if min(prompt, completion) < 0 or (cached is not None and not 0 <= cached <= prompt):
                raise ValueError("invalid token count")
        except (TypeError, ValueError):
            if stage not in self.usage:
                self.usage[stage] = {"status": "unavailable"}
            return
        previous = self.usage.get(stage, {})
        self.reported_attempts[stage] += 1
        self.usage[stage] = {
            "status": "reported",
            "prompt_tokens": previous.get("prompt_tokens", 0) + prompt,
            "completion_tokens": previous.get("completion_tokens", 0) + completion,
            "cached_prompt_tokens": previous.get("cached_prompt_tokens", 0) or 0,
            "reported_attempts": self.reported_attempts[stage],
        }
        if cached is not None:
            self.usage[stage]["cached_prompt_tokens"] += cached

    def finish(self, status: str, error_category: str | None = None) -> None:
        self.status = status
        self.error_category = error_category
        self.total_ms = round((perf_counter() - self.started) * 1000, 3)

    def _token_totals(self) -> tuple[str, int | None, int | None]:
        called = [name for name, attempts in self.attempts.items() if attempts]
        if not called:
            return "unavailable", None, None
        if any(self.reported_attempts[name] != self.attempts[name] for name in called):
            # Failed attempts may have consumed unreported tokens.
            status = "partial" if any(self.reported_attempts.values()) else "unavailable"
            return status, None, None
        return (
            "reported",
            sum(self.usage[name]["prompt_tokens"] for name in called),
            sum(self.usage[name]["completion_tokens"] for name in called),
        )

    def storage_values(self) -> dict[str, Any]:
        token_status, prompt_tokens, completion_tokens = self._token_totals()
        stage_usage = {}
        for name in ("planning", "answer"):
            usage = dict(self.usage.get(name, {"status": "unavailable"}))
            if 0 < self.reported_attempts[name] < self.attempts[name]:
                usage["status"] = "partial"
            stage_usage[name] = usage
        return {
            "planning_ms": self.stage_ms["planning"],
            "retrieval_ms": self.stage_ms["retrieval"],
            "answer_ms": self.stage_ms["answer"],
            "total_ms": self.total_ms,
            "planner_provider": self.planner_provider,
            "answer_provider": self.answer_provider,
            "planner_prompt_version": self.planner_prompt_version,
            "answer_prompt_version": self.answer_prompt_version,
            "token_usage_status": token_status,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "token_usage": stage_usage,
            "retry_count": sum(max(0, count - 1) for count in self.attempts.values()),
            "fallback_count": self.fallback_count,
            "error_category": self.error_category,
        }

    def _span_attributes(self, span: Any) -> None:
        values = self.storage_values()
        span.set_attribute("signalforge.status", self.status)
        for name in (
            "planning_ms",
            "retrieval_ms",
            "answer_ms",
            "total_ms",
            "retry_count",
            "fallback_count",
            "prompt_tokens",
            "completion_tokens",
        ):
            if values[name] is not None:
                span.set_attribute(f"signalforge.{name}", values[name])
        for name in (
            "planner_provider",
            "answer_provider",
            "planner_prompt_version",
            "answer_prompt_version",
            "token_usage_status",
            "error_category",
        ):
            if values[name] is not None:
                span.set_attribute(f"signalforge.{name}", values[name])
        for name in ("planner_model", "answer_model"):
            model = getattr(self, name)
            if model is not None:
                span.set_attribute(f"signalforge.{name}", model)
