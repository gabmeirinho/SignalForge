from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from threading import Thread

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from signalforge.config import RuntimeConfig
from signalforge.answer_generator import AnswerGenerator
from signalforge.query_planner import DeepSeekQueryPlanner, PlannerContext, SearchPlan
from signalforge.telemetry import PRICE_CATALOG_VERSION, QueryTelemetry
from signalforge.vector_store import SearchResult


def test_reported_usage_prices_cached_and_uncached_tokens():
    metrics = QueryTelemetry(
        planner_provider="deepseek",
        answer_provider="deepseek",
        planner_model="deepseek-v4-flash",
        answer_model="deepseek-v4-flash",
        price_tier="peak",
    )
    metrics.record_attempt("planning")
    metrics.record_attempt("answer")
    metrics.record_usage(
        "planning",
        SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=1_000_000,
                completion_tokens=100_000,
                prompt_tokens_details=SimpleNamespace(cached_tokens=500_000),
            )
        ),
    )
    metrics.record_usage(
        "answer",
        SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=500_000,
                completion_tokens=100_000,
                prompt_tokens_details=None,
            )
        ),
    )
    metrics.finish("completed")
    values = metrics.storage_values()

    assert values["token_usage_status"] == "reported"
    assert values["prompt_tokens"] == 1_500_000
    assert values["completion_tokens"] == 200_000
    assert values["estimated_cost_usd"] == Decimal("0.54300000")
    assert values["price_catalog_version"] == PRICE_CATALOG_VERSION
    assert values["total_ms"] >= 0


def test_missing_usage_or_retries_make_total_cost_unavailable():
    metrics = QueryTelemetry(planner_provider="deepseek", planner_model="deepseek-v4-flash")
    metrics.record_attempt("planning")
    metrics.record_usage("planning", SimpleNamespace(usage=None))
    assert metrics.storage_values()["token_usage_status"] == "unavailable"
    assert metrics.storage_values()["estimated_cost_usd"] is None

    metrics.record_attempt("planning")
    metrics.record_usage(
        "planning",
        SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=5,
                prompt_tokens_details=None,
            )
        ),
    )
    assert metrics.storage_values()["token_usage_status"] == "partial"
    assert metrics.storage_values()["retry_count"] == 1
    assert metrics.storage_values()["estimated_cost_usd"] is None


def test_retries_with_reported_usage_sum_every_attempt():
    metrics = QueryTelemetry(planner_provider="deepseek", planner_model="deepseek-v4-flash")
    for prompt, completion in ((10, 5), (20, 7)):
        metrics.record_attempt("planning")
        metrics.record_usage(
            "planning",
            SimpleNamespace(
                usage=SimpleNamespace(
                    prompt_tokens=prompt,
                    completion_tokens=completion,
                    prompt_tokens_details=None,
                )
            ),
        )
    values = metrics.storage_values()
    assert values["retry_count"] == 1
    assert values["token_usage_status"] == "reported"
    assert values["prompt_tokens"] == 30
    assert values["completion_tokens"] == 12
    assert values["estimated_cost_usd"] is not None


def test_model_calls_capture_provider_reported_usage():
    class Client:
        def __init__(self, content):
            response = SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                usage=SimpleNamespace(
                    prompt_tokens=40,
                    completion_tokens=10,
                    prompt_tokens_details=SimpleNamespace(cached_tokens=5),
                ),
            )
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: response)
            )

    metrics = QueryTelemetry(
        planner_provider="deepseek",
        answer_provider="deepseek",
        planner_model="deepseek-v4-flash",
        answer_model="deepseek-v4-flash",
    )
    context = PlannerContext(
        available_tickers=("NVDA",),
        available_sections=("1A",),
        filing_years_by_ticker={"NVDA": (2026,)},
    )
    with metrics.activate():
        result = DeepSeekQueryPlanner(
            client=Client(
                '{"tickers": ["NVDA"], "sections": ["1A"], "semantic_queries": [], '
                '"time_scope": "latest", "intent": "summary"}'
            )
        ).create_plan("What are NVIDIA's risks?", context)
        AnswerGenerator(client=Client("Supply risk [1].")).generate(
            question="What are NVIDIA's risks?",
            plan=SearchPlan(tickers=["NVDA"], sections=["1A"], semantic_queries=["NVIDIA risks"]),
            chunks=[
                SearchResult(
                    score=0.9,
                    payload={
                        "ticker": "NVDA",
                        "filing_date": "2026-01-01",
                        "section_id": "1A",
                        "chunk_index": 0,
                        "text": "Supply risk.",
                    },
                )
            ],
        )
        metrics.finish("completed")

    assert result.used_fallback is False
    assert metrics.storage_values()["token_usage_status"] == "reported"
    assert metrics.storage_values()["prompt_tokens"] == 80
    assert metrics.storage_values()["completion_tokens"] == 20
    assert metrics.attempts == {"planning": 1, "answer": 1}


def test_trace_spans_contain_metrics_without_question_or_source_text(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr("signalforge.telemetry._tracer", lambda *args: provider.get_tracer("test"))
    config = RuntimeConfig(
        tracing_enabled=True,
        otlp_traces_endpoint="https://example.test/api/public/otel/v1/traces",
        langfuse_public_key="public",
        langfuse_secret_key="secret",
    )
    metrics = QueryTelemetry(config=config, run_id=7, planner_model="deepseek-v4-flash")
    with metrics.activate():
        with metrics.stage("planning"):
            metrics.record_attempt("planning")
            metrics.record_usage("planning", SimpleNamespace(usage=None))
        metrics.finish("failed", "RuntimeError")

    spans = exporter.get_finished_spans()
    assert {span.name for span in spans} == {"signalforge.query", "signalforge.planning"}
    root = next(span for span in spans if span.name == "signalforge.query")
    assert root.attributes["signalforge.query_run.id"] == 7
    assert root.attributes["signalforge.error_category"] == "RuntimeError"
    assert root.attributes["signalforge.token_usage_status"] == "unavailable"
    assert "question" not in str(root.attributes).lower()
    assert "secret" not in str(root.attributes).lower()


def test_configured_otlp_exporter_sends_trace_with_basic_auth(monkeypatch):
    requests = []

    class Receiver(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(
                (
                    self.path,
                    dict(self.headers),
                    self.rfile.read(int(self.headers["Content-Length"])),
                )
            )
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Receiver)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    import signalforge.telemetry as telemetry_module

    monkeypatch.setattr(telemetry_module, "BatchSpanProcessor", SimpleSpanProcessor)
    telemetry_module._tracer.cache_clear()
    try:
        config = RuntimeConfig(
            tracing_enabled=True,
            otlp_traces_endpoint=f"http://127.0.0.1:{server.server_port}/api/public/otel/v1/traces",
            langfuse_public_key="public",
            langfuse_secret_key="secret",
        )
        metrics = QueryTelemetry(config=config)
        with metrics.activate():
            metrics.finish("completed")
    finally:
        server.shutdown()
        thread.join()
        telemetry_module._tracer.cache_clear()

    assert len(requests) == 1
    path, headers, body = requests[0]
    assert path == "/api/public/otel/v1/traces"
    assert headers["Authorization"] == "Basic cHVibGljOnNlY3JldA=="
    assert headers["x-langfuse-ingestion-version"] == "4"
    assert body
