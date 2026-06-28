import json
from types import SimpleNamespace

from plan_query import build_planner_context
from query_planner import DeepSeekQueryPlanner, PlannerContext


def test_deepseek_planner_returns_normalized_plan():
    client = FakeClient(
        {
            "tickers": ["nvda", "UNKNOWN"],
            "sections": ["1A", "7", "8"],
            "semantic_queries": [
                "supplier dependency",
                "manufacturing capacity",
            ],
            "time_scope": "latest",
            "intent": "summary",
            "top_k": 8,
        }
    )
    context = PlannerContext(
        available_tickers=("NVDA",),
        available_sections=("1", "1A", "7", "7A"),
        filing_years_by_ticker={"NVDA": (2026,)},
    )

    result = DeepSeekQueryPlanner(client=client).create_plan(
        "What are NVIDIA's latest supply-chain risks?",
        context,
    )

    assert result.used_fallback is False
    assert result.error is None
    assert result.plan.tickers == ["NVDA"]
    assert result.plan.sections == ["1A", "7"]
    assert result.plan.semantic_queries == [
        "What are NVIDIA's latest supply-chain risks?",
        "supplier dependency",
        "manufacturing capacity",
    ]
    assert client.request["model"] == "deepseek-v4-flash"
    assert client.request["response_format"] == {"type": "json_object"}
    assert client.request["extra_body"] == {"thinking": {"type": "disabled"}}


def test_deepseek_planner_falls_back_when_response_is_invalid():
    client = FakeClient({"answer": "NVIDIA has supply-chain risk."})
    context = PlannerContext(
        available_tickers=("NVDA",),
        available_sections=("1A",),
        filing_years_by_ticker={"NVDA": (2026,)},
    )

    result = DeepSeekQueryPlanner(client=client).create_plan(
        "What are NVIDIA's risks?",
        context,
    )

    assert result.used_fallback is True
    assert result.error
    assert result.plan.semantic_queries == ["What are NVIDIA's risks?"]
    assert result.plan.tickers == []
    assert result.plan.sections == []
    assert result.plan.time_scope == "latest"


def test_deepseek_planner_accepts_empty_model_semantic_queries():
    client = FakeClient(
        {
            "tickers": [],
            "sections": [],
            "semantic_queries": [],
            "time_scope": "latest",
            "intent": "summary",
            "top_k": 5,
        }
    )
    context = PlannerContext(
        available_tickers=("AMD",),
        available_sections=("1A",),
        filing_years_by_ticker={"AMD": (2026,)},
    )

    result = DeepSeekQueryPlanner(client=client).create_plan(
        "Summarize Intel's latest risks.",
        context,
    )

    assert result.used_fallback is False
    assert result.plan.tickers == []
    assert result.plan.sections == ["1A"]
    assert result.plan.semantic_queries == ["Summarize Intel's latest risks."]


def test_deepseek_planner_retries_an_empty_response():
    client = EmptyThenValidClient()
    context = PlannerContext(
        available_tickers=("NVDA",),
        available_sections=("1A",),
        filing_years_by_ticker={"NVDA": (2026,)},
    )

    result = DeepSeekQueryPlanner(client=client).create_plan(
        "Summarize NVIDIA's risks.",
        context,
    )

    assert result.used_fallback is False
    assert client.call_count == 2
    assert result.plan.sections == ["1A"]


def test_build_planner_context_uses_database_metadata():
    rows = [
        {"ticker": "NVDA", "filing_date": "2026-02-25", "section_id": "1A"},
        {"ticker": "NVDA", "filing_date": "2025-02-26", "section_id": "7"},
        {"ticker": "AMD", "filing_date": "2026-02-04", "section_id": "8"},
    ]

    context = build_planner_context(rows)

    assert context.available_tickers == ("AMD", "NVDA")
    assert context.available_sections == ("1A", "7")
    assert context.filing_years_by_ticker == {
        "AMD": (2026,),
        "NVDA": (2026, 2025),
    }


class FakeClient:
    def __init__(self, response: dict):
        self.response = response
        self.request = None
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=self._create,
            )
        )

    def _create(self, **kwargs):
        self.request = kwargs
        message = SimpleNamespace(content=json.dumps(self.response))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class EmptyThenValidClient:
    def __init__(self):
        self.call_count = 0
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=self._create,
            )
        )

    def _create(self, **kwargs):
        self.call_count += 1
        if self.call_count == 1:
            content = None
        else:
            content = json.dumps(
                {
                    "tickers": ["NVDA"],
                    "sections": ["1A"],
                    "semantic_queries": ["NVIDIA risk factors"],
                    "time_scope": "latest",
                    "intent": "summary",
                    "top_k": 5,
                }
            )
        message = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])
