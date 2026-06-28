import json
from types import SimpleNamespace

from plan_query import build_planner_context
from query_planner import DeepSeekQueryPlanner, LocalQueryPlanner, PlannerContext


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
    assert result.plan.sections == ["1A"]
    assert result.plan.semantic_queries == [
        "What are NVIDIA's latest supply-chain risks?",
        "supplier dependency",
        "manufacturing capacity",
    ]
    assert client.request["model"] == "deepseek-v4-flash"
    assert client.request["response_format"] == {"type": "json_object"}
    assert client.request["temperature"] == 0.0
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


def test_deepseek_planner_normalizes_specific_filing_years():
    client = FakeClient(
        {
            "tickers": ["AVGO"],
            "sections": ["1"],
            "semantic_queries": ["Broadcom business operations"],
            "time_scope": "specific_years",
            "filing_years": [2022],
            "intent": "fact",
            "top_k": 5,
        }
    )
    context = PlannerContext(
        available_tickers=("AVGO",),
        available_sections=("1",),
        filing_years_by_ticker={"AVGO": (2025, 2024, 2023, 2022, 2021)},
    )

    result = DeepSeekQueryPlanner(client=client).create_plan(
        "What business operations did Broadcom report in 2022?",
        context,
    )

    assert result.used_fallback is False
    assert result.plan.time_scope == "specific_years"
    assert result.plan.filing_years == [2022]


def test_deepseek_planner_filters_unavailable_specific_filing_years():
    client = FakeClient(
        {
            "tickers": ["MSFT"],
            "sections": ["1A"],
            "semantic_queries": ["Microsoft risk factors"],
            "time_scope": "specific_years",
            "filing_years": [2026],
            "intent": "summary",
            "top_k": 5,
        }
    )
    context = PlannerContext(
        available_tickers=("MSFT",),
        available_sections=("1A",),
        filing_years_by_ticker={"MSFT": (2025, 2024, 2023, 2022, 2021)},
    )

    result = DeepSeekQueryPlanner(client=client).create_plan(
        "Show me Microsoft's risk factors from 2026.",
        context,
    )

    assert result.used_fallback is False
    assert result.plan.time_scope == "specific_years"
    assert result.plan.filing_years == []


def test_deepseek_planner_infers_oldest_available_filing_year():
    client = FakeClient(
        {
            "tickers": ["AMZN"],
            "sections": ["1A"],
            "semantic_queries": ["Amazon primary risk concerns"],
            "time_scope": "latest",
            "intent": "summary",
            "top_k": 5,
        }
    )
    context = PlannerContext(
        available_tickers=("AMZN",),
        available_sections=("1A",),
        filing_years_by_ticker={"AMZN": (2026, 2025, 2024, 2023, 2022)},
    )

    result = DeepSeekQueryPlanner(client=client).create_plan(
        "What were Amazon's primary risk concerns in the oldest filing available in the system?",
        context,
    )

    assert result.used_fallback is False
    assert result.plan.time_scope == "specific_years"
    assert result.plan.filing_years == [2022]


def test_deepseek_planner_keeps_open_ended_trends_all_available():
    client = FakeClient(
        {
            "tickers": ["QCOM"],
            "sections": ["1A"],
            "semantic_queries": ["Qualcomm risk factors changed over time"],
            "time_scope": "specific_years",
            "intent": "trend",
            "top_k": 5,
        }
    )
    context = PlannerContext(
        available_tickers=("QCOM",),
        available_sections=("1A",),
        filing_years_by_ticker={"QCOM": (2025, 2024, 2023, 2022, 2021)},
    )

    result = DeepSeekQueryPlanner(client=client).create_plan(
        "How have Qualcomm's risk factors changed over time?",
        context,
    )

    assert result.used_fallback is False
    assert result.plan.time_scope == "all_available"
    assert result.plan.filing_years == []


def test_deepseek_planner_distinguishes_market_risks_from_risk_factors():
    client = FakeClient(
        {
            "tickers": ["MU"],
            "sections": ["1", "1A"],
            "semantic_queries": ["Micron business description and market risks"],
            "time_scope": "latest",
            "intent": "summary",
            "top_k": 5,
        }
    )
    context = PlannerContext(
        available_tickers=("MU",),
        available_sections=("1", "1A", "7A"),
        filing_years_by_ticker={"MU": (2025, 2024, 2023, 2022, 2021)},
    )

    result = DeepSeekQueryPlanner(client=client).create_plan(
        "Give me a complete overview of Micron's business description and market risks from their latest filing.",
        context,
    )

    assert result.used_fallback is False
    assert result.plan.sections == ["1", "7A"]


def test_deepseek_planner_drops_unsupported_item_sections():
    client = FakeClient(
        {
            "tickers": ["NVDA"],
            "sections": ["1", "1A", "7", "7A"],
            "semantic_queries": ["NVIDIA executive compensation tables"],
            "time_scope": "latest",
            "intent": "summary",
            "top_k": 5,
        }
    )
    context = PlannerContext(
        available_tickers=("NVDA",),
        available_sections=("1", "1A", "7", "7A"),
        filing_years_by_ticker={"NVDA": (2026, 2025, 2024, 2023, 2022)},
    )

    result = DeepSeekQueryPlanner(client=client).create_plan(
        "Can you show me the executive compensation tables (Item 11) for NVIDIA?",
        context,
    )

    assert result.used_fallback is False
    assert result.plan.sections == []
    assert result.plan.intent == "fact"


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


def test_local_planner_infers_company_section_year_and_scope():
    context = PlannerContext(
        available_tickers=("NVDA",),
        available_sections=("1", "1A", "7", "7A"),
        filing_years_by_ticker={"NVDA": (2026, 2025, 2024)},
        company_names_by_ticker={"NVDA": ("NVIDIA CORP",)},
    )

    result = LocalQueryPlanner().create_plan(
        "Summarize NVIDIA's risk factors from 2025.",
        context,
    )

    assert result.used_fallback is True
    assert result.plan.tickers == ["NVDA"]
    assert result.plan.sections == ["1A"]
    assert result.plan.time_scope == "specific_years"
    assert result.plan.filing_years == [2025]
    assert result.plan.semantic_queries == ["Summarize NVIDIA's risk factors from 2025."]


def test_build_planner_context_uses_database_metadata():
    rows = [
        {
            "ticker": "NVDA",
            "company_name": "NVIDIA CORP",
            "filing_date": "2026-02-25",
            "section_id": "1A",
        },
        {
            "ticker": "NVDA",
            "company_name": "NVIDIA CORP",
            "filing_date": "2025-02-26",
            "section_id": "7",
        },
        {
            "ticker": "AMD",
            "company_name": "ADVANCED MICRO DEVICES INC",
            "filing_date": "2026-02-04",
            "section_id": "8",
        },
    ]

    context = build_planner_context(rows)

    assert context.available_tickers == ("AMD", "NVDA")
    assert context.available_sections == ("1A", "7")
    assert context.filing_years_by_ticker == {
        "AMD": (2026,),
        "NVDA": (2026, 2025),
    }
    assert context.company_names_by_ticker == {
        "AMD": ("ADVANCED MICRO DEVICES INC",),
        "NVDA": ("NVIDIA CORP",),
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
