import json
from types import SimpleNamespace

from answer_generator import AnswerGenerator, ExtractiveAnswerGenerator, format_evidence
from answer_query import select_ready_accessions_by_ticker, years_for_plan_scope
from evaluate_answers import evaluate_answer_quality
from query_planner import PlannerContext, SearchPlan
from vector_store import SearchResult


def test_format_evidence_labels_chunks_with_source_metadata():
    chunks = [
        SearchResult(
            score=0.91,
            payload={
                "ticker": "NVDA",
                "filing_date": "2026-02-25",
                "section_id": "1A",
                "section_title": "Risk Factors",
                "chunk_index": 4,
                "accession_number": "0001045810-26-000021",
                "text": "Supply chain risk.",
            },
        )
    ]

    evidence = format_evidence(chunks)

    assert evidence[0].label == "[1] NVDA 2026 Item 1A chunk 4"
    assert evidence[0].metadata["accession_number"] == "0001045810-26-000021"
    assert evidence[0].text == "Supply chain risk."


def test_answer_generator_sends_question_plan_and_evidence_to_model():
    client = FakeClient("NVIDIA cites supply-chain risks [1].")
    generator = AnswerGenerator(client=client, model="answer-model")
    plan = SearchPlan(
        tickers=["NVDA"],
        sections=["1A"],
        semantic_queries=["NVIDIA risk factors"],
        time_scope="latest",
        intent="summary",
        top_k=5,
    )
    chunks = [
        SearchResult(
            score=0.91,
            payload={
                "ticker": "NVDA",
                "filing_date": "2026-02-25",
                "section_id": "1A",
                "chunk_index": 4,
                "accession_number": "0001045810-26-000021",
                "text": "Supply chain risk.",
            },
        )
    ]

    generated = generator.generate(question="What are NVIDIA's risks?", plan=plan, chunks=chunks)
    user_payload = json.loads(client.request["messages"][1]["content"])

    assert generated.answer == "NVIDIA cites supply-chain risks [1]."
    assert generated.evidence_labels == ["[1] NVDA 2026 Item 1A chunk 4"]
    assert user_payload["question"] == "What are NVIDIA's risks?"
    assert user_payload["retrieval_plan"]["tickers"] == ["NVDA"]
    assert user_payload["evidence"][0]["text"] == "Supply chain risk."
    assert client.request["model"] == "answer-model"
    assert client.request["temperature"] == 0.0


def test_answer_generator_returns_unavailable_year_without_model_call():
    client = FakeClient("should not be used")
    generator = AnswerGenerator(client=client)
    plan = SearchPlan(
        tickers=["MSFT"],
        sections=["1A"],
        semantic_queries=["Microsoft risk factors"],
        time_scope="specific_years",
        filing_years=[],
        intent="summary",
    )

    generated = generator.generate(
        question="Show me Microsoft's risk factors from 2026.",
        plan=plan,
        chunks=[],
        available_years_by_ticker={"MSFT": (2025, 2024, 2023, 2022, 2021)},
    )

    assert "requested filing year is not available" in generated.answer
    assert "MSFT: 2025, 2024, 2023, 2022, 2021" in generated.answer
    assert generated.evidence_labels == []
    assert generated.warnings == ["no retrieved evidence"]
    assert client.request is None


def test_answer_generator_returns_no_supported_ticker_without_model_call():
    client = FakeClient("should not be used")
    generator = AnswerGenerator(client=client)
    plan = SearchPlan(
        tickers=[],
        sections=["7"],
        semantic_queries=["Intel revenue last year"],
        time_scope="latest",
        intent="fact",
    )

    generated = generator.generate(
        question="What is the reported revenue of Intel last year?",
        plan=plan,
        chunks=[],
        available_years_by_ticker={"AMD": (2026,), "NVDA": (2026,)},
    )

    assert "did not match an indexed company ticker" in generated.answer
    assert "AMD, NVDA" in generated.answer
    assert client.request is None


def test_extractive_answer_generator_returns_retrieved_evidence_without_model_call():
    generator = ExtractiveAnswerGenerator()
    plan = SearchPlan(
        tickers=["NVDA"],
        sections=["1A"],
        semantic_queries=["NVIDIA risk factors"],
        time_scope="latest",
        intent="summary",
    )
    chunks = [
        SearchResult(
            score=0.91,
            payload={
                "ticker": "NVDA",
                "filing_date": "2026-02-25",
                "section_id": "1A",
                "chunk_index": 4,
                "accession_number": "0001045810-26-000021",
                "text": "Supply chain risk.",
            },
        )
    ]

    generated = generator.generate(
        question="What are NVIDIA's risks?",
        plan=plan,
        chunks=chunks,
    )

    assert "DEEPSEEK_API_KEY is not set" in generated.answer
    assert "[1] NVDA 2026 Item 1A chunk 4: Supply chain risk." in generated.answer
    assert generated.evidence_labels == ["[1] NVDA 2026 Item 1A chunk 4"]
    assert generated.warnings == ["llm answer generation unavailable"]


def test_empty_ticker_plan_does_not_select_broad_accessions():
    context = PlannerContext(
        available_tickers=("AMD", "NVDA"),
        available_sections=("7",),
        filing_years_by_ticker={"AMD": (2026,), "NVDA": (2026,)},
    )
    plan = SearchPlan(
        tickers=[],
        sections=["7"],
        semantic_queries=["Intel revenue last year"],
        time_scope="latest",
        intent="fact",
    )

    accessions = select_ready_accessions_by_ticker(
        object(),
        plan=plan,
        context=context,
        embedding_model="model",
        collection="collection",
    )

    assert accessions == {}


def test_years_for_plan_scope_selects_periods_from_context():
    context = PlannerContext(
        available_tickers=("QCOM",),
        available_sections=("1A",),
        filing_years_by_ticker={"QCOM": (2025, 2024, 2023, 2022, 2021)},
    )
    plan = SearchPlan(
        tickers=["QCOM"],
        sections=["1A"],
        semantic_queries=["Qualcomm risk trend"],
        time_scope="latest_and_previous",
        intent="comparison",
    )

    assert years_for_plan_scope(plan, ticker="QCOM", context=context) == [2025, 2024]


def test_answer_quality_requires_citations_when_chunks_exist():
    plan = {
        "tickers": ["NVDA"],
        "sections": ["1A"],
        "semantic_queries": ["NVIDIA risks"],
        "time_scope": "latest",
        "filing_years": [],
        "intent": "summary",
        "top_k": 5,
    }
    expected = {
        "tickers": ["NVDA"],
        "sections": ["1A"],
        "time_scope": "latest",
        "intent": "summary",
    }
    context = PlannerContext(
        available_tickers=("NVDA",),
        available_sections=("1A",),
        filing_years_by_ticker={"NVDA": (2026,)},
    )

    failures = evaluate_answer_quality(
        answer="NVIDIA has supply chain risks.",
        chunks=[SearchResult(score=0.9, payload={"ticker": "NVDA"})],
        actual_plan=plan,
        expected=expected,
        context=context,
        used_fallback=False,
        plan_error=None,
        use_expected_plan=True,
    )

    assert failures == ["answer has retrieved chunks but no citation labels"]


class FakeClient:
    def __init__(self, response: str):
        self.response = response
        self.request = None
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=self._create,
            )
        )

    def _create(self, **kwargs):
        self.request = kwargs
        message = SimpleNamespace(content=self.response)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])
