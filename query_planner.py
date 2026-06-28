import json
import os
from dataclasses import dataclass
from typing import Literal, Protocol

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from tenacity import retry, stop_after_attempt, wait_fixed


DEFAULT_PLANNER_MODEL = "deepseek-v4-flash"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
PLANNER_PROMPT_VERSION = "planner-v1.2.0"
PLANNER_TEMPERATURE = 0.0
SUPPORTED_SECTIONS = ("1", "1A", "7", "7A")


class SearchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tickers: list[str] = Field(default_factory=list)
    sections: list[Literal["1", "1A", "7", "7A"]] = Field(default_factory=list)
    semantic_queries: list[str] = Field(min_length=1, max_length=4)
    time_scope: Literal["latest", "latest_and_previous", "all_available"] = "latest"
    intent: Literal["fact", "summary", "comparison", "trend"] = "summary"
    top_k: int = Field(default=5, ge=1, le=20)


class PlannerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tickers: list[str] = Field(default_factory=list)
    sections: list[str] = Field(default_factory=list)
    semantic_queries: list[str] = Field(max_length=3)
    time_scope: Literal["latest", "latest_and_previous", "all_available"] = "latest"
    intent: Literal["fact", "summary", "comparison", "trend"] = "summary"
    top_k: int = Field(default=5, ge=1, le=20)


@dataclass(frozen=True)
class PlannerContext:
    available_tickers: tuple[str, ...]
    available_sections: tuple[str, ...]
    filing_years_by_ticker: dict[str, tuple[int, ...]]

    def to_prompt_dict(self) -> dict:
        return {
            "available_tickers": list(self.available_tickers),
            "available_sections": list(self.available_sections),
            "available_filing_years": {
                ticker: list(years) for ticker, years in self.filing_years_by_ticker.items()
            },
        }


@dataclass(frozen=True)
class PlanningResult:
    plan: SearchPlan
    used_fallback: bool
    error: str | None = None


class ChatCompletionsClient(Protocol):
    @property
    def chat(self): ...


class DeepSeekQueryPlanner:
    def __init__(
        self,
        *,
        client: ChatCompletionsClient,
        model: str = DEFAULT_PLANNER_MODEL,
    ) -> None:
        self.client = client
        self.model = model

    @classmethod
    def from_environment(
        cls,
        *,
        model: str = DEFAULT_PLANNER_MODEL,
    ) -> "DeepSeekQueryPlanner":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")

        client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL),
        )
        return cls(client=client, model=model)

    def create_plan(self, question: str, context: PlannerContext) -> PlanningResult:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")

        try:
            content = self._request_plan(question=question, context=context)
            raw_plan = PlannerResponse.model_validate_json(content)
            plan = normalize_plan(raw_plan, question=question, context=context)
            return PlanningResult(plan=plan, used_fallback=False)
        except (ValidationError, ValueError, TypeError, KeyError, IndexError) as error:
            return PlanningResult(
                plan=fallback_plan(question),
                used_fallback=True,
                error=str(error),
            )
        except Exception as error:
            return PlanningResult(
                plan=fallback_plan(question),
                used_fallback=True,
                error=f"{type(error).__name__}: {error}",
            )

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(0.5), reraise=True)
    def _request_plan(self, *, question: str, context: PlannerContext) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _system_prompt()},
                {
                    "role": "user",
                    "content": _user_prompt(question=question, context=context),
                },
            ],
            response_format={"type": "json_object"},
            max_tokens=800,
            temperature=PLANNER_TEMPERATURE,
            extra_body={"thinking": {"type": "disabled"}},
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("DeepSeek returned an empty planner response")
        # print(_system_prompt())
        # print(_user_prompt(question=question, context=context))
        # print(content)
        return content


def normalize_plan(
    plan: PlannerResponse,
    *,
    question: str,
    context: PlannerContext,
) -> SearchPlan:
    available_tickers = set(context.available_tickers)
    available_sections = set(context.available_sections)

    tickers = _unique(
        ticker.upper() for ticker in plan.tickers if ticker.upper() in available_tickers
    )
    sections = _unique(section for section in plan.sections if section in available_sections)
    if not sections:
        sections = [
            section
            for section in _infer_sections_from_question(question)
            if section in available_sections
        ]

    queries = _unique(
        query.strip() for query in [question, *plan.semantic_queries] if query.strip()
    )[:4]

    return SearchPlan(
        tickers=tickers,
        sections=sections,
        semantic_queries=queries,
        time_scope=plan.time_scope,
        intent=plan.intent,
        top_k=plan.top_k,
    )


def fallback_plan(question: str) -> SearchPlan:
    question = question.strip()
    if not question:
        raise ValueError("question must not be empty")

    return SearchPlan(
        semantic_queries=[question],
        time_scope="latest",
        intent="summary",
        top_k=5,
    )


def _infer_sections_from_question(question: str) -> list[str]:
    question = question.lower()

    if any(
        term in question
        for term in (
            "market risk",
            "interest rate",
            "foreign currency",
            "exchange rate",
            "commodity price",
            "credit risk",
            "sensitivity analysis",
        )
    ):
        return ["7A"]

    if any(term in question for term in ("risk factor", "risk factors", "risks")):
        return ["1A"]

    if any(
        term in question
        for term in (
            "md&a",
            "management discussion",
            "financial performance",
            "revenue",
            "expenses",
            "margins",
            "profitability",
            "results of operations",
            "liquidity",
            "capital resources",
            "cash flows",
            "management outlook",
        )
    ):
        return ["7"]

    if any(
        term in question
        for term in (
            "business overview",
            "business",
            "products",
            "services",
            "customers",
            "segments",
            "strategy",
            "operations",
        )
    ):
        return ["1"]

    return []


def _system_prompt() -> str:
    return """
You are a query planner for a local SEC 10-K search system.

Convert the user's question into a JSON retrieval plan. Do not answer the question.
Do not claim that a company, filing, date, or section exists unless it is present in
the supplied database context.

Rules:
- Output one JSON object and no surrounding prose.
- Use only tickers and sections supplied in the database context.
- If the requested company is not available in the database context, do not choose a peer or similar company.
- Generate one to three concise semantic search queries.
- Prefer the smallest sufficient set of sections.
- Use section "1" for business overview, products, services, customers, segments,
  strategy, and operations.
- Use section "1A" for risk factors, business risks, material risks,
  cybersecurity risks, regulatory risks, and litigation risks.
- Use section "7" for MD&A, management discussion, financial performance,
  revenue, expenses, margins, profitability, results of operations, liquidity,
  capital resources, cash flows, and management outlook.
- Use section "7A" only for quantitative or qualitative market risk, interest
  rate risk, foreign currency risk, exchange rate risk, commodity price risk,
  credit risk, or sensitivity analysis.
- Do not include section "7A" just because the question is financial. Include
  "7A" only when the question explicitly asks about market risk or one of its
  subtypes.
- Use "latest" for questions about the newest available information.
- Use "latest_and_previous" only when comparing the latest period with an earlier
  period and at least two filing years are available for the selected ticker.
- Use "all_available" for historical trends only when at least two filing years
  are available for the selected ticker.
- If the user requests a period that is not available, use "latest" while preserving
  the user's comparison or trend intent. Never invent a missing filing period.
- Use intent "fact" when the user asks for a specific reported value, metric,
  amount, figure, date, or number.
- Use intent "summary" when the user asks to summarize, describe, explain, or
  provide an overview.
- Use intent "comparison" only for explicit company or period comparisons.
- Use intent "trend" for change-over-time questions spanning multiple periods.
- top_k must be between 1 and 20.

Required JSON shape:
{
  "tickers": ["NVDA"],
  "sections": ["1A", "7"],
  "semantic_queries": ["supply chain risks", "supplier dependency and shortages"],
  "time_scope": "latest",
  "intent": "summary",
  "top_k": 8
}
""".strip()


def _user_prompt(*, question: str, context: PlannerContext) -> str:
    return json.dumps(
        {
            "database_context": context.to_prompt_dict(),
            "question": question,
        },
        ensure_ascii=False,
    )


def _unique(values) -> list[str]:
    result = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
