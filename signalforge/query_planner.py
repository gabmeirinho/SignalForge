import json
import os
import re
from dataclasses import dataclass
from typing import Literal, Protocol

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from tenacity import retry, stop_after_attempt, wait_fixed


DEFAULT_PLANNER_MODEL = "deepseek-v4-flash"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
PLANNER_PROMPT_VERSION = "planner-v1.3.0"
PLANNER_TEMPERATURE = 0.0
SUPPORTED_SECTIONS = ("1", "1A", "7", "7A")
TIME_SCOPES = ("latest", "latest_and_previous", "all_available", "specific_years")


class SearchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tickers: list[str] = Field(default_factory=list)
    sections: list[Literal["1", "1A", "7", "7A"]] = Field(default_factory=list)
    semantic_queries: list[str] = Field(min_length=1, max_length=4)
    time_scope: Literal[
        "latest", "latest_and_previous", "all_available", "specific_years"
    ] = "latest"
    filing_years: list[int] = Field(default_factory=list)
    intent: Literal["fact", "summary", "comparison", "trend"] = "summary"
    top_k: int = Field(default=5, ge=1, le=20)


class PlannerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tickers: list[str] = Field(default_factory=list)
    sections: list[str] = Field(default_factory=list)
    semantic_queries: list[str] = Field(max_length=3)
    time_scope: Literal[
        "latest", "latest_and_previous", "all_available", "specific_years"
    ] = "latest"
    filing_years: list[int] = Field(default_factory=list)
    intent: Literal["fact", "summary", "comparison", "trend"] = "summary"
    top_k: int = Field(default=5, ge=1, le=20)


@dataclass(frozen=True)
class PlannerContext:
    available_tickers: tuple[str, ...]
    available_sections: tuple[str, ...]
    filing_years_by_ticker: dict[str, tuple[int, ...]]
    company_names_by_ticker: dict[str, tuple[str, ...]] | None = None

    def to_prompt_dict(self) -> dict:
        return {
            "available_tickers": list(self.available_tickers),
            "available_sections": list(self.available_sections),
            "available_filing_years": {
                ticker: list(years) for ticker, years in self.filing_years_by_ticker.items()
            },
            "company_names_by_ticker": {
                ticker: list(names)
                for ticker, names in (self.company_names_by_ticker or {}).items()
            },
        }


def build_planner_context(rows) -> PlannerContext:
    tickers = set()
    sections = set()
    years_by_ticker: dict[str, set[int]] = {}
    company_names_by_ticker: dict[str, set[str]] = {}

    for row in rows:
        ticker = str(row["ticker"]).upper()
        tickers.add(ticker)
        if "company_name" in row.keys() and row["company_name"]:
            company_names_by_ticker.setdefault(ticker, set()).add(str(row["company_name"]))

        section_id = str(row["section_id"]).upper()
        if section_id in SUPPORTED_SECTIONS:
            sections.add(section_id)

        filing_date = row["filing_date"]
        if filing_date and len(str(filing_date)) >= 4:
            try:
                years_by_ticker.setdefault(ticker, set()).add(int(str(filing_date)[:4]))
            except ValueError:
                pass

    return PlannerContext(
        available_tickers=tuple(sorted(tickers)),
        available_sections=tuple(section for section in SUPPORTED_SECTIONS if section in sections),
        filing_years_by_ticker={
            ticker: tuple(sorted(years, reverse=True))
            for ticker, years in sorted(years_by_ticker.items())
        },
        company_names_by_ticker={
            ticker: tuple(sorted(names))
            for ticker, names in sorted(company_names_by_ticker.items())
        },
    )


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


class LocalQueryPlanner:
    def create_plan(self, question: str, context: PlannerContext) -> PlanningResult:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")

        tickers = _infer_tickers_from_question(question, context)
        sections = _infer_sections_from_question(question) or []
        sections = [section for section in sections if section in set(context.available_sections)]
        intent = _infer_intent_from_question(question)
        time_scope = _infer_time_scope_from_question(question, intent=intent)
        available_years = _available_years_for_tickers(tickers, context)

        raw_plan = PlannerResponse(
            tickers=tickers,
            sections=sections,
            semantic_queries=[],
            time_scope=time_scope,
            filing_years=_infer_filing_years_from_question(question, available_years),
            intent=intent,
            top_k=5,
        )
        plan = normalize_plan(raw_plan, question=question, context=context)
        return PlanningResult(
            plan=plan,
            used_fallback=True,
            error="DEEPSEEK_API_KEY is not set; used local rule-based planner",
        )


def create_query_planner_from_environment(
    *,
    model: str = DEFAULT_PLANNER_MODEL,
) -> DeepSeekQueryPlanner | LocalQueryPlanner:
    if os.getenv("DEEPSEEK_API_KEY"):
        return DeepSeekQueryPlanner.from_environment(model=model)
    return LocalQueryPlanner()


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
    inferred_sections = _infer_sections_from_question(question)
    if inferred_sections is not None:
        sections = [section for section in inferred_sections if section in available_sections]

    queries = _unique(
        query.strip() for query in [question, *plan.semantic_queries] if query.strip()
    )[:4]
    time_scope, filing_years = _normalize_time_scope(
        plan=plan,
        question=question,
        tickers=tickers,
        context=context,
    )

    return SearchPlan(
        tickers=tickers,
        sections=sections,
        semantic_queries=queries,
        time_scope=time_scope,
        filing_years=filing_years,
        intent=_normalize_intent(plan.intent, question),
        top_k=plan.top_k,
    )


def fallback_plan(question: str) -> SearchPlan:
    question = question.strip()
    if not question:
        raise ValueError("question must not be empty")

    return SearchPlan(
        semantic_queries=[question],
        time_scope="latest",
        filing_years=[],
        intent="summary",
        top_k=5,
    )


def _normalize_time_scope(
    *,
    plan: PlannerResponse,
    question: str,
    tickers: list[str],
    context: PlannerContext,
) -> tuple[str, list[int]]:
    available_years = _available_years_for_tickers(tickers, context)
    has_specific_year_request = _question_requests_specific_years(question)
    requested_years = _unique_ints(plan.filing_years) if has_specific_year_request else []
    inferred_years = _infer_filing_years_from_question(question, available_years)

    if requested_years or inferred_years:
        years = requested_years or inferred_years
        return "specific_years", [year for year in years if year in available_years]

    if plan.time_scope == "specific_years":
        return "all_available" if plan.intent == "trend" else "latest", []

    return plan.time_scope, []


def _available_years_for_tickers(tickers: list[str], context: PlannerContext) -> set[int]:
    if tickers:
        years = set()
        for ticker in tickers:
            years.update(context.filing_years_by_ticker.get(ticker, ()))
        return years

    years = set()
    for ticker_years in context.filing_years_by_ticker.values():
        years.update(ticker_years)
    return years


def _infer_filing_years_from_question(question: str, available_years: set[int]) -> list[int]:
    question = question.lower()
    if "oldest" in question or "earliest" in question:
        return [min(available_years)] if available_years else []

    range_match = re.search(r"\b(20\d{2})\b\s+(?:and|to|-|through)\s+\b(20\d{2})\b", question)
    if range_match:
        start, end = sorted(int(year) for year in range_match.groups())
        return list(range(start, end + 1))

    return _unique_ints(int(year) for year in re.findall(r"\b(20\d{2})\b", question))


def _question_requests_specific_years(question: str) -> bool:
    question = question.lower()
    return (
        "oldest" in question
        or "earliest" in question
        or re.search(r"\b20\d{2}\b", question) is not None
    )


def _unique_ints(values) -> list[int]:
    result = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _infer_sections_from_question(question: str) -> list[str] | None:
    question = question.lower()
    item_match = re.search(r"\bitem\s+(\d+[a-z]?)\b", question)
    if item_match:
        item = item_match.group(1).upper()
        return [item] if item in SUPPORTED_SECTIONS else []

    sections = []

    if any(
        term in question
        for term in (
            "business overview",
            "business description",
            "products",
            "services",
            "customers",
            "strategy",
            "operations",
        )
    ):
        sections.append("1")

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
        sections.append("7A")

    if any(term in question for term in ("risk factor", "risk factors")):
        sections.append("1A")
    elif "risks" in question and "7A" not in sections:
        sections.append("1A")

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
        sections.append("7")

    if sections:
        return _unique(sections)

    if "business" in question:
        return ["1"]

    return None


def _normalize_intent(intent: str, question: str) -> str:
    question = question.lower()
    if any(term in question for term in ("table", "tables", "what revenue", "how much")):
        return "fact"
    return intent


def _infer_tickers_from_question(question: str, context: PlannerContext) -> list[str]:
    normalized_question = _normalize_match_text(question)
    matched = []

    for ticker in context.available_tickers:
        ticker_pattern = rf"\b{re.escape(ticker.lower())}\b"
        if re.search(ticker_pattern, question.lower()):
            matched.append(ticker)
            continue

        for company_name in (context.company_names_by_ticker or {}).get(ticker, ()):
            normalized_name = _normalize_company_name(company_name)
            if normalized_name and normalized_name in normalized_question:
                matched.append(ticker)
                break

    return _unique(matched)


def _infer_intent_from_question(question: str) -> str:
    lowered = question.lower()
    if any(term in lowered for term in ("over time", "trend", "changed", "evolved", "historical")):
        return "trend"
    if any(term in lowered for term in ("compare", "comparison", "versus", " vs ", "between")):
        return "comparison"
    return _normalize_intent("summary", question)


def _infer_time_scope_from_question(question: str, *, intent: str) -> str:
    lowered = question.lower()
    if _question_requests_specific_years(question):
        return "specific_years"
    if intent == "trend":
        return "all_available"
    if "previous" in lowered or "prior" in lowered or "year over year" in lowered:
        return "latest_and_previous"
    return "latest"


def _normalize_company_name(company_name: str) -> str:
    normalized = _normalize_match_text(company_name)
    suffixes = (
        " inc",
        " incorporated",
        " corp",
        " corporation",
        " co",
        " company",
        " ltd",
        " limited",
        " plc",
        " class a",
        " common stock",
    )
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)].strip()
                changed = True
    return normalized


def _normalize_match_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()


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
- Use "specific_years" when the user asks for a specific filing year, a specific
  range of filing years, or the oldest/earliest available filing.
- Populate "filing_years" only with years available for the selected ticker.
- If the user asks for the oldest or earliest available filing, use
  "specific_years" and the oldest available year for the selected ticker.
- If the user requests a period that is not available, use "specific_years" with
  an empty "filing_years" list. Never invent a missing filing period.
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
  "filing_years": [],
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
