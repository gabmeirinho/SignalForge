import json
import os
from dataclasses import dataclass

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed

from query_planner import (
    ChatCompletionsClient,
    DEFAULT_DEEPSEEK_BASE_URL,
    SearchPlan,
)
from vector_store import SearchResult


DEFAULT_ANSWER_MODEL = "deepseek-v4-flash"
ANSWER_TEMPERATURE = 0.0


@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str
    evidence_labels: list[str]
    warnings: list[str]


class AnswerGenerator:
    def __init__(
        self,
        *,
        client: ChatCompletionsClient,
        model: str = DEFAULT_ANSWER_MODEL,
    ) -> None:
        self.client = client
        self.model = model

    @classmethod
    def from_environment(
        cls,
        *,
        model: str = DEFAULT_ANSWER_MODEL,
    ) -> "AnswerGenerator":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")

        client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL),
        )
        return cls(client=client, model=model)

    def generate(
        self,
        *,
        question: str,
        plan: SearchPlan,
        chunks: list[SearchResult],
        available_years_by_ticker: dict[str, tuple[int, ...]] | None = None,
    ) -> GeneratedAnswer:
        if not chunks:
            return no_evidence_answer(
                question=question,
                plan=plan,
                available_years_by_ticker=available_years_by_ticker or {},
            )

        evidence = format_evidence(chunks)
        answer = self._request_answer(question=question, plan=plan, evidence=evidence)
        return GeneratedAnswer(
            answer=answer,
            evidence_labels=[block.label for block in evidence],
            warnings=[],
        )

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(0.5), reraise=True)
    def _request_answer(
        self,
        *,
        question: str,
        plan: SearchPlan,
        evidence: list["EvidenceBlock"],
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _system_prompt()},
                {
                    "role": "user",
                    "content": _user_prompt(question=question, plan=plan, evidence=evidence),
                },
            ],
            max_tokens=1_200,
            temperature=ANSWER_TEMPERATURE,
            extra_body={"thinking": {"type": "disabled"}},
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Answer model returned an empty response")
        return content.strip()


@dataclass(frozen=True)
class EvidenceBlock:
    label: str
    metadata: dict
    text: str


def format_evidence(chunks: list[SearchResult]) -> list[EvidenceBlock]:
    evidence = []
    for index, result in enumerate(chunks, start=1):
        payload = result.payload
        filing_date = payload.get("filing_date") or ""
        filing_year = filing_date[:4] if filing_date[:4].isdigit() else "unknown"
        label = (
            f"[{index}] {payload.get('ticker')} {filing_year} "
            f"Item {payload.get('section_id')} chunk {payload.get('chunk_index')}"
        )
        evidence.append(
            EvidenceBlock(
                label=label,
                metadata={
                    "score": result.score,
                    "ticker": payload.get("ticker"),
                    "filing_date": payload.get("filing_date"),
                    "section_id": payload.get("section_id"),
                    "section_title": payload.get("section_title"),
                    "chunk_index": payload.get("chunk_index"),
                    "accession_number": payload.get("accession_number"),
                },
                text=payload.get("text", ""),
            )
        )
    return evidence


def no_evidence_answer(
    *,
    question: str,
    plan: SearchPlan,
    available_years_by_ticker: dict[str, tuple[int, ...]],
) -> GeneratedAnswer:
    warnings = ["no retrieved evidence"]

    if plan.time_scope == "specific_years" and not plan.filing_years:
        answer = _unavailable_specific_year_answer(
            plan=plan,
            available_years_by_ticker=available_years_by_ticker,
        )
    else:
        answer = (
            "I could not answer from the local SEC filing index because retrieval returned "
            "no matching chunks for the requested scope."
        )

    return GeneratedAnswer(answer=answer, evidence_labels=[], warnings=warnings)


def _unavailable_specific_year_answer(
    *,
    plan: SearchPlan,
    available_years_by_ticker: dict[str, tuple[int, ...]],
) -> str:
    if not plan.tickers:
        return "The requested filing year is not available in the local filing index."

    parts = []
    for ticker in plan.tickers:
        years = available_years_by_ticker.get(ticker, ())
        if years:
            parts.append(f"{ticker}: {', '.join(str(year) for year in years)}")
        else:
            parts.append(f"{ticker}: no available filing years")

    return (
        "The requested filing year is not available in the local filing index. "
        f"Available filing years are {', '.join(parts)}."
    )


def _system_prompt() -> str:
    return """
You are a financial research assistant answering questions from local SEC filing chunks.

Rules:
- Use only the retrieved evidence supplied by the user.
- Cite every material claim with evidence labels such as [1] or [2].
- If the evidence is incomplete, say what is missing.
- If multiple companies are requested, cover each company explicitly.
- If multiple filing years are requested, cover each year explicitly.
- Do not use outside knowledge.
- Be concise and factual.
""".strip()


def _user_prompt(
    *,
    question: str,
    plan: SearchPlan,
    evidence: list[EvidenceBlock],
) -> str:
    return json.dumps(
        {
            "question": question,
            "retrieval_plan": plan.model_dump(),
            "evidence": [
                {
                    "label": block.label,
                    "metadata": block.metadata,
                    "text": block.text,
                }
                for block in evidence
            ],
        },
        ensure_ascii=False,
    )
