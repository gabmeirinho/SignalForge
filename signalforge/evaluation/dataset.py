"""Load and validate reproducible evaluation cases.

Three dataset shapes are accepted:

* the legacy planner-only fixture (a bare JSON list), replayed for planner checks;
* version 1.0, where each case carries its own evidence passages;
* version 2.0, where a shared ``corpus`` of passages carries the metadata that
  retrieval filters on, and each case grades which corpus passages are relevant.

Version 2.0 exists so retrieval can be scored against labels: passages from one
case act as realistic distractors for the others, and every answerable case must
have at least one in-scope passage that is *not* relevant, so ranking is tested
rather than filtering alone.
"""

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "evaluation/datasets/benchmark-v1.json"
QueryClass = Literal["lookup", "synthesis", "comparison", "trend", "missing_evidence", "conflicting_source"]
SectionId = Literal["1", "1A", "7", "7A"]
TimeScope = Literal["latest", "latest_and_previous", "all_available", "specific_years"]

# Relevance grades for version 2.0 cases. Corpus passages a case does not list are
# treated as grade 0 (not relevant).
REQUIRED = 2  # evidence the answer needs; recall is measured on these
RELATED = 1  # on-topic and acceptable to retrieve, but not required
# Each answerable case needs this many in-scope, unjudged passages. With fewer,
# retrieval returns the whole scope and recall@k is trivially perfect.
MIN_IN_SCOPE_DISTRACTORS = 3


def _nonblank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1)
    passage: str = Field(min_length=1)
    kind: Literal["passage", "absence"] = "passage"

    @field_validator("label", "passage")
    @classmethod
    def nonblank(cls, value: str) -> str:
        return _nonblank(value)


class CaseContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    available_tickers: list[str]
    available_sections: list[SectionId]
    filing_years_by_ticker: dict[str, list[int]]


class ExpectedPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tickers: list[str]
    sections: list[SectionId]
    time_scope: TimeScope
    filing_years: list[int] = Field(default_factory=list)
    intent: Literal["fact", "summary", "comparison", "trend"]
    top_k: int | None = Field(default=None, ge=1, le=20)


class _CaseBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    question: str = Field(min_length=1)
    query_class: QueryClass
    context: CaseContext
    expected_plan: ExpectedPlan
    answer_facts: list[str]
    rubric_tags: list[str] = Field(min_length=1)

    @field_validator("question")
    @classmethod
    def nonblank_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value

    def _validate_common(self) -> None:
        if not set(self.expected_plan.tickers) <= set(self.context.available_tickers):
            raise ValueError("expected tickers must be available in planner context")
        if not set(self.expected_plan.sections) <= set(self.context.available_sections):
            raise ValueError("expected sections must be available in planner context")
        if len(set(self.rubric_tags)) != len(self.rubric_tags):
            raise ValueError("duplicate rubric tag")
        if any(not fact.strip() for fact in self.answer_facts):
            raise ValueError("answer facts must not be blank")

    def to_evaluator_case(self) -> dict:
        """Return the historical evaluator shape while retaining the new fields."""
        data = self.model_dump(exclude_unset=True)
        data["expected"] = data.pop("expected_plan")
        return data


class EvaluationCase(_CaseBase):
    evidence: list[Evidence] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_case(self):
        self._validate_common()
        if len({item.label for item in self.evidence}) != len(self.evidence):
            raise ValueError("duplicate evidence label within case")
        if self.query_class == "missing_evidence":
            if any(item.kind != "absence" for item in self.evidence):
                raise ValueError("missing-evidence cases must use absence evidence")
        elif any(item.kind != "passage" for item in self.evidence):
            raise ValueError("only missing-evidence cases may use absence evidence")
        elif not self.answer_facts:
            raise ValueError("answer facts are required when evidence exists")
        return self


class EvaluationDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal["1.0"]
    cases: list[EvaluationCase] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_identifiers(self):
        ids = [case.id for case in self.cases]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate case id")
        labels = [e.label for case in self.cases for e in case.evidence]
        if len(set(labels)) != len(labels):
            raise ValueError("duplicate evidence label")
        return self


class CorpusPassage(BaseModel):
    """One retrievable passage, with the metadata retrieval filters on."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(pattern=r"^fx:[a-z0-9][a-z0-9-]*$")
    ticker: str = Field(pattern=r"^[A-Z]{1,5}$")
    section_id: SectionId
    filing_year: int = Field(ge=1990, le=2100)
    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def nonblank_text(cls, value: str) -> str:
        return _nonblank(value)


class RelevanceJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1)
    grade: Literal[1, 2]


class EvaluationCaseV2(_CaseBase):
    relevant: list[RelevanceJudgment]

    @model_validator(mode="after")
    def validate_case(self):
        self._validate_common()
        labels = [item.label for item in self.relevant]
        if len(set(labels)) != len(labels):
            raise ValueError("duplicate relevance label within case")
        if self.query_class == "missing_evidence":
            if self.relevant:
                raise ValueError("missing-evidence cases must not list relevant passages")
        else:
            if not any(item.grade == REQUIRED for item in self.relevant):
                raise ValueError("answerable cases need at least one required (grade 2) passage")
            if not self.answer_facts:
                raise ValueError("answer facts are required when evidence exists")
        return self

    def scope_years(self, ticker: str) -> set[int]:
        """Filing years in retrieval scope for ``ticker``; mirrors ``years_for_plan_scope``."""
        plan = self.expected_plan
        if plan.time_scope == "specific_years":
            return set(plan.filing_years)
        years = sorted(self.context.filing_years_by_ticker.get(ticker, ()), reverse=True)
        if plan.time_scope == "latest":
            return set(years[:1])
        if plan.time_scope == "latest_and_previous":
            return set(years[:2])
        return set(years)

    def in_scope(self, passage: CorpusPassage) -> bool:
        """Whether retrieval with the expected plan's filters can return ``passage``."""
        plan = self.expected_plan
        if passage.ticker not in plan.tickers:
            return False
        if plan.sections and passage.section_id not in plan.sections:
            return False
        return passage.filing_year in self.scope_years(passage.ticker)


class EvaluationDatasetV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal["2.0"]
    corpus: list[CorpusPassage] = Field(min_length=1)
    cases: list[EvaluationCaseV2] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_dataset(self):
        ids = [case.id for case in self.cases]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate case id")
        corpus = {}
        for passage in self.corpus:
            if passage.label in corpus:
                raise ValueError(f"duplicate corpus label: {passage.label}")
            corpus[passage.label] = passage

        known_filings = {
            (ticker, year)
            for case in self.cases
            for ticker, years in case.context.filing_years_by_ticker.items()
            for year in years
        }
        for passage in self.corpus:
            if (passage.ticker, passage.filing_year) not in known_filings:
                raise ValueError(
                    f"{passage.label}: {passage.ticker} {passage.filing_year} "
                    "is not an available filing in any case context"
                )

        for case in self.cases:
            _validate_case_against_corpus(case, corpus)
        return self


def _validate_case_against_corpus(case: EvaluationCaseV2, corpus: dict[str, CorpusPassage]) -> None:
    for judgment in case.relevant:
        passage = corpus.get(judgment.label)
        if passage is None:
            raise ValueError(f"{case.id}: unknown corpus label {judgment.label}")
        if not case.in_scope(passage):
            raise ValueError(
                f"{case.id}: {judgment.label} is outside the expected plan's retrieval scope"
            )

    if case.query_class == "missing_evidence":
        return

    required = [corpus[item.label] for item in case.relevant if item.grade == REQUIRED]
    plan = case.expected_plan
    for ticker in plan.tickers:
        if not any(passage.ticker == ticker for passage in required):
            raise ValueError(f"{case.id}: no required passage for ticker {ticker}")
    for section in plan.sections:
        if not any(passage.section_id == section for passage in required):
            raise ValueError(f"{case.id}: no required passage for section {section}")
    if plan.intent == "trend" or plan.time_scope == "latest_and_previous":
        for ticker in plan.tickers:
            covered = {p.filing_year for p in required if p.ticker == ticker}
            expected = case.scope_years(ticker)
            if plan.time_scope == "all_available":
                if len(covered) < 2:
                    raise ValueError(f"{case.id}: trend needs required passages from 2+ years")
            elif covered != expected:
                raise ValueError(
                    f"{case.id}: required passages cover years {sorted(covered)}, "
                    f"expected {sorted(expected)}"
                )

    judged = {item.label for item in case.relevant}
    distractors = sum(case.in_scope(p) and p.label not in judged for p in corpus.values())
    if distractors < MIN_IN_SCOPE_DISTRACTORS:
        raise ValueError(
            f"{case.id}: has {distractors} in-scope distractors, needs "
            f"{MIN_IN_SCOPE_DISTRACTORS} so ranking is tested"
        )


class LegacyPlannerCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    question: str = Field(min_length=1)
    context: CaseContext
    expected: ExpectedPlan


class LoadedDataset(BaseModel):
    version: str
    content_hash: str
    cases: list[dict]
    corpus: list[dict] = Field(default_factory=list)


def load_dataset(path: str | Path = DEFAULT_DATASET) -> LoadedDataset:
    """Validate a v1 or v2 dataset, or adapt the original planner-only fixture."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    corpus: list[dict] = []
    if isinstance(raw, list):
        cases = [LegacyPlannerCase.model_validate(case).model_dump(exclude_unset=True) for case in raw]
        ids = [case["id"] for case in cases]
        if not cases or len(set(ids)) != len(ids):
            raise ValueError("legacy dataset is empty or contains duplicate case ids")
        version = "planner-legacy-v0"
    elif isinstance(raw, dict) and raw.get("version") == "2.0":
        dataset = EvaluationDatasetV2.model_validate(raw)
        cases = [case.to_evaluator_case() for case in dataset.cases]
        corpus = [passage.model_dump() for passage in dataset.corpus]
        version = dataset.version
    else:
        dataset = EvaluationDataset.model_validate(raw)
        cases = [case.to_evaluator_case() for case in dataset.cases]
        version = dataset.version
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return LoadedDataset(
        version=version,
        content_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        cases=cases,
        corpus=corpus,
    )
