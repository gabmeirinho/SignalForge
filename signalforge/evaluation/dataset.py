"""Load and validate reproducible evaluation cases.

The legacy planner fixture is accepted for replay, but new datasets use the
versioned envelope and explicit evidence and answer expectations.
"""

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "evaluation/datasets/benchmark-v1.json"
QueryClass = Literal["lookup", "synthesis", "comparison", "trend", "missing_evidence", "conflicting_source"]


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1)
    passage: str = Field(min_length=1)
    kind: Literal["passage", "absence"] = "passage"

    @field_validator("label", "passage")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class CaseContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    available_tickers: list[str]
    available_sections: list[Literal["1", "1A", "7", "7A"]]
    filing_years_by_ticker: dict[str, list[int]]


class ExpectedPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tickers: list[str]
    sections: list[Literal["1", "1A", "7", "7A"]]
    time_scope: Literal["latest", "latest_and_previous", "all_available", "specific_years"]
    filing_years: list[int] = Field(default_factory=list)
    intent: Literal["fact", "summary", "comparison", "trend"]
    top_k: int | None = Field(default=None, ge=1, le=20)


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    question: str = Field(min_length=1)
    query_class: QueryClass
    context: CaseContext
    expected_plan: ExpectedPlan
    evidence: list[Evidence] = Field(min_length=1)
    answer_facts: list[str]
    rubric_tags: list[str] = Field(min_length=1)

    @field_validator("question")
    @classmethod
    def nonblank_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value

    @model_validator(mode="after")
    def validate_case(self):
        if not set(self.expected_plan.tickers) <= set(self.context.available_tickers):
            raise ValueError("expected tickers must be available in planner context")
        if not set(self.expected_plan.sections) <= set(self.context.available_sections):
            raise ValueError("expected sections must be available in planner context")
        if len({item.label for item in self.evidence}) != len(self.evidence):
            raise ValueError("duplicate evidence label within case")
        if len(set(self.rubric_tags)) != len(self.rubric_tags):
            raise ValueError("duplicate rubric tag")
        if any(not fact.strip() for fact in self.answer_facts):
            raise ValueError("answer facts must not be blank")
        if self.query_class == "missing_evidence":
            if any(item.kind != "absence" for item in self.evidence):
                raise ValueError("missing-evidence cases must use absence evidence")
        elif any(item.kind != "passage" for item in self.evidence):
            raise ValueError("only missing-evidence cases may use absence evidence")
        elif not self.answer_facts:
            raise ValueError("answer facts are required when evidence exists")
        return self

    def to_evaluator_case(self) -> dict:
        """Return the historical evaluator shape while retaining the new fields."""
        data = self.model_dump(exclude_unset=True)
        data["expected"] = data.pop("expected_plan")
        return data


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


def load_dataset(path: str | Path = DEFAULT_DATASET) -> LoadedDataset:
    """Validate a v1 dataset or adapt the original planner-only fixture."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, list):
        cases = [LegacyPlannerCase.model_validate(case).model_dump(exclude_unset=True) for case in raw]
        ids = [case["id"] for case in cases]
        if not cases or len(set(ids)) != len(ids):
            raise ValueError("legacy dataset is empty or contains duplicate case ids")
        version = "planner-legacy-v0"
    else:
        dataset = EvaluationDataset.model_validate(raw)
        cases = [case.to_evaluator_case() for case in dataset.cases]
        version = dataset.version
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return LoadedDataset(
        version=version,
        content_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        cases=cases,
    )
