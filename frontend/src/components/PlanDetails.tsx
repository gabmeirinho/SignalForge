import { ListChecks } from "lucide-react";
import type { SearchPlan } from "../types";

type PlanDetailsProps = {
  plan: SearchPlan | null | undefined;
};

export function PlanDetails({ plan }: PlanDetailsProps) {
  return (
    <section className="panel plan-panel">
      <div className="panel-heading">
        <ListChecks size={18} aria-hidden="true" />
        <h2>Retrieval Plan</h2>
      </div>
      {!plan ? (
        <p className="muted">Structured planning details will appear after a query.</p>
      ) : (
        <div className="plan-grid">
          <PlanField label="Tickers" value={plan.tickers.join(", ") || "-"} />
          <PlanField label="Sections" value={plan.sections.map((section) => `Item ${section}`).join(", ") || "-"} />
          <PlanField label="Time scope" value={plan.time_scope} />
          <PlanField label="Filing years" value={plan.filing_years.join(", ") || "-"} />
          <PlanField label="Intent" value={plan.intent} />
          <PlanField label="Top K" value={String(plan.top_k)} />
          <div className="plan-field wide">
            <span>Semantic queries</span>
            <ul>
              {plan.semantic_queries.map((query) => (
                <li key={query}>{query}</li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </section>
  );
}

function PlanField({ label, value }: { label: string; value: string }) {
  return (
    <div className="plan-field">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
