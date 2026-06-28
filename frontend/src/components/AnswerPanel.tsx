import { AlertTriangle, FileText, Loader2 } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { QueryResponse } from "../types";

type AnswerPanelProps = {
  response: QueryResponse | null;
  error: string | null;
  isLoading: boolean;
};

export function AnswerPanel({ response, error, isLoading }: AnswerPanelProps) {
  if (isLoading) {
    return (
      <section className="panel answer-panel">
        <div className="state-block">
          <Loader2 className="spin" aria-hidden="true" />
          <h2>Searching filings</h2>
          <p>Planning the query, retrieving local chunks, and preparing cited evidence.</p>
        </div>
      </section>
    );
  }

  if (error) {
    return (
      <section className="panel answer-panel error-panel">
        <div className="state-block">
          <AlertTriangle aria-hidden="true" />
          <h2>Query failed</h2>
          <p>{error}</p>
        </div>
      </section>
    );
  }

  if (!response) {
    return (
      <section className="panel answer-panel">
        <div className="state-block">
          <FileText aria-hidden="true" />
          <h2>Ask a filing question</h2>
          <p>Try a company comparison, a latest-filing summary, or a trend across filing years.</p>
        </div>
      </section>
    );
  }

  const badges = [
    response.used_fallback ? "planner fallback" : null,
    ...response.warnings,
  ].filter(Boolean);

  return (
    <section className="panel answer-panel">
      <div className="panel-title-row">
        <div>
          <p className="eyebrow">Answer</p>
          <h2>{response.question}</h2>
        </div>
      </div>
      {badges.length > 0 && (
        <div className="warning-row">
          {badges.map((badge) => (
            <span className="warning-badge" key={badge}>
              <AlertTriangle size={14} aria-hidden="true" />
              {badge}
            </span>
          ))}
        </div>
      )}
      <div className="answer-text">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{response.answer}</ReactMarkdown>
      </div>
      {response.planner_error && <p className="planner-error">{response.planner_error}</p>}
    </section>
  );
}
