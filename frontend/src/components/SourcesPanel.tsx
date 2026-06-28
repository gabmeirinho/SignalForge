import { ChevronDown, Files } from "lucide-react";
import { useState } from "react";
import type { SourceChunk } from "../types";

type SourcesPanelProps = {
  sources: SourceChunk[];
};

export function SourcesPanel({ sources }: SourcesPanelProps) {
  const [openLabels, setOpenLabels] = useState<Set<string>>(new Set());

  function toggle(label: string) {
    setOpenLabels((current) => {
      const next = new Set(current);
      if (next.has(label)) {
        next.delete(label);
      } else {
        next.add(label);
      }
      return next;
    });
  }

  return (
    <section className="panel sources-panel">
      <div className="panel-heading">
        <Files size={18} aria-hidden="true" />
        <h2>Sources</h2>
        <span>{sources.length}</span>
      </div>
      {sources.length === 0 ? (
        <p className="muted">Retrieved chunks will appear here after a query.</p>
      ) : (
        <div className="source-list">
          {sources.map((source) => {
            const isOpen = openLabels.has(source.label);
            const filingYear = source.filing_date?.slice(0, 4) ?? "unknown";
            return (
              <article className="source-card" key={source.label}>
                <button type="button" className="source-toggle" onClick={() => toggle(source.label)}>
                  <span>
                    <strong>{source.label}</strong>
                    <small>
                      {source.ticker ?? "Unknown"} {filingYear} · Item {source.section_id ?? "-"}
                    </small>
                  </span>
                  <ChevronDown className={isOpen ? "rotated" : ""} size={18} aria-hidden="true" />
                </button>
                <dl className="source-meta">
                  <div>
                    <dt>Section</dt>
                    <dd>{source.section_title ?? "Unknown"}</dd>
                  </div>
                  <div>
                    <dt>Score</dt>
                    <dd>{source.score.toFixed(3)}</dd>
                  </div>
                  <div>
                    <dt>Accession</dt>
                    <dd>{source.accession_number ?? "-"}</dd>
                  </div>
                </dl>
                {isOpen && <p className="source-text">{source.text || "Source text not included."}</p>}
              </article>
            );
          })}
        </div>
      )}
    </section>
  );
}
