import { Database, PlusCircle } from "lucide-react";
import type { IndexResponse, IndexTicker } from "../types";

type IndexSidebarProps = {
  index: IndexResponse | null;
  isLoading: boolean;
  error: string | null;
  onTickerClick: (ticker: string) => void;
};

export function IndexSidebar({ index, isLoading, error, onTickerClick }: IndexSidebarProps) {
  return (
    <aside className="index-sidebar" aria-label="Indexed filings">
      <div className="sidebar-header">
        <Database size={19} aria-hidden="true" />
        <div>
          <h2>Indexed Coverage</h2>
          <p>{index ? `${index.collection} · ${index.embedding_model}` : "Local SEC index"}</p>
        </div>
      </div>
      {isLoading && <p className="muted">Loading indexed companies...</p>}
      {error && <p className="sidebar-error">{error}</p>}
      {index && (
        <div className="ticker-list">
          {index.tickers.map((ticker) => (
            <TickerCard key={ticker.ticker} ticker={ticker} onTickerClick={onTickerClick} />
          ))}
        </div>
      )}
    </aside>
  );
}

function TickerCard({
  ticker,
  onTickerClick,
}: {
  ticker: IndexTicker;
  onTickerClick: (ticker: string) => void;
}) {
  const years = uniqueYears(ticker.filings);
  const totals = ticker.filings.reduce(
    (accumulator, filing) => {
      accumulator.expected += filing.expected_point_count;
      accumulator.indexed += filing.indexed_point_count;
      return accumulator;
    },
    { expected: 0, indexed: 0 },
  );

  return (
    <article className="ticker-card">
      <button type="button" className="ticker-button" onClick={() => onTickerClick(ticker.ticker)}>
        <span>
          <strong>{ticker.ticker}</strong>
          <small>{ticker.company_name ?? "Company name unavailable"}</small>
        </span>
        <PlusCircle size={17} aria-hidden="true" />
      </button>
      <div className="coverage-row">
        <span>{ticker.filings.length} filings</span>
        <span>{years.join(", ") || "No dates"}</span>
      </div>
      <div className="progress-line" aria-label={`${totals.indexed} of ${totals.expected} points indexed`}>
        <span style={{ width: `${progressPercent(totals.indexed, totals.expected)}%` }} />
      </div>
      <div className="coverage-row">
        <span>{totals.indexed}/{totals.expected} points</span>
        <span>{ticker.filings.every((filing) => filing.status === "ready") ? "ready" : "mixed"}</span>
      </div>
      <div className="section-pills">
        {ticker.sections.map((section) => (
          <span key={section.section_id}>
            Item {section.section_id}: {section.chunk_count}
          </span>
        ))}
      </div>
    </article>
  );
}

function uniqueYears(filings: IndexTicker["filings"]): string[] {
  return Array.from(
    new Set(
      filings
        .map((filing) => filing.filing_date?.slice(0, 4))
        .filter((year): year is string => Boolean(year)),
    ),
  );
}

function progressPercent(indexed: number, expected: number): number {
  if (expected <= 0) {
    return 0;
  }
  return Math.max(0, Math.min(100, Math.round((indexed / expected) * 100)));
}
