import { Send, Search } from "lucide-react";

const EXAMPLES = [
  "What are NVIDIA's latest risk factors?",
  "Compare AMD and NVDA business risks.",
  "Summarize Amazon's latest MD&A.",
  "How have Intel risk factors changed over the last two filings?",
];

type QueryComposerProps = {
  question: string;
  isLoading: boolean;
  onQuestionChange: (question: string) => void;
  onSubmit: () => void;
};

export function QueryComposer({
  question,
  isLoading,
  onQuestionChange,
  onSubmit,
}: QueryComposerProps) {
  const canSubmit = question.trim().length >= 3 && !isLoading;

  return (
    <section className="query-composer" aria-label="Query composer">
      <div className="composer-header">
        <div>
          <p className="eyebrow">SEC filing research</p>
          <h1>SignalForge</h1>
        </div>
        <Search className="header-icon" aria-hidden="true" />
      </div>
      <textarea
        value={question}
        onChange={(event) => onQuestionChange(event.target.value)}
        placeholder="Ask about indexed 10-K filings, risks, business sections, MD&A, or market risk."
        rows={4}
      />
      <div className="composer-actions">
        <div className="example-chips" aria-label="Example questions">
          {EXAMPLES.map((example) => (
            <button
              key={example}
              type="button"
              className="chip"
              onClick={() => onQuestionChange(example)}
              disabled={isLoading}
            >
              {example}
            </button>
          ))}
        </div>
        <button type="button" className="submit-button" disabled={!canSubmit} onClick={onSubmit}>
          <Send size={18} aria-hidden="true" />
          <span>{isLoading ? "Searching" : "Ask"}</span>
        </button>
      </div>
    </section>
  );
}
