import { useEffect, useState } from "react";
import { fetchHealth, fetchIndex, submitQuery } from "./api";
import { AnswerPanel } from "./components/AnswerPanel";
import { IndexSidebar } from "./components/IndexSidebar";
import { PlanDetails } from "./components/PlanDetails";
import { QueryComposer } from "./components/QueryComposer";
import { SourcesPanel } from "./components/SourcesPanel";
import { StatusBar } from "./components/StatusBar";
import type { HealthResponse, IndexResponse, QueryResponse } from "./types";

export default function App() {
  const [question, setQuestion] = useState("");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [index, setIndex] = useState<IndexResponse | null>(null);
  const [indexError, setIndexError] = useState<string | null>(null);
  const [isIndexLoading, setIsIndexLoading] = useState(true);
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [queryError, setQueryError] = useState<string | null>(null);
  const [isQueryLoading, setIsQueryLoading] = useState(false);

  useEffect(() => {
    let isMounted = true;

    async function loadInitialData() {
      setIsIndexLoading(true);
      try {
        const [healthPayload, indexPayload] = await Promise.all([fetchHealth(), fetchIndex()]);
        if (!isMounted) {
          return;
        }
        setHealth(healthPayload);
        setIndex(indexPayload);
        setIndexError(null);
      } catch (error) {
        if (!isMounted) {
          return;
        }
        setIndexError(error instanceof Error ? error.message : "Unable to load index metadata");
      } finally {
        if (isMounted) {
          setIsIndexLoading(false);
        }
      }
    }

    loadInitialData();
    return () => {
      isMounted = false;
    };
  }, []);

  async function handleSubmit() {
    const trimmed = question.trim();
    if (trimmed.length < 3 || isQueryLoading) {
      return;
    }

    setIsQueryLoading(true);
    setQueryError(null);
    try {
      const payload = await submitQuery(trimmed, {
        includePlan: true,
        includeSourceText: true,
      });
      setResponse(payload);
    } catch (error) {
      setQueryError(error instanceof Error ? error.message : "Unable to run query");
    } finally {
      setIsQueryLoading(false);
    }
  }

  function handleTickerClick(ticker: string) {
    setQuestion((current) => {
      const trimmed = current.trim();
      if (!trimmed) {
        return `Summarize ${ticker}'s latest risk factors.`;
      }
      if (new RegExp(`\\b${ticker}\\b`, "i").test(trimmed)) {
        return current;
      }
      return `${trimmed} ${ticker}`;
    });
  }

  return (
    <div className="app-shell">
      <IndexSidebar
        index={index}
        isLoading={isIndexLoading}
        error={indexError}
        onTickerClick={handleTickerClick}
      />
      <main className="workspace">
        <QueryComposer
          question={question}
          isLoading={isQueryLoading}
          onQuestionChange={setQuestion}
          onSubmit={handleSubmit}
        />
        <div className="content-grid">
          <AnswerPanel response={response} error={queryError} isLoading={isQueryLoading} />
          <div className="side-panels">
            <PlanDetails plan={response?.plan} />
            <SourcesPanel sources={response?.sources ?? []} />
          </div>
        </div>
      </main>
      <StatusBar health={health} indexCount={index?.tickers.length ?? 0} />
    </div>
  );
}
