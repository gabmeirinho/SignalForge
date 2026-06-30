import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import { fetchHealth, fetchIndex, submitQuery } from "../api";
import type { HealthResponse, IndexResponse, QueryResponse } from "../types";

vi.mock("../api", () => ({
  fetchHealth: vi.fn(),
  fetchIndex: vi.fn(),
  submitQuery: vi.fn(),
}));

const mockFetchHealth = vi.mocked(fetchHealth);
const mockFetchIndex = vi.mocked(fetchIndex);
const mockSubmitQuery = vi.mocked(submitQuery);

const health: HealthResponse = {
  status: "ok",
  database: true,
  qdrant_path: true,
};

const indexPayload: IndexResponse = {
  collection: "sec_chunks",
  embedding_model: "jinaai/jina-embeddings-v2-small-en",
  tickers: [
    {
      ticker: "NVDA",
      company_name: "NVIDIA CORP",
      filings: [
        {
          accession_number: "0001045810-26-000021",
          form_type: "10-K",
          filing_date: "2026-02-25",
          period_of_report: "2026-01-25",
          status: "ready",
          expected_point_count: 120,
          indexed_point_count: 120,
        },
      ],
      sections: [{ section_id: "1A", chunk_count: 42 }],
    },
    {
      ticker: "AMD",
      company_name: "ADVANCED MICRO DEVICES INC",
      filings: [
        {
          accession_number: "0000002488-26-000012",
          form_type: "10-K",
          filing_date: "2026-02-04",
          period_of_report: "2025-12-27",
          status: "ready",
          expected_point_count: 88,
          indexed_point_count: 88,
        },
      ],
      sections: [{ section_id: "7", chunk_count: 31 }],
    },
  ],
};

const queryPayload: QueryResponse = {
  question: "Compare AMD and NVDA business risks.",
  answer: "NVIDIA cites supply-chain risk [1]. AMD cites market risk [2].",
  warnings: ["llm answer generation unavailable"],
  used_fallback: true,
  planner_error: "DEEPSEEK_API_KEY is not set; used local rule-based planner",
  plan: {
    tickers: ["AMD", "NVDA"],
    sections: ["1A"],
    semantic_queries: ["Compare AMD and NVDA business risks."],
    time_scope: "latest",
    filing_years: [],
    intent: "comparison",
    top_k: 5,
  },
  sources: [
    {
      label: "[1] NVDA 2026 Item 1A chunk 3",
      score: 0.82,
      chunk_source: "sec_filing",
      ticker: "NVDA",
      company_name: "NVIDIA CORP",
      filing_date: "2026-02-25",
      published_at: null,
      section_id: "1A",
      section_title: "Risk Factors",
      chunk_index: 3,
      accession_number: "0001045810-26-000021",
      document_id: null,
      source_id: null,
      source_name: null,
      source_type: null,
      url: null,
      title: null,
      text: "Supply-chain risk text.",
    },
  ],
};

describe("App", () => {
  beforeEach(() => {
    mockFetchHealth.mockResolvedValue(health);
    mockFetchIndex.mockResolvedValue(indexPayload);
    mockSubmitQuery.mockResolvedValue(queryPayload);
  });

  it("renders indexed tickers and status after loading metadata", async () => {
    render(<App />);

    expect(await screen.findByText("NVDA")).toBeInTheDocument();
    expect(screen.getByText("AMD")).toBeInTheDocument();
    expect(screen.getByText("local index ready")).toBeInTheDocument();
    expect(screen.getByText("2 tickers")).toBeInTheDocument();
  });

  it("inserts a ticker into the query composer without submitting", async () => {
    const user = userEvent.setup();
    render(<App />);
    const sidebar = screen.getByRole("complementary", { name: /indexed filings/i });

    await within(sidebar).findByText("NVDA");
    await user.click(within(sidebar).getByRole("button", { name: /NVDANVIDIA CORP/i }));

    expect(screen.getByRole("textbox")).toHaveValue("Summarize NVDA's latest risk factors.");
    expect(mockSubmitQuery).not.toHaveBeenCalled();
  });

  it("submits a query and renders answer, warnings, plan, and source metadata", async () => {
    const user = userEvent.setup();
    render(<App />);

    await screen.findByText("NVDA");
    await user.click(screen.getByRole("button", { name: "Compare AMD and NVDA business risks." }));
    await user.click(screen.getByRole("button", { name: /Ask/i }));

    await waitFor(() => {
      expect(mockSubmitQuery).toHaveBeenCalledWith("Compare AMD and NVDA business risks.", {
        includePlan: true,
        includeSourceText: true,
      });
    });
    expect(await screen.findByText(/NVIDIA cites supply-chain risk/)).toBeInTheDocument();
    expect(screen.getByText("planner fallback")).toBeInTheDocument();
    expect(screen.getByText("llm answer generation unavailable")).toBeInTheDocument();
    expect(screen.getByText("comparison")).toBeInTheDocument();
    expect(screen.getByText("[1] NVDA 2026 Item 1A chunk 3")).toBeInTheDocument();
  });

  it("expands source text on demand", async () => {
    const user = userEvent.setup();
    render(<App />);

    await screen.findByText("NVDA");
    await user.click(screen.getByRole("button", { name: "Compare AMD and NVDA business risks." }));
    await user.click(screen.getByRole("button", { name: /Ask/i }));
    const sourceButton = await screen.findByRole("button", {
      name: /\[1\] NVDA 2026 Item 1A chunk 3/i,
    });

    expect(screen.queryByText("Supply-chain risk text.")).not.toBeInTheDocument();
    await user.click(sourceButton);

    const sourcesPanel = screen.getByRole("region", { name: /sources/i });
    expect(within(sourcesPanel).getByText("Supply-chain risk text.")).toBeInTheDocument();
  });

  it("renders API errors from failed queries", async () => {
    const user = userEvent.setup();
    mockSubmitQuery.mockRejectedValue(new Error("SignalForge database not found"));
    render(<App />);

    await screen.findByText("NVDA");
    await user.type(screen.getByRole("textbox"), "What are NVIDIA risks?");
    await user.click(screen.getByRole("button", { name: /Ask/i }));

    expect(await screen.findByText("Query failed")).toBeInTheDocument();
    expect(screen.getByText("SignalForge database not found")).toBeInTheDocument();
  });
});
