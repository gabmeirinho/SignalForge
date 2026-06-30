export type HealthResponse = {
  status: string;
  database: boolean;
  qdrant_path: boolean;
};

export type IndexFiling = {
  accession_number: string;
  form_type: string;
  filing_date: string | null;
  period_of_report: string | null;
  status: string;
  expected_point_count: number;
  indexed_point_count: number;
};

export type IndexSection = {
  section_id: string;
  chunk_count: number;
};

export type IndexTicker = {
  ticker: string;
  company_name: string | null;
  filings: IndexFiling[];
  sections: IndexSection[];
};

export type IndexResponse = {
  tickers: IndexTicker[];
  embedding_model: string;
  collection: string;
};

export type SearchPlan = {
  tickers: string[];
  sections: string[];
  semantic_queries: string[];
  time_scope: string;
  filing_years: number[];
  intent: string;
  top_k: number;
};

export type SourceChunk = {
  label: string;
  score: number;
  chunk_source: string;
  ticker: string | null;
  company_name: string | null;
  filing_date: string | null;
  published_at: string | null;
  section_id: string | null;
  section_title: string | null;
  chunk_index: number | null;
  accession_number: string | null;
  document_id: number | null;
  source_id: number | null;
  source_name: string | null;
  source_type: string | null;
  url: string | null;
  title: string | null;
  text: string | null;
};

export type QueryResponse = {
  question: string;
  answer: string;
  warnings: string[];
  used_fallback: boolean;
  planner_error: string | null;
  plan: SearchPlan | null;
  sources: SourceChunk[];
};
