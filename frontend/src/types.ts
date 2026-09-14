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

export type IndexSource = {
  id: number;
  ticker: string | null;
  company_name: string | null;
  name: string;
  url: string;
  source_type: string;
  ownership: string;
  trust_level: string;
  discovery_status: string;
  enabled: boolean;
  confidence_score: number | null;
  document_count: number;
  last_ingestion_status: string | null;
  last_ingestion_completed_at: string | null;
};

export type IndexSummary = {
  indexed_filing_count: number;
  approved_source_count: number;
  candidate_source_count: number;
  document_count: number;
};

export type IndexResponse = {
  tickers: IndexTicker[];
  sources: IndexSource[];
  summary: IndexSummary;
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
  query_run_id: number;
  research_session_id: number;
  question: string;
  answer: string;
  warnings: string[];
  used_fallback: boolean;
  planner_error: string | null;
  plan: SearchPlan | null;
  sources: SourceChunk[];
};

export type QueryRunSummary = {
  id: number;
  research_session_id: number | null;
  question: string;
  status: string;
  answer_preview: string | null;
  started_at: string;
  completed_at: string | null;
};

export type RetrievalMetadataSource = Omit<SourceChunk, "text">;

export type RetrievalMetadata = {
  sources?: RetrievalMetadataSource[];
  warnings?: string[];
  used_fallback?: boolean;
  planner_error?: string | null;
  [key: string]: unknown;
};

export type QueryRunDetail = {
  id: number;
  research_session_id: number | null;
  question: string;
  status: string;
  planner_model: string | null;
  answer_model: string | null;
  embedding_model: string | null;
  vector_collection: string | null;
  planned_query: Partial<SearchPlan> & Record<string, unknown>;
  retrieval_metadata: RetrievalMetadata;
  answer_text: string | null;
  error_message: string | null;
  started_at: string;
  completed_at: string | null;
  created_at: string;
  updated_at: string;
};
