import type { HealthResponse, IndexResponse, QueryResponse } from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

type QueryOptions = {
  includePlan?: boolean;
  includeSourceText?: boolean;
};

export async function fetchHealth(): Promise<HealthResponse> {
  return requestJson<HealthResponse>("/health");
}

export async function fetchIndex(): Promise<IndexResponse> {
  return requestJson<IndexResponse>("/api/index");
}

export async function submitQuery(
  question: string,
  options: QueryOptions = {},
): Promise<QueryResponse> {
  return requestJson<QueryResponse>("/api/query", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      question,
      include_plan: options.includePlan ?? true,
      include_source_text: options.includeSourceText ?? true,
    }),
  });
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, init);
  const payload = await response.json().catch(() => null);

  if (!response.ok) {
    const detail = payload?.detail;
    const message = typeof detail === "string" ? detail : `Request failed with ${response.status}`;
    throw new Error(message);
  }

  return payload as T;
}
