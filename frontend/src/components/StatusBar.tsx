import { CircleCheck, CircleX, Server } from "lucide-react";
import type { HealthResponse } from "../types";

type StatusBarProps = {
  health: HealthResponse | null;
  indexCount: number;
};

export function StatusBar({ health, indexCount }: StatusBarProps) {
  const ready = Boolean(health?.database && health?.qdrant_path);

  return (
    <footer className="status-bar">
      <div>
        <Server size={16} aria-hidden="true" />
        <span>FastAPI</span>
      </div>
      <div className={ready ? "status-ok" : "status-bad"}>
        {ready ? <CircleCheck size={16} aria-hidden="true" /> : <CircleX size={16} aria-hidden="true" />}
        <span>{ready ? "local index ready" : "index unavailable"}</span>
      </div>
      <div>
        <span>{indexCount} tickers</span>
      </div>
    </footer>
  );
}
