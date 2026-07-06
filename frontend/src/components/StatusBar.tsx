import { CircleCheck, CircleX, Database, HardDrive, LoaderCircle, Server } from "lucide-react";
import type { HealthResponse } from "../types";

type StatusBarProps = {
  health: HealthResponse | null;
  healthError: string | null;
  isHealthLoading: boolean;
  lastCheckedAt: Date | null;
  indexCount: number;
};

export function StatusBar({
  health,
  healthError,
  isHealthLoading,
  lastCheckedAt,
  indexCount,
}: StatusBarProps) {
  const currentHealth = healthError ? null : health;
  const apiStatus = getApiStatus(health, healthError, isHealthLoading);
  const databaseStatus = getTargetStatus(currentHealth?.database, "Database");
  const vectorStatus = getTargetStatus(currentHealth?.qdrant_path, "Vector store");
  const checkedTime = lastCheckedAt?.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });

  return (
    <footer className="status-bar">
      <div className={apiStatus.className} aria-live="polite">
        <StatusIcon status={apiStatus.status} />
        <span>{apiStatus.label}</span>
      </div>
      <div className={databaseStatus.className}>
        <Database size={16} aria-hidden="true" />
        <span>{databaseStatus.label}</span>
      </div>
      <div className={vectorStatus.className}>
        <HardDrive size={16} aria-hidden="true" />
        <span>{vectorStatus.label}</span>
      </div>
      <div>
        <span>{indexCount} tickers</span>
      </div>
      {checkedTime ? (
        <div className="status-muted">
          <span>checked {checkedTime}</span>
        </div>
      ) : null}
    </footer>
  );
}

type StatusKind = "ok" | "bad" | "checking" | "muted";

type StatusDescriptor = {
  status: StatusKind;
  className: string;
  label: string;
};

function getApiStatus(
  health: HealthResponse | null,
  healthError: string | null,
  isHealthLoading: boolean,
): StatusDescriptor {
  if (isHealthLoading && !health) {
    return {
      status: "checking",
      className: "status-checking",
      label: "API checking",
    };
  }

  if (healthError) {
    return {
      status: "bad",
      className: "status-bad",
      label: "API unreachable",
    };
  }

  if (health) {
    return {
      status: "ok",
      className: "status-ok",
      label: "API online",
    };
  }

  return {
    status: "muted",
    className: "status-muted",
    label: "API unknown",
  };
}

function getTargetStatus(isReady: boolean | undefined, label: "Database" | "Vector store"): StatusDescriptor {
  if (isReady === true) {
    return {
      status: "ok",
      className: "status-ok",
      label: `${label} ready`,
    };
  }

  if (isReady === false) {
    return {
      status: "bad",
      className: "status-bad",
      label: `${label} unavailable`,
    };
  }

  return {
    status: "muted",
    className: "status-muted",
    label: `${label} unknown`,
  };
}

function StatusIcon({ status }: { status: StatusKind }) {
  if (status === "ok") {
    return <CircleCheck size={16} aria-hidden="true" />;
  }

  if (status === "bad") {
    return <CircleX size={16} aria-hidden="true" />;
  }

  if (status === "checking") {
    return <LoaderCircle size={16} aria-hidden="true" />;
  }

  return <Server size={16} aria-hidden="true" />;
}
