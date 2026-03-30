from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ops.refresh import persist_operations_refresh
from schemas.pipeline import ArtifactRef, OperationsJobCycle, OperationsJobReport


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _artifact(path: Path, kind: str) -> ArtifactRef:
    return ArtifactRef(
        name=path.name,
        path=str(path),
        kind=kind,
        size_bytes=path.stat().st_size,
    )


def persist_operations_job(
    artifact_root: str = "artifacts",
    run_id: Optional[str] = None,
    *,
    cycles: int = 1,
    interval_seconds: int = 0,
    limit: int = 10,
    required_hours: int = 72,
    include_soak: bool = True,
) -> OperationsJobReport:
    if cycles < 1:
        raise ValueError("cycles must be at least 1")
    if interval_seconds < 0:
        raise ValueError("interval_seconds cannot be negative")

    completed_cycles: list[OperationsJobCycle] = []
    target_run_id = run_id
    latest_release_status = "blocked"
    latest_soak_status = "skipped"

    for cycle_index in range(1, cycles + 1):
        started_at = _utc_now_iso()
        refresh = persist_operations_refresh(
            artifact_root=artifact_root,
            run_id=target_run_id,
            limit=limit,
            required_hours=required_hours,
            include_soak=include_soak,
        )
        finished_at = _utc_now_iso()
        target_run_id = refresh.run_id
        latest_release_status = refresh.release_gate.status
        latest_soak_status = ((refresh.soak or {}).get("soak_gate") or {}).get("status", "skipped")
        completed_cycles.append(
            OperationsJobCycle(
                cycle_index=cycle_index,
                run_id=refresh.run_id,
                readyz_status=refresh.readyz.status,
                release_status=refresh.release_gate.status,
                soak_status=latest_soak_status,
                started_at=started_at,
                finished_at=finished_at,
            )
        )
        if cycle_index < cycles and interval_seconds > 0:
            time.sleep(interval_seconds)

    if not target_run_id:
        raise ValueError("No completed or persisted runs were found under the artifact root.")

    report = OperationsJobReport(
        artifact_root=artifact_root,
        run_id=target_run_id,
        cycles_requested=cycles,
        cycles_completed=len(completed_cycles),
        interval_seconds=interval_seconds,
        include_soak=include_soak,
        latest_release_status=latest_release_status,
        latest_soak_status=latest_soak_status,
        cycles=completed_cycles,
    )

    path = Path(artifact_root) / "runs" / target_run_id / "ops" / "job_runner_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.model_dump(mode="json")
    payload.pop("artifact", None)
    path.write_text(json.dumps(payload, indent=2))
    report.artifact = _artifact(path, "operations_job_runner")
    path.write_text(json.dumps(report.model_dump(mode="json"), indent=2))
    return report
