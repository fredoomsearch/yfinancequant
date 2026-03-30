from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from pipeline.orchestrator import PipelineOrchestrator
from ops.job_runner import persist_operations_job
from schemas.pipeline import ArtifactRef, OperationsSchedule, OperationsScheduleRunReport


def _artifact(path: Path, kind: str) -> ArtifactRef:
    return ArtifactRef(
        name=path.name,
        path=str(path),
        kind=kind,
        size_bytes=path.stat().st_size,
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _parse_iso(value: str | None) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _resolve_run_id(artifact_root: str, run_id: Optional[str]) -> str:
    orchestrator = PipelineOrchestrator(artifact_root=artifact_root)
    target_run_id = run_id or orchestrator.latest_run_id()
    if not target_run_id:
        raise ValueError("No completed or persisted runs were found under the artifact root.")
    bundle = orchestrator.load_run(target_run_id)
    if not bundle.get("manifest"):
        raise ValueError(f"Run {target_run_id} was not found under {artifact_root}.")
    return target_run_id


def persist_operations_schedule(
    artifact_root: str = "artifacts",
    run_id: Optional[str] = None,
    *,
    interval_seconds: int = 300,
    limit: int = 10,
    required_hours: int = 72,
    include_soak: bool = True,
    enabled: bool = True,
    start_immediately: bool = False,
) -> OperationsSchedule:
    if interval_seconds < 0:
        raise ValueError("interval_seconds cannot be negative")
    if limit < 1:
        raise ValueError("limit must be at least 1")
    target_run_id = _resolve_run_id(artifact_root, run_id)
    now = _utc_now()
    next_run_at = now if start_immediately else now + timedelta(seconds=interval_seconds)

    schedule = OperationsSchedule(
        artifact_root=artifact_root,
        run_id=target_run_id,
        enabled=enabled,
        interval_seconds=interval_seconds,
        limit=limit,
        required_hours=required_hours,
        include_soak=include_soak,
        created_at=now.isoformat(),
        updated_at=now.isoformat(),
        last_run_at="",
        next_run_at=next_run_at.isoformat(),
    )
    path = Path(artifact_root) / "runs" / target_run_id / "ops" / "schedule.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = schedule.model_dump(mode="json")
    payload.pop("artifact", None)
    path.write_text(json.dumps(payload, indent=2))
    schedule.artifact = _artifact(path, "operations_schedule")
    path.write_text(json.dumps(schedule.model_dump(mode="json"), indent=2))
    return schedule


def run_due_operations_schedule(
    artifact_root: str = "artifacts",
    run_id: Optional[str] = None,
    *,
    force: bool = False,
) -> OperationsScheduleRunReport:
    target_run_id = _resolve_run_id(artifact_root, run_id)
    path = Path(artifact_root) / "runs" / target_run_id / "ops" / "schedule.json"
    if not path.exists():
        raise ValueError(f"Run {target_run_id} does not contain an operations schedule.")

    schedule_payload = json.loads(path.read_text())
    schedule = OperationsSchedule(**schedule_payload)
    checked_at = _utc_now()
    due = bool(schedule.enabled) and (_parse_iso(schedule.next_run_at) or checked_at) <= checked_at

    report = OperationsScheduleRunReport(
        artifact_root=artifact_root,
        run_id=target_run_id,
        forced=force,
        due=due or force,
        executed=False,
        checked_at=checked_at.isoformat(),
        reason="",
        schedule=schedule,
    )

    if not schedule.enabled:
        report.reason = "The operations schedule is disabled."
    elif not (due or force):
        report.reason = "The operations schedule is not due yet."
    else:
        job = persist_operations_job(
            artifact_root=artifact_root,
            run_id=target_run_id,
            cycles=1,
            interval_seconds=0,
            limit=schedule.limit,
            required_hours=schedule.required_hours,
            include_soak=schedule.include_soak,
        )
        now = _utc_now()
        schedule.updated_at = now.isoformat()
        schedule.last_run_at = now.isoformat()
        schedule.next_run_at = (now + timedelta(seconds=schedule.interval_seconds)).isoformat()
        updated_payload = schedule.model_dump(mode="json")
        updated_payload.pop("artifact", None)
        path.write_text(json.dumps(updated_payload, indent=2))
        schedule.artifact = _artifact(path, "operations_schedule")
        path.write_text(json.dumps(schedule.model_dump(mode="json"), indent=2))
        report.executed = True
        report.reason = "The due operations schedule executed one refresh cycle."
        report.schedule = schedule
        report.job = job

    report_path = Path(artifact_root) / "runs" / target_run_id / "ops" / "schedule_run_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_payload = report.model_dump(mode="json")
    report_payload.pop("artifact", None)
    report_path.write_text(json.dumps(report_payload, indent=2))
    report.artifact = _artifact(report_path, "operations_schedule_run_report")
    report_path.write_text(json.dumps(report.model_dump(mode="json"), indent=2))
    return report
