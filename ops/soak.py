from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from pipeline.orchestrator import PipelineOrchestrator
from schemas.pipeline import ArtifactRef, ReleaseSummary, SoakGate


def _artifact(path: Path, kind: str) -> ArtifactRef:
    return ArtifactRef(
        name=path.name,
        path=str(path),
        kind=kind,
        size_bytes=path.stat().st_size,
    )


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _load_run_observability(artifact_root: str, run_id: str) -> Dict[str, Any]:
    run_dir = Path(artifact_root) / "runs" / run_id
    path = run_dir / "ops" / "runtime_observability.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _load_verify_gate(artifact_root: str, run_id: str) -> Dict[str, Any]:
    run_dir = Path(artifact_root) / "runs" / run_id
    path = run_dir / "ops" / "verify_gate.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _build_release_summary_payload(*, verify_ok: bool, soak_gate: SoakGate) -> Dict[str, Any]:
    if verify_ok and soak_gate.ok:
        return {
            "ok": True,
            "release_stage": "release_ready",
            "reasons": ["Verify gate and soak gate are green."],
        }
    if verify_ok:
        return {
            "ok": False,
            "release_stage": "ops_pending",
            "reasons": ["Core verification passed, but soak validation is still pending."],
        }
    return {
        "ok": False,
        "release_stage": "dev_ready",
        "reasons": ["Core verification is not green yet."],
    }


def persist_soak_gate(
    artifact_root: str = "artifacts",
    run_id: Optional[str] = None,
    *,
    required_hours: int = 72,
    limit: int = 50,
) -> Dict[str, Any]:
    orchestrator = PipelineOrchestrator(artifact_root=artifact_root)
    target_run_id = run_id or orchestrator.latest_run_id()
    if not target_run_id:
        raise ValueError("No completed or persisted runs were found under the artifact root.")

    run_dir = Path(artifact_root) / "runs" / target_run_id
    if not run_dir.exists():
        raise ValueError(f"Run {target_run_id} was not found under {artifact_root}.")

    summaries = orchestrator.list_runs()
    recent_summaries = list(reversed(summaries[-max(1, limit):]))
    records: list[Dict[str, Any]] = []
    for summary in recent_summaries:
        candidate_run_id = summary.get("run_id")
        if not candidate_run_id:
            continue
        observability = _load_run_observability(artifact_root, candidate_run_id)
        verify_gate = _load_verify_gate(artifact_root, candidate_run_id)
        start = _parse_timestamp(observability.get("run_started_at"))
        end = _parse_timestamp(observability.get("run_finished_at"))
        if not start and not end:
            continue
        records.append(
            {
                "run_id": candidate_run_id,
                "start": start or end,
                "end": end or start,
                "verify_ok": bool(verify_gate.get("ok", False)),
                "status": summary.get("status"),
            }
        )

    if not records:
        soak_gate = SoakGate(
            ok=False,
            executed=False,
            status="not_executed",
            required_hours=required_hours,
            reasons=["No runtime observability window was available to evaluate soak behavior."],
        )
    else:
        window_start = min(record["start"] for record in records if record["start"])
        window_end = max(record["end"] for record in records if record["end"])
        observed_hours = max(0.0, (window_end - window_start).total_seconds() / 3600.0)
        sampled_runs = len(records)
        ready_runs = sum(1 for record in records if record["verify_ok"] and record["status"] == "succeeded")
        all_verified = ready_runs == sampled_runs
        any_failed = any(record["status"] not in {"succeeded"} for record in records)

        if observed_hours < required_hours:
            soak_gate = SoakGate(
                ok=False,
                executed=True,
                status="pending",
                required_hours=required_hours,
                observed_hours=round(observed_hours, 2),
                sampled_runs=sampled_runs,
                ready_runs=ready_runs,
                window_start=window_start.isoformat(),
                window_end=window_end.isoformat(),
                reasons=["The observed runtime window is still below the required soak threshold."],
            )
        elif any_failed or not all_verified:
            soak_gate = SoakGate(
                ok=False,
                executed=True,
                status="failed",
                required_hours=required_hours,
                observed_hours=round(observed_hours, 2),
                sampled_runs=sampled_runs,
                ready_runs=ready_runs,
                window_start=window_start.isoformat(),
                window_end=window_end.isoformat(),
                reasons=["At least one sampled run failed or did not pass the verify gate during the soak window."],
            )
        else:
            soak_gate = SoakGate(
                ok=True,
                executed=True,
                status="passed",
                required_hours=required_hours,
                observed_hours=round(observed_hours, 2),
                sampled_runs=sampled_runs,
                ready_runs=ready_runs,
                window_start=window_start.isoformat(),
                window_end=window_end.isoformat(),
                reasons=["The sampled runtime window met the soak threshold and all sampled runs passed verify."],
            )

    soak_path = run_dir / "ops" / "soak_gate.json"
    soak_payload = soak_gate.model_dump()
    soak_payload.pop("artifact", None)
    _write_json(soak_path, soak_payload)
    soak_gate.artifact = _artifact(soak_path, "soak_gate")

    verify_gate = _load_verify_gate(artifact_root, target_run_id)
    release_payload = _build_release_summary_payload(verify_ok=bool(verify_gate.get("ok", False)), soak_gate=soak_gate)
    release_path = run_dir / "ops" / "release_summary.json"
    _write_json(release_path, release_payload)
    release_summary = ReleaseSummary(**release_payload, artifact=_artifact(release_path, "release_summary"))

    return {
        "run_id": target_run_id,
        "soak_gate": soak_gate.model_dump(),
        "release_summary": release_summary.model_dump(),
    }
