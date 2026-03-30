from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from schemas.pipeline import (
    ArtifactRef,
    OperationsReport,
    ReleaseSummary,
    RuntimeObservability,
    SoakGate,
    VerifyGate,
)


_STAGE_RULES = {
    "orchestrator": {"start": {"run_started"}, "finish": {"run_finished", "run_failed"}},
    "extraction": {"start": {"starting_extraction"}, "finish": {"extraction_finished", "extraction_failed"}},
    "cleaning": {"start": {"starting_cleaning"}, "finish": {"cleaning_finished", "cleaning_failed"}},
    "modeling": {"start": {"starting_modeling"}, "finish": {"modeling_finished", "modeling_failed"}},
    "legacy_bridge": {"start": {"starting_legacy_bridge"}, "finish": {"legacy_bridge_completed", "legacy_bridge_failed"}},
    "reviewer": {"start": set(), "finish": {"review_completed"}},
    "brain": {"start": set(), "finish": {"brain_completed"}},
    "source_comparison": {"start": set(), "finish": {"source_comparison_completed", "source_comparison_skipped", "source_comparison_failed"}},
    "adaptive": {"start": set(), "finish": {"adaptive_report_generated"}},
    "operations": {"start": set(), "finish": {"operations_report_generated"}},
}


def _artifact(path: Path, kind: str) -> ArtifactRef:
    return ArtifactRef(
        name=path.name,
        path=str(path),
        kind=kind,
        size_bytes=path.stat().st_size,
    )


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _artifact_kind_counts(artifacts: Iterable[Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for artifact in artifacts:
        kind = str(getattr(artifact, "kind", "") or (artifact.get("kind") if isinstance(artifact, dict) else "") or "unknown")
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def _status_counts(logs: Iterable[Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for log in logs:
        status = getattr(log, "status", None)
        if hasattr(status, "value"):
            status = status.value
        status_name = str(status or "unknown")
        counts[status_name] = counts.get(status_name, 0) + 1
    return counts


def _agent_event_counts(logs: Iterable[Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for log in logs:
        agent = str(getattr(log, "agent", "") or "unknown")
        counts[agent] = counts.get(agent, 0) + 1
    return counts


def _stage_snapshot(logs: Iterable[Any]) -> tuple[Dict[str, str], Dict[str, int], Optional[datetime], Optional[datetime]]:
    stage_statuses: Dict[str, str] = {}
    stage_durations_ms: Dict[str, int] = {}
    stage_start_times: Dict[str, datetime] = {}
    run_started_at: Optional[datetime] = None
    run_finished_at: Optional[datetime] = None

    for log in logs:
        message = str(getattr(log, "message", "") or "")
        started_at = _parse_timestamp(getattr(log, "started_at", None))
        if message == "run_started" and started_at:
            run_started_at = started_at
        if message in {"run_finished", "run_failed"} and started_at:
            run_finished_at = started_at

        for stage, rule in _STAGE_RULES.items():
            if message in rule["start"] and started_at:
                stage_start_times.setdefault(stage, started_at)
                stage_statuses[stage] = "running"
            if message in rule["finish"]:
                if "failed" in message:
                    stage_statuses[stage] = "failed"
                elif "skipped" in message:
                    stage_statuses[stage] = "skipped"
                else:
                    stage_statuses[stage] = "completed"
                if started_at:
                    start = stage_start_times.get(stage, started_at)
                    delta_ms = max(0, int((started_at - start).total_seconds() * 1000))
                    stage_durations_ms[stage] = delta_ms

    for stage in _STAGE_RULES:
        stage_statuses.setdefault(stage, "not_executed")
        stage_durations_ms.setdefault(stage, 0)

    return stage_statuses, stage_durations_ms, run_started_at, run_finished_at


class OperationsReportBuilder:
    def build(
        self,
        *,
        manifest,
        run_dir: Path,
        run_mode: str,
        extraction=None,
        cleaning=None,
        modeling=None,
        adaptive_report=None,
    ) -> OperationsReport:
        ops_dir = run_dir / "ops"
        ops_dir.mkdir(parents=True, exist_ok=True)

        runtime_fingerprint_id = ""
        if adaptive_report and adaptive_report.runtime_fingerprint:
            runtime_fingerprint_id = adaptive_report.runtime_fingerprint.fingerprint_id

        stage_statuses, stage_durations_ms, run_started_at, run_finished_at = _stage_snapshot(manifest.logs)
        if run_started_at and not run_finished_at:
            run_finished_at = _parse_timestamp(getattr(manifest, "updated_at", None))
        run_duration_ms = 0
        if run_started_at and run_finished_at:
            run_duration_ms = max(0, int((run_finished_at - run_started_at).total_seconds() * 1000))

        observability_notes = [
            "Operational artifacts were captured for this run.",
            "Runtime fingerprint and decision path are persisted for reproducibility.",
        ]
        observability_payload = {
            "status": "captured",
            "runtime_fingerprint_id": runtime_fingerprint_id,
            "run_started_at": run_started_at.isoformat() if run_started_at else "",
            "run_finished_at": run_finished_at.isoformat() if run_finished_at else "",
            "run_duration_ms": run_duration_ms,
            "log_count": len(manifest.logs),
            "artifact_count": len(manifest.artifacts),
            "decision_path": (manifest.motor or {}).get("decision_path", "n/a"),
            "review_mode": manifest.request.review_mode,
            "run_mode": run_mode or "local_only",
            "status_counts": _status_counts(manifest.logs),
            "agent_event_counts": _agent_event_counts(manifest.logs),
            "artifact_kind_counts": _artifact_kind_counts(manifest.artifacts),
            "stage_statuses": stage_statuses,
            "stage_durations_ms": stage_durations_ms,
            "data_profile": {
                "ticker_count": len(getattr(manifest.request, "tickers", []) or []),
                "raw_rows": getattr(extraction, "rows", 0) if extraction else 0,
                "clean_rows": getattr(cleaning, "rows_out", 0) if cleaning else 0,
                "feature_count": len(getattr(cleaning, "feature_columns", []) or []) if cleaning else 0,
                "model_count": len(getattr(modeling, "models", []) or []) if modeling else 0,
            },
            "decision_profile": {
                "final_decision": getattr(manifest, "decision", None),
                "final_confidence": getattr(manifest, "confidence", None),
                "deterministic_decision": getattr(manifest, "deterministic_decision", None),
                "deterministic_confidence": getattr(manifest, "deterministic_confidence", None),
                "selected_model": (manifest.motor or {}).get("selected"),
                "reviewer_used": bool(getattr(manifest, "reviewer_used", False)),
                "brain_used": bool(getattr(manifest, "groq_brain_used", False)),
                "compare_binance": bool(getattr(manifest.request, "compare_binance", False)),
                "legacy_enabled": bool((manifest.motor or {}).get("legacy_enabled", False)),
                "adaptive_mode": getattr(adaptive_report, "mode", None),
                "adaptive_validation": getattr(getattr(adaptive_report, "validation", None), "status", None),
            },
            "notes": observability_notes,
        }
        observability_path = ops_dir / "runtime_observability.json"
        observability_path.write_text(json.dumps(observability_payload, indent=2))
        observability = RuntimeObservability(
            **observability_payload,
            artifact=_artifact(observability_path, "runtime_observability"),
        )

        checks = {
            "manifest_present": True,
            "result_present": True,
            "decision_present": bool(manifest.decision),
            "summary_artifact_present": True,
            "adaptive_present": adaptive_report is not None,
            "adaptive_approval_present": bool(
                adaptive_report and getattr(getattr(adaptive_report, "approval", None), "artifact", None)
            ),
            "logs_present": len(manifest.logs) > 0,
            "observability_metrics_present": bool(run_started_at) and "orchestrator" in stage_statuses,
        }
        verify_reasons = []
        if all(checks.values()):
            verify_reasons.append("Core runtime, decision, logs, and adaptive artifacts are present.")
        else:
            verify_reasons.append("One or more core artifacts are missing.")
        verify_payload = {
            "ok": all(checks.values()),
            "checks": checks,
            "reasons": verify_reasons,
        }
        verify_path = ops_dir / "verify_gate.json"
        verify_path.write_text(json.dumps(verify_payload, indent=2))
        verify_gate = VerifyGate(
            ok=verify_payload["ok"],
            checks=checks,
            reasons=verify_reasons,
            artifact=_artifact(verify_path, "verify_gate"),
        )

        soak_reasons = [
            "A real 72h soak test was not executed inside the per-run orchestrator flow.",
            "The soak gate remains pending until long-duration operational validation is run externally.",
        ]
        soak_payload = {
            "ok": False,
            "executed": False,
            "status": "not_executed",
            "reasons": soak_reasons,
        }
        soak_path = ops_dir / "soak_gate.json"
        soak_path.write_text(json.dumps(soak_payload, indent=2))
        soak_gate = SoakGate(
            ok=False,
            executed=False,
            status="not_executed",
            reasons=soak_reasons,
            artifact=_artifact(soak_path, "soak_gate"),
        )

        if verify_gate.ok and soak_gate.ok:
            release_stage = "release_ready"
            release_ok = True
            release_reasons = ["Verify gate and soak gate are green."]
        elif verify_gate.ok:
            release_stage = "ops_pending"
            release_ok = False
            release_reasons = ["Core verification passed, but soak validation is still pending."]
        else:
            release_stage = "dev_ready"
            release_ok = False
            release_reasons = ["Core verification is not green yet."]

        release_payload = {
            "ok": release_ok,
            "release_stage": release_stage,
            "reasons": release_reasons,
        }
        release_path = ops_dir / "release_summary.json"
        release_path.write_text(json.dumps(release_payload, indent=2))
        release_summary = ReleaseSummary(
            ok=release_ok,
            release_stage=release_stage,
            reasons=release_reasons,
            artifact=_artifact(release_path, "release_summary"),
        )

        return OperationsReport(
            observability=observability,
            verify_gate=verify_gate,
            soak_gate=soak_gate,
            release_summary=release_summary,
        )
