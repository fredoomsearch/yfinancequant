from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

from pipeline.orchestrator import PipelineOrchestrator
from schemas.pipeline import AdaptivePromotionExecution, ArtifactRef


def _artifact(path: Path, kind: str) -> ArtifactRef:
    return ArtifactRef(
        name=path.name,
        path=str(path),
        kind=kind,
        size_bytes=path.stat().st_size,
    )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_run_context(artifact_root: str, run_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    orchestrator = PipelineOrchestrator(artifact_root=artifact_root)
    bundle = orchestrator.load_run(run_id)
    manifest = bundle.get("manifest") or {}
    if not manifest:
        raise ValueError(f"Run {run_id} was not found under {artifact_root}.")
    summary = bundle.get("summary") or {}
    result = bundle.get("result") or {}
    adaptive = summary.get("adaptive") or result.get("adaptive") or manifest.get("adaptive") or {}
    if not adaptive:
        raise ValueError(f"Run {run_id} does not contain adaptive artifacts to promote.")
    adaptive_dir = Path(artifact_root) / "runs" / run_id / "adaptive"
    application_path = adaptive_dir / "promotion_application.json"
    if application_path.exists():
        adaptive["promotion_application"] = json.loads(application_path.read_text())
    return bundle, adaptive


def persist_promotion_execution(
    artifact_root: str = "artifacts",
    run_id: str | None = None,
    *,
    operator: str,
    notes: str = "",
) -> AdaptivePromotionExecution:
    orchestrator = PipelineOrchestrator(artifact_root=artifact_root)
    target_run_id = run_id or orchestrator.latest_run_id()
    if not target_run_id:
        raise ValueError("No completed or persisted runs were found under the artifact root.")

    operator_name = str(operator or "").strip()
    if not operator_name:
        raise ValueError("operator is required to mark a promotion as applied.")

    _, adaptive = _load_run_context(artifact_root, target_run_id)
    application = adaptive.get("promotion_application") or {}
    if application.get("status") != "prepared":
        raise ValueError("Adaptive candidate must be prepared before it can be marked as applied.")

    adaptive_dir = Path(artifact_root) / "runs" / target_run_id / "adaptive"
    adaptive_dir.mkdir(parents=True, exist_ok=True)
    applied_at = _utc_now_iso()
    config_version = str(application.get("config_version") or "").strip()
    application_artifact = ArtifactRef(**dict(application.get("artifact") or {})) if application.get("artifact") else None
    if application_artifact is None:
        application_path = adaptive_dir / "promotion_application.json"
        if application_path.exists():
            application_artifact = _artifact(application_path, "adaptive_promotion_application")

    reasons = [
        "The prepared adaptive candidate was marked as applied through a traced manual operation.",
        "This artifact closes the promotion lifecycle from review to package to applied state.",
    ]
    payload = {
        "run_id": target_run_id,
        "status": "applied",
        "applied_by": operator_name,
        "applied_at": applied_at,
        "notes": notes.strip(),
        "config_version": config_version,
        "reasons": reasons,
        "application_artifact": application_artifact.model_dump(mode="json") if application_artifact else None,
    }

    path = adaptive_dir / "promotion_execution.json"
    path.write_text(json.dumps(payload, indent=2))
    return AdaptivePromotionExecution(
        run_id=target_run_id,
        status="applied",
        applied_by=operator_name,
        applied_at=applied_at,
        notes=notes.strip(),
        config_version=config_version,
        reasons=reasons,
        application_artifact=application_artifact,
        artifact=_artifact(path, "adaptive_promotion_execution"),
    )
