from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

from pipeline.orchestrator import PipelineOrchestrator
from schemas.pipeline import AdaptiveCandidateConfig, AdaptivePromotionApplication, ArtifactRef


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
        raise ValueError(f"Run {run_id} does not contain adaptive artifacts to apply.")
    review_path = Path(artifact_root) / "runs" / run_id / "adaptive" / "promotion_review.json"
    if review_path.exists():
        adaptive["promotion_review"] = json.loads(review_path.read_text())
    return bundle, adaptive


def _next_config_version(adaptive_dir: Path, run_id: str) -> str:
    existing = sorted(adaptive_dir.glob("candidate_config_v*.json"))
    next_index = len(existing) + 1
    return f"{run_id}-v{next_index:03d}"


def persist_promotion_application(
    artifact_root: str = "artifacts",
    run_id: str | None = None,
    *,
    operator: str,
    notes: str = "",
) -> AdaptivePromotionApplication:
    orchestrator = PipelineOrchestrator(artifact_root=artifact_root)
    target_run_id = run_id or orchestrator.latest_run_id()
    if not target_run_id:
        raise ValueError("No completed or persisted runs were found under the artifact root.")

    operator_name = str(operator or "").strip()
    if not operator_name:
        raise ValueError("operator is required to prepare a promotion application.")

    bundle, adaptive = _load_run_context(artifact_root, target_run_id)
    promotion_review = adaptive.get("promotion_review") or {}
    if promotion_review.get("status") != "approved" or not promotion_review.get("approved"):
        raise ValueError("Adaptive candidate must be manually approved before preparing promotion artifacts.")

    approval = adaptive.get("approval") or {}
    if approval.get("status") != "approved_for_promotion_review":
        raise ValueError("Adaptive candidate is not policy-approved for promotion packaging.")

    adaptive_dir = Path(artifact_root) / "runs" / target_run_id / "adaptive"
    adaptive_dir.mkdir(parents=True, exist_ok=True)
    prepared_at = _utc_now_iso()
    config_version = _next_config_version(adaptive_dir, target_run_id)

    candidate_payload = {
        "version": config_version,
        "run_id": target_run_id,
        "created_at": prepared_at,
        "source_review_status": promotion_review.get("status") or "",
        "source_reviewer": promotion_review.get("reviewer") or "",
        "proposed_changes": dict(approval.get("proposed_changes") or {}),
        "runtime_fingerprint_id": ((adaptive.get("runtime_fingerprint") or {}).get("fingerprint_id") or ""),
        "feature_registry_version": ((adaptive.get("feature_registry") or {}).get("version") or ""),
    }
    config_path = adaptive_dir / f"candidate_config_{config_version}.json"
    config_path.write_text(json.dumps(candidate_payload, indent=2))
    config_artifact = _artifact(config_path, "adaptive_candidate_config")
    candidate_config = AdaptiveCandidateConfig(**candidate_payload, artifact=config_artifact)

    reasons = [
        "Promotion packaging was prepared from a manually approved adaptive candidate.",
        "This artifact is versioned and traceable; it does not auto-apply live production changes.",
    ]
    application_payload = {
        "run_id": target_run_id,
        "status": "prepared",
        "prepared_by": operator_name,
        "prepared_at": prepared_at,
        "notes": notes.strip(),
        "config_version": config_version,
        "reasons": reasons,
        "config_artifact": config_artifact.model_dump(mode="json"),
    }
    application_path = adaptive_dir / "promotion_application.json"
    application_path.write_text(json.dumps(application_payload, indent=2))
    application_artifact = _artifact(application_path, "adaptive_promotion_application")
    return AdaptivePromotionApplication(
        run_id=target_run_id,
        status="prepared",
        prepared_by=operator_name,
        prepared_at=prepared_at,
        notes=notes.strip(),
        config_version=config_version,
        reasons=reasons,
        config_artifact=config_artifact,
        artifact=application_artifact,
    )
