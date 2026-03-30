from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from pipeline.orchestrator import PipelineOrchestrator
from schemas.pipeline import AdaptivePromotionReview, ArtifactRef


def _artifact(path: Path, kind: str) -> ArtifactRef:
    return ArtifactRef(
        name=path.name,
        path=str(path),
        kind=kind,
        size_bytes=path.stat().st_size,
    )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_adaptive_context(artifact_root: str, run_id: str) -> Dict[str, Any]:
    orchestrator = PipelineOrchestrator(artifact_root=artifact_root)
    bundle = orchestrator.load_run(run_id)
    manifest = bundle.get("manifest") or {}
    if not manifest:
        raise ValueError(f"Run {run_id} was not found under {artifact_root}.")
    summary = bundle.get("summary") or {}
    result = bundle.get("result") or {}
    adaptive = summary.get("adaptive") or result.get("adaptive") or manifest.get("adaptive") or {}
    if not adaptive:
        raise ValueError(f"Run {run_id} does not contain adaptive artifacts to review.")
    return adaptive


def persist_promotion_review(
    artifact_root: str = "artifacts",
    run_id: Optional[str] = None,
    *,
    reviewer: str,
    decision: str,
    notes: str = "",
) -> AdaptivePromotionReview:
    orchestrator = PipelineOrchestrator(artifact_root=artifact_root)
    target_run_id = run_id or orchestrator.latest_run_id()
    if not target_run_id:
        raise ValueError("No completed or persisted runs were found under the artifact root.")

    normalized_decision = str(decision or "").strip().lower()
    if normalized_decision not in {"approve", "reject"}:
        raise ValueError("decision must be either 'approve' or 'reject'.")

    reviewer_name = str(reviewer or "").strip()
    if not reviewer_name:
        raise ValueError("reviewer is required to persist a manual promotion review.")

    adaptive = _load_adaptive_context(artifact_root, target_run_id)
    approval = adaptive.get("approval") or {}
    promotion = adaptive.get("promotion") or {}

    if normalized_decision == "approve":
        if approval.get("status") != "approved_for_promotion_review":
            raise ValueError("Adaptive candidate is not approved by policy for manual promotion review.")
        if not promotion.get("eligible"):
            raise ValueError("Adaptive candidate is not promotion-eligible yet.")

    reasons = [
        "Manual promotion review was persisted after the adaptive pipeline finished.",
        "This artifact records operator signoff without auto-applying live configuration changes.",
    ]
    if normalized_decision == "approve":
        reasons.append("The reviewer approved the adaptive candidate for tracked promotion.")
    else:
        reasons.append("The reviewer rejected the adaptive candidate and kept the current baseline.")

    proposed_changes = dict((approval.get("proposed_changes") or {}))
    payload = {
        "run_id": target_run_id,
        "status": "approved" if normalized_decision == "approve" else "rejected",
        "approved": normalized_decision == "approve",
        "reviewer": reviewer_name,
        "reviewed_at": _utc_now_iso(),
        "notes": notes.strip(),
        "policy_status": approval.get("status") or "",
        "promotion_mode": promotion.get("mode") or "",
        "proposed_changes": proposed_changes,
        "reasons": reasons,
    }

    adaptive_dir = Path(artifact_root) / "runs" / target_run_id / "adaptive"
    adaptive_dir.mkdir(parents=True, exist_ok=True)
    path = adaptive_dir / "promotion_review.json"
    path.write_text(json.dumps(payload, indent=2))
    return AdaptivePromotionReview(
        **payload,
        artifact=_artifact(path, "adaptive_promotion_review"),
    )
