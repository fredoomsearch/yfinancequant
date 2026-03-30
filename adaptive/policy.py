from __future__ import annotations

import json
from pathlib import Path

from schemas.pipeline import AdaptiveApproval, ArtifactRef


class AdaptivePolicyEngine:
    def __init__(self) -> None:
        self.allowed_models = {
            "ensemble",
            "majority",
            "logistic_regression",
            "random_forest",
            "gradient_boosting",
        }
        self.minimum_approved_feature_pct = 85.0

    def evaluate(
        self,
        *,
        request,
        drift,
        selection,
        feature_registry,
        shadow,
        validation,
        retraining,
        run_dir: Path,
    ) -> AdaptiveApproval:
        threshold = float(selection.recommended_confidence_threshold or request.confidence_threshold or 0.0)
        review_mode = str(selection.recommended_review_mode or request.review_mode or "auto")
        recommended_model = str(selection.recommended_model or "")

        checks = {
            "approved_feature_catalog": feature_registry.approved_pct >= self.minimum_approved_feature_pct,
            "recommended_model_allowed": recommended_model in self.allowed_models,
            "no_auto_apply_mode": selection.mode != "auto_apply",
            "review_mode_hardened_if_escalated": True,
            "threshold_hardened_if_drifted": True,
            "shadow_completed_for_candidate": True,
            "retraining_plan_present_for_escalation": True,
        }

        escalated = validation.status in {"shadow_only", "candidate_ready"} or drift.level in {"watch", "drifted"}
        if escalated:
            checks["review_mode_hardened_if_escalated"] = review_mode == "on"
            checks["retraining_plan_present_for_escalation"] = retraining.status in {"scheduled", "immediate"}
        if drift.level == "drifted":
            checks["threshold_hardened_if_drifted"] = threshold >= max(0.65, float(request.confidence_threshold))
        if validation.status == "candidate_ready":
            checks["shadow_completed_for_candidate"] = bool(shadow.executed and shadow.ready_for_promotion)

        critical_checks = {
            "approved_feature_catalog",
            "recommended_model_allowed",
            "no_auto_apply_mode",
            "review_mode_hardened_if_escalated",
            "threshold_hardened_if_drifted",
            "shadow_completed_for_candidate",
            "retraining_plan_present_for_escalation",
        }
        blocked = any(not checks[name] for name in critical_checks)

        reasons: list[str] = [
            "Adaptive changes are governed by explicit approval checks before promotion review.",
            "The policy engine only authorizes controlled, manually promoted changes.",
        ]

        if not checks["approved_feature_catalog"]:
            reasons.append(
                f"Observed features cover only {feature_registry.approved_pct:.2f}% of the approved catalog."
            )
        if not checks["recommended_model_allowed"]:
            reasons.append(f"Recommended model {recommended_model or 'n/a'} is outside the approved model registry.")
        if not checks["no_auto_apply_mode"]:
            reasons.append("Auto-apply mode is blocked in the current architecture phase.")
        if not checks["review_mode_hardened_if_escalated"]:
            reasons.append("Escalated adaptive candidates must enable reviewer mode before approval.")
        if not checks["threshold_hardened_if_drifted"]:
            reasons.append("Drifted candidates must harden the confidence threshold to at least 0.65.")
        if not checks["shadow_completed_for_candidate"]:
            reasons.append("Candidate-ready changes require a completed shadow benchmark before approval.")
        if not checks["retraining_plan_present_for_escalation"]:
            reasons.append("Escalated adaptive changes must carry a scheduled retraining plan.")

        if selection.recommended_action == "keep_current_strategy" and validation.status == "observe_only" and not blocked:
            status = "observe_only"
            reasons.append("No adaptive change is proposed for the current run.")
        elif blocked:
            status = "blocked"
            reasons.append("The adaptive candidate is blocked until policy checks are green.")
        elif validation.status == "candidate_ready":
            status = "approved_for_promotion_review"
            reasons.append("The adaptive candidate cleared policy for manual promotion review.")
        else:
            status = "manual_review"
            reasons.append("The adaptive candidate may proceed only through manual review and shadow validation.")

        adaptive_dir = run_dir / "adaptive"
        adaptive_dir.mkdir(parents=True, exist_ok=True)
        path = adaptive_dir / "approval_decision.json"
        payload = {
            "status": status,
            "requires_manual_signoff": True,
            "auto_apply_allowed": False,
            "proposed_changes": {
                "action": selection.recommended_action,
                "model": recommended_model or None,
                "review_mode": review_mode,
                "confidence_threshold": round(threshold, 4),
                "retraining_within_hours": retraining.recommended_within_hours,
            },
            "checks": checks,
            "reasons": reasons,
        }
        path.write_text(json.dumps(payload, indent=2))
        artifact = ArtifactRef(
            name=path.name,
            path=str(path),
            kind="adaptive_approval",
            size_bytes=path.stat().st_size,
        )
        return AdaptiveApproval(
            status=status,
            artifact=artifact,
            requires_manual_signoff=True,
            auto_apply_allowed=False,
            proposed_changes=payload["proposed_changes"],
            checks=checks,
            reasons=reasons,
        )
