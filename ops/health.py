from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from pipeline.orchestrator import PipelineOrchestrator
from schemas.pipeline import ReadinessReport, ReleaseBoard, ReleaseBoardEntry, ReleaseGateReport


def _read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _artifact_status(path: Path, *, required: bool = True) -> Dict[str, Any]:
    return {
        "path": str(path),
        "required": required,
        "exists": path.exists(),
    }


def _artifact_glob_status(directory: Path, pattern: str, *, required: bool = True) -> Dict[str, Any]:
    matches = sorted(directory.glob(pattern))
    return {
        "path": str(directory / pattern),
        "required": required,
        "exists": bool(matches),
        "matches": [str(path) for path in matches],
    }


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _evaluate_operational_policy(run_report: Dict[str, Any]) -> Dict[str, bool]:
    operations = run_report.get("operations") or {}
    observability = run_report.get("observability") or {}
    data_profile = observability.get("data_profile") or {}
    decision_profile = observability.get("decision_profile") or {}
    stage_statuses = observability.get("stage_statuses") or {}
    status_counts = observability.get("status_counts") or {}

    return {
        "run_succeeded": run_report.get("status") == "succeeded",
        "verify_gate_ok": bool(run_report["checks"]["operations_verify_ok"]),
        "required_artifacts_present": bool(run_report["all_required_artifacts_present"]),
        "core_stages_completed": all(
            stage_statuses.get(stage) == "completed"
            for stage in ("orchestrator", "extraction", "cleaning", "modeling")
        ),
        "failed_log_count_zero": _safe_int(status_counts.get("failed")) == 0,
        "decision_present": bool(run_report.get("decision", {}).get("final")),
        "raw_rows_positive": _safe_int(data_profile.get("raw_rows")) > 0,
        "clean_rows_positive": _safe_int(data_profile.get("clean_rows")) > 0,
        "feature_count_minimum": _safe_int(data_profile.get("feature_count")) >= 5,
        "model_count_minimum": _safe_int(data_profile.get("model_count")) >= 1,
        "run_duration_captured": _safe_int(observability.get("run_duration_ms")) >= 0,
        "release_stage_known": bool(operations.get("release_stage")),
        "decision_path_known": bool(observability.get("decision_path")),
        "selected_model_known": bool(decision_profile.get("selected_model")),
    }


def _evaluate_adaptive_release_policy(run_report: Dict[str, Any]) -> Dict[str, bool]:
    adaptive = run_report.get("adaptive") or {}
    promotion_mode = str(adaptive.get("promotion_mode") or "").strip()
    manual_review_status = str(adaptive.get("manual_review_status") or "").strip()
    manual_review_approved = bool(adaptive.get("manual_review_approved"))
    promotion_application_status = str(adaptive.get("promotion_application_status") or "").strip()

    requires_manual_review = promotion_mode == "promotion_ready"
    review_signed = (not requires_manual_review) or manual_review_approved
    application_prepared = (manual_review_status != "approved") or promotion_application_status == "prepared"
    review_not_rejected = manual_review_status != "rejected"

    return {
        "adaptive_review_not_rejected": review_not_rejected,
        "adaptive_review_signed_if_required": review_signed,
        "adaptive_application_prepared_if_reviewed": application_prepared,
    }


def _promotion_lifecycle_status(adaptive: Dict[str, Any]) -> str:
    execution_status = str(adaptive.get("promotion_execution_status") or "").strip()
    application_status = str(adaptive.get("promotion_application_status") or "").strip()
    review_status = str(adaptive.get("manual_review_status") or "").strip()
    if execution_status:
        return execution_status
    if application_status:
        return application_status
    if review_status:
        return review_status
    return "n/a"


def build_run_verification_report(artifact_root: str = "artifacts", run_id: Optional[str] = None) -> Dict[str, Any]:
    orchestrator = PipelineOrchestrator(artifact_root=artifact_root)
    selected_run_id = run_id or orchestrator.latest_run_id()
    if not selected_run_id:
        raise ValueError("No completed or persisted runs were found under the artifact root.")

    bundle = orchestrator.load_run(selected_run_id)
    manifest = bundle.get("manifest") or {}
    result = bundle.get("result") or {}
    summary = bundle.get("summary") or {}
    logs = bundle.get("logs") or []

    if not manifest:
        raise ValueError(f"Run {selected_run_id} was not found under {artifact_root}.")

    run_dir = Path(artifact_root) / "runs" / selected_run_id
    adaptive = summary.get("adaptive") or result.get("adaptive") or manifest.get("adaptive") or {}
    operations = summary.get("operations") or result.get("operations") or manifest.get("operations") or {}
    models = summary.get("models") or {}
    ops_dir = run_dir / "ops"
    persisted_observability = _read_json_if_exists(ops_dir / "runtime_observability.json")
    persisted_verify = _read_json_if_exists(ops_dir / "verify_gate.json")
    persisted_soak = _read_json_if_exists(ops_dir / "soak_gate.json")
    persisted_release = _read_json_if_exists(ops_dir / "release_summary.json")
    persisted_job_runner = _read_json_if_exists(ops_dir / "job_runner_report.json")
    persisted_schedule = _read_json_if_exists(ops_dir / "schedule.json")
    persisted_schedule_run = _read_json_if_exists(ops_dir / "schedule_run_report.json")
    persisted_automation = _read_json_if_exists(ops_dir / "automation_bundle.json")
    persisted_manual_review = _read_json_if_exists(run_dir / "adaptive" / "promotion_review.json")
    persisted_promotion_application = _read_json_if_exists(run_dir / "adaptive" / "promotion_application.json")
    persisted_promotion_execution = _read_json_if_exists(run_dir / "adaptive" / "promotion_execution.json")
    if persisted_observability:
        operations["observability"] = persisted_observability
    if persisted_verify:
        operations["verify_gate"] = persisted_verify
    if persisted_soak:
        operations["soak_gate"] = persisted_soak
    if persisted_release:
        operations["release_summary"] = persisted_release
    if persisted_job_runner:
        operations["job_runner"] = persisted_job_runner
    if persisted_schedule:
        operations["schedule"] = persisted_schedule
    if persisted_schedule_run:
        operations["schedule_run"] = persisted_schedule_run
    if persisted_automation:
        operations["automation"] = persisted_automation
    if persisted_manual_review:
        adaptive["promotion_review"] = persisted_manual_review
    if persisted_promotion_application:
        adaptive["promotion_application"] = persisted_promotion_application
    if persisted_promotion_execution:
        adaptive["promotion_execution"] = persisted_promotion_execution

    artifacts = {
        "manifest": _artifact_status(run_dir / "manifest.json"),
        "result": _artifact_status(run_dir / "result.json"),
        "summary": _artifact_status(run_dir / "summary.json"),
        "logs": _artifact_status(run_dir / "logs.json"),
        "adaptive_report": _artifact_status(run_dir / "adaptive" / "adaptive_report.json"),
        "adaptive_approval": _artifact_status(run_dir / "adaptive" / "approval_decision.json"),
        "adaptive_promotion_review": _artifact_status(run_dir / "adaptive" / "promotion_review.json", required=False),
        "adaptive_candidate_config": _artifact_glob_status(run_dir / "adaptive", "candidate_config_*.json", required=False),
        "adaptive_promotion_application": _artifact_status(run_dir / "adaptive" / "promotion_application.json", required=False),
        "adaptive_promotion_execution": _artifact_status(run_dir / "adaptive" / "promotion_execution.json", required=False),
        "feature_registry": _artifact_status(run_dir / "adaptive" / "feature_registry.json"),
        "retraining_plan": _artifact_status(run_dir / "adaptive" / "retraining_plan.json"),
        "runtime_fingerprint": _artifact_status(run_dir / "adaptive" / "runtime_fingerprint.json"),
        "runtime_observability": _artifact_status(run_dir / "ops" / "runtime_observability.json"),
        "verify_gate": _artifact_status(run_dir / "ops" / "verify_gate.json"),
        "soak_gate": _artifact_status(run_dir / "ops" / "soak_gate.json"),
        "release_summary": _artifact_status(run_dir / "ops" / "release_summary.json"),
        "operations_dashboard": _artifact_status(run_dir / "ops" / "operations_dashboard.json", required=False),
        "operations_dashboard_html": _artifact_status(run_dir / "ops" / "operations_dashboard.html", required=False),
        "operations_job_runner": _artifact_status(run_dir / "ops" / "job_runner_report.json", required=False),
        "operations_schedule": _artifact_status(run_dir / "ops" / "schedule.json", required=False),
        "operations_schedule_run": _artifact_status(run_dir / "ops" / "schedule_run_report.json", required=False),
        "operations_automation_bundle": _artifact_status(run_dir / "ops" / "automation_bundle.json", required=False),
        "operations_schedule_shell": _artifact_status(run_dir / "ops" / "run_ops_schedule.sh", required=False),
        "operations_schedule_cron": _artifact_status(run_dir / "ops" / "ops_schedule.crontab", required=False),
        "operations_schedule_service": _artifact_status(
            run_dir / "ops" / f"iactest-ops-{selected_run_id}.service", required=False
        ),
        "operations_schedule_timer": _artifact_status(
            run_dir / "ops" / f"iactest-ops-{selected_run_id}.timer", required=False
        ),
        "shadow_run": _artifact_status(run_dir / "adaptive" / "shadow_run.json", required=False),
    }

    checks = {
        "manifest_present": artifacts["manifest"]["exists"],
        "result_present": artifacts["result"]["exists"],
        "summary_present": artifacts["summary"]["exists"],
        "logs_present": artifacts["logs"]["exists"],
        "adaptive_present": artifacts["adaptive_report"]["exists"],
        "adaptive_approval_present": artifacts["adaptive_approval"]["exists"],
        "feature_registry_present": artifacts["feature_registry"]["exists"],
        "retraining_plan_present": artifacts["retraining_plan"]["exists"],
        "runtime_fingerprint_present": artifacts["runtime_fingerprint"]["exists"],
        "runtime_observability_present": artifacts["runtime_observability"]["exists"],
        "verify_gate_present": artifacts["verify_gate"]["exists"],
        "soak_gate_present": artifacts["soak_gate"]["exists"],
        "release_summary_present": artifacts["release_summary"]["exists"],
        "operations_verify_ok": bool((operations.get("verify_gate") or {}).get("ok", False)),
    }

    return {
        "run_id": selected_run_id,
        "artifact_root": artifact_root,
        "status": summary.get("status") or manifest.get("status"),
        "tickers": summary.get("tickers") or manifest.get("request", {}).get("tickers", []),
        "decision": {
            "final": models.get("final_decision") or result.get("final_decision") or manifest.get("decision"),
            "confidence": models.get("confidence") or result.get("final_confidence") or manifest.get("confidence"),
            "source": summary.get("brain", {}).get("decision_source") or result.get("decision_source") or manifest.get("decision_source"),
        },
        "adaptive": {
            "mode": adaptive.get("mode"),
            "drift_level": (adaptive.get("drift") or {}).get("level"),
            "validation_status": (adaptive.get("validation") or {}).get("status"),
            "approval_status": (adaptive.get("approval") or {}).get("status"),
            "manual_review_status": (adaptive.get("promotion_review") or {}).get("status"),
            "manual_review_approved": (adaptive.get("promotion_review") or {}).get("approved"),
            "manual_review_reviewer": (adaptive.get("promotion_review") or {}).get("reviewer"),
            "promotion_application_status": (adaptive.get("promotion_application") or {}).get("status"),
            "promotion_application_version": (adaptive.get("promotion_application") or {}).get("config_version"),
            "promotion_application_operator": (adaptive.get("promotion_application") or {}).get("prepared_by"),
            "promotion_execution_status": (adaptive.get("promotion_execution") or {}).get("status"),
            "promotion_execution_version": (adaptive.get("promotion_execution") or {}).get("config_version"),
            "promotion_execution_operator": (adaptive.get("promotion_execution") or {}).get("applied_by"),
            "promotion_lifecycle_status": _promotion_lifecycle_status(
                {
                    "promotion_execution_status": (adaptive.get("promotion_execution") or {}).get("status"),
                    "promotion_application_status": (adaptive.get("promotion_application") or {}).get("status"),
                    "manual_review_status": (adaptive.get("promotion_review") or {}).get("status"),
                }
            ),
            "promotion_mode": (adaptive.get("promotion") or {}).get("mode"),
            "promotion_eligible": (adaptive.get("promotion") or {}).get("eligible"),
            "shadow_status": (adaptive.get("shadow") or {}).get("status"),
            "retraining_status": (adaptive.get("retraining") or {}).get("status"),
            "feature_registry_version": (adaptive.get("feature_registry") or {}).get("version"),
        },
        "operations": {
            "run_mode": (operations.get("observability") or {}).get("run_mode"),
            "verify_ok": (operations.get("verify_gate") or {}).get("ok"),
            "soak_status": (operations.get("soak_gate") or {}).get("status"),
            "release_stage": (operations.get("release_summary") or {}).get("release_stage"),
            "job_cycles_completed": (operations.get("job_runner") or {}).get("cycles_completed"),
            "job_latest_release_status": (operations.get("job_runner") or {}).get("latest_release_status"),
            "schedule_enabled": (operations.get("schedule") or {}).get("enabled"),
            "schedule_next_run_at": (operations.get("schedule") or {}).get("next_run_at"),
            "schedule_last_run_at": (operations.get("schedule") or {}).get("last_run_at"),
            "schedule_last_execution_status": (operations.get("schedule_run") or {}).get("executed"),
            "automation_generated_at": (operations.get("automation") or {}).get("generated_at"),
        },
        "observability": operations.get("observability") or {},
        "checks": checks,
        "all_required_artifacts_present": all(
            item["exists"] for item in artifacts.values() if item["required"]
        ),
        "artifacts": artifacts,
        "log_messages": [entry.get("message") for entry in logs],
    }


def build_readyz_report(artifact_root: str = "artifacts", run_id: Optional[str] = None) -> ReadinessReport:
    artifact_root_path = Path(artifact_root)
    if not artifact_root_path.exists():
        return ReadinessReport(
            ok=False,
            status="not_ready",
            artifact_root=artifact_root,
            checks={"artifact_root_present": False, "latest_run_present": False},
            reasons=["The artifact root does not exist yet."],
        )

    try:
        run_report = build_run_verification_report(artifact_root, run_id)
    except ValueError as exc:
        return ReadinessReport(
            ok=False,
            status="not_ready",
            artifact_root=artifact_root,
            checks={"artifact_root_present": True, "latest_run_present": False},
            reasons=[str(exc)],
        )

    checks = {
        "artifact_root_present": True,
        "latest_run_present": True,
        "verify_gate_ok": bool(run_report["checks"]["operations_verify_ok"]),
        "required_artifacts_present": bool(run_report["all_required_artifacts_present"]),
    }
    policy_checks = _evaluate_operational_policy(run_report)
    ok = all(checks.values())
    status = "ready" if ok else "degraded"
    reasons = (
        ["The latest persisted run passed the verify gate and all required artifacts are present."]
        if ok
        else ["The system is reachable, but the latest run is missing some operational guarantees."]
    )
    return ReadinessReport(
        ok=ok,
        status=status,
        artifact_root=artifact_root,
        latest_run_id=run_report["run_id"],
        checks={**checks, **policy_checks},
        reasons=reasons,
        details={
            "status": run_report["status"],
            "decision": run_report["decision"],
            "adaptive": run_report["adaptive"],
            "operations": run_report["operations"],
        },
    )


def build_release_gate_report(artifact_root: str = "artifacts", run_id: Optional[str] = None) -> ReleaseGateReport:
    artifact_root_path = Path(artifact_root)
    if not artifact_root_path.exists():
        return ReleaseGateReport(
            ok=False,
            status="blocked",
            artifact_root=artifact_root,
            checks={"artifact_root_present": False, "latest_run_present": False},
            reasons=["The artifact root does not exist yet."],
        )

    try:
        run_report = build_run_verification_report(artifact_root, run_id)
    except ValueError as exc:
        return ReleaseGateReport(
            ok=False,
            status="blocked",
            artifact_root=artifact_root,
            checks={"artifact_root_present": True, "latest_run_present": False},
            reasons=[str(exc)],
        )

    operations = run_report["operations"]
    adaptive = run_report.get("adaptive") or {}
    policy_checks = _evaluate_operational_policy(run_report)
    adaptive_checks = _evaluate_adaptive_release_policy(run_report)
    promotion_lifecycle = str(adaptive.get("promotion_lifecycle_status") or "").strip()
    promotion_mode = str(adaptive.get("promotion_mode") or "").strip()
    checks = {
        "artifact_root_present": True,
        "latest_run_present": True,
        **policy_checks,
        **adaptive_checks,
        "soak_gate_passed": operations.get("soak_status") == "passed",
        "release_summary_ready": operations.get("release_stage") == "release_ready",
    }
    non_soak_checks_ok = all(
        value for name, value in checks.items() if name not in {"soak_gate_passed", "release_summary_ready"}
    )
    if all(checks.values()) and promotion_mode == "promotion_ready" and promotion_lifecycle == "prepared":
        status = "pending_apply"
        ok = False
        reasons = ["The run passed release validation, but the prepared adaptive package has not been applied yet."]
    elif all(checks.values()) and promotion_lifecycle == "applied":
        status = "release_applied"
        ok = True
        reasons = ["The latest run passed release validation and its adaptive package was marked as applied."]
    elif all(checks.values()):
        status = "release_ready"
        ok = True
        reasons = ["The latest run passed verify, soak, and release summary gates."]
    elif non_soak_checks_ok and not checks["soak_gate_passed"]:
        status = "pending_soak"
        ok = False
        reasons = ["Core verification is green, but soak validation has not been completed yet."]
    else:
        status = "blocked"
        ok = False
        reasons = ["The latest run is not eligible for release because verify, adaptive promotion, or artifacts are not green."]

    return ReleaseGateReport(
        ok=ok,
        status=status,
        artifact_root=artifact_root,
        latest_run_id=run_report["run_id"],
        checks=checks,
        reasons=reasons,
        details={
            "status": run_report["status"],
            "decision": run_report["decision"],
            "observability": run_report["observability"],
            "operations": operations,
        },
    )


def build_release_board(artifact_root: str = "artifacts", limit: int = 10) -> ReleaseBoard:
    orchestrator = PipelineOrchestrator(artifact_root=artifact_root)
    summaries = orchestrator.list_runs()
    entries: list[ReleaseBoardEntry] = []
    for summary in reversed(summaries[-max(0, limit):]):
        run_id = summary.get("run_id")
        if not run_id:
            continue
        try:
            run_report = build_run_verification_report(artifact_root, run_id)
            readyz = build_readyz_report(artifact_root, run_id)
            release_gate = build_release_gate_report(artifact_root, run_id)
        except ValueError:
            continue
        operations = run_report.get("operations") or {}
        observability = run_report.get("observability") or {}
        entries.append(
            ReleaseBoardEntry(
                run_id=run_id,
                status=run_report.get("status") or "",
                tickers=list(run_report.get("tickers") or []),
                final_decision=run_report.get("decision", {}).get("final"),
                confidence=run_report.get("decision", {}).get("confidence"),
                run_mode=operations.get("run_mode"),
                decision_path=observability.get("decision_path"),
                verify_ok=bool(operations.get("verify_ok")),
                readiness_status=readyz.status,
                release_status=release_gate.status,
                release_stage=operations.get("release_stage"),
                soak_status=operations.get("soak_status"),
                manual_review_status=(run_report.get("adaptive") or {}).get("manual_review_status"),
                promotion_application_status=(run_report.get("adaptive") or {}).get("promotion_application_status"),
                promotion_application_version=(run_report.get("adaptive") or {}).get("promotion_application_version"),
                promotion_lifecycle_status=(run_report.get("adaptive") or {}).get("promotion_lifecycle_status"),
                run_duration_ms=_safe_int(observability.get("run_duration_ms")),
            )
        )
    latest_run_id = entries[0].run_id if entries else None
    return ReleaseBoard(
        artifact_root=artifact_root,
        latest_run_id=latest_run_id,
        total_runs=len(entries),
        entries=entries,
    )
