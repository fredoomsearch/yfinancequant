from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env_loader import load_project_env

load_project_env()

from ops.health import build_run_verification_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify run artifacts, adaptive state, and operations gates")
    parser.add_argument("--artifact-root", default="artifacts", help="Artifact root directory")
    parser.add_argument("--run-id", help="Run id to verify. Defaults to the latest run.")
    parser.add_argument("--json", action="store_true", help="Print the verification report as JSON")
    return parser


build_verification_report = build_run_verification_report


def _render_text(report: Dict[str, Any]) -> str:
    decision = report.get("decision") or {}
    adaptive = report.get("adaptive") or {}
    operations = report.get("operations") or {}
    lines = [
        f"Run verification: {report.get('run_id')}",
        f"Status: {report.get('status')}",
        f"Tickers: {', '.join(report.get('tickers') or []) or 'n/a'}",
        f"Decision: {decision.get('final')} ({decision.get('confidence')}) source={decision.get('source')}",
        (
            "Adaptive: "
            f"mode={adaptive.get('mode')} drift={adaptive.get('drift_level')} "
            f"validation={adaptive.get('validation_status')} approval={adaptive.get('approval_status')} "
            f"promotion={adaptive.get('promotion_mode')} manual_review={adaptive.get('manual_review_status')} "
            f"application={adaptive.get('promotion_application_status')} "
            f"execution={adaptive.get('promotion_execution_status')} "
            f"lifecycle={adaptive.get('promotion_lifecycle_status')}"
        ),
        (
            "Operations: "
            f"run_mode={operations.get('run_mode')} verify_ok={operations.get('verify_ok')} "
            f"soak={operations.get('soak_status')} release={operations.get('release_stage')} "
            f"schedule_enabled={operations.get('schedule_enabled')} next_run={operations.get('schedule_next_run_at')} "
            f"automation_generated_at={operations.get('automation_generated_at')}"
        ),
        f"Required artifacts present: {report.get('all_required_artifacts_present')}",
    ]
    for name, artifact in report.get("artifacts", {}).items():
        marker = "ok" if artifact.get("exists") else "missing"
        suffix = "" if artifact.get("required", True) else " (optional)"
        lines.append(f"- {name}: {marker}{suffix}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = build_verification_report(args.artifact_root, args.run_id)
    except ValueError as exc:
        parser.exit(1, f"{exc}\n")

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(_render_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
