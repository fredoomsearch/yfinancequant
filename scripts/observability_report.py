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

from pipeline.orchestrator import PipelineOrchestrator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect runtime observability metrics for a persisted run")
    parser.add_argument("--artifact-root", default="artifacts", help="Artifact root directory")
    parser.add_argument("--run-id", help="Run id to inspect. Defaults to the latest run.")
    parser.add_argument("--json", action="store_true", help="Print the observability report as JSON")
    return parser


def build_observability_report(artifact_root: str = "artifacts", run_id: str | None = None) -> dict:
    orchestrator = PipelineOrchestrator(artifact_root=artifact_root)
    selected_run_id = run_id or orchestrator.latest_run_id()
    if not selected_run_id:
        raise ValueError("No completed or persisted runs were found under the artifact root.")
    bundle = orchestrator.load_run(selected_run_id)
    manifest = bundle.get("manifest") or {}
    if not manifest:
        raise ValueError(f"Run {selected_run_id} was not found under {artifact_root}.")
    summary = bundle.get("summary") or {}
    result = bundle.get("result") or {}
    operations = summary.get("operations") or result.get("operations") or manifest.get("operations") or {}
    observability = operations.get("observability") or {}
    return {
        "run_id": selected_run_id,
        "artifact_root": artifact_root,
        "status": summary.get("status") or manifest.get("status"),
        "observability": observability,
    }


def _render_text(report: dict) -> str:
    observability = report.get("observability") or {}
    lines = [
        f"Observability report: {report.get('run_id')}",
        f"Status: {report.get('status')}",
        f"Run mode: {observability.get('run_mode')}",
        f"Decision path: {observability.get('decision_path')}",
        f"Run duration ms: {observability.get('run_duration_ms')}",
    ]
    for stage, status in (observability.get("stage_statuses") or {}).items():
        duration = (observability.get("stage_durations_ms") or {}).get(stage)
        lines.append(f"- stage {stage}: status={status} duration_ms={duration}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = build_observability_report(args.artifact_root, args.run_id)
    except ValueError as exc:
        parser.exit(1, f"{exc}\n")

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(_render_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
