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

from ops.job_runner import persist_operations_job


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a periodic operations refresh job for a persisted run")
    parser.add_argument("--artifact-root", default="artifacts", help="Artifact root directory")
    parser.add_argument("--run-id", help="Run id to operate on. Defaults to the latest run.")
    parser.add_argument("--cycles", type=int, default=1, help="Number of refresh cycles to execute")
    parser.add_argument("--interval-seconds", type=int, default=0, help="Sleep interval between cycles")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of runs in release board and soak window")
    parser.add_argument("--required-hours", type=int, default=72, help="Soak threshold in hours")
    parser.add_argument("--skip-soak", action="store_true", help="Do not recompute soak during the job")
    parser.add_argument("--json", action="store_true", help="Print the job report as JSON")
    return parser


def _render_text(report: dict) -> str:
    lines = [
        f"Operations job: run={report.get('run_id')}",
        f"Cycles: {report.get('cycles_completed')}/{report.get('cycles_requested')}",
        f"Latest release status: {report.get('latest_release_status')}",
        f"Latest soak status: {report.get('latest_soak_status')}",
    ]
    for cycle in report.get("cycles") or []:
        lines.append(
            f"- cycle {cycle.get('cycle_index')}: readyz={cycle.get('readyz_status')} "
            f"release={cycle.get('release_status')} soak={cycle.get('soak_status')}"
        )
    artifact = report.get("artifact") or {}
    if artifact.get("path"):
        lines.append(f"- artifact: {artifact.get('path')}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = persist_operations_job(
            artifact_root=args.artifact_root,
            run_id=args.run_id,
            cycles=args.cycles,
            interval_seconds=args.interval_seconds,
            limit=args.limit,
            required_hours=args.required_hours,
            include_soak=not args.skip_soak,
        ).model_dump(mode="json")
    except ValueError as exc:
        parser.exit(1, f"{exc}\n")

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(_render_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
