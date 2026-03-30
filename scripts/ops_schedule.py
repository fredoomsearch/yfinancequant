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

from ops.scheduler import persist_operations_schedule, run_due_operations_schedule


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persist or execute a scheduled operations refresh plan")
    subparsers = parser.add_subparsers(dest="command", required=True)

    set_parser = subparsers.add_parser("set", help="Persist a schedule for a run")
    set_parser.add_argument("--artifact-root", default="artifacts", help="Artifact root directory")
    set_parser.add_argument("--run-id", help="Run id to schedule. Defaults to the latest run.")
    set_parser.add_argument("--interval-seconds", type=int, default=300, help="Interval between due executions")
    set_parser.add_argument("--limit", type=int, default=10, help="Maximum number of runs in release board and soak window")
    set_parser.add_argument("--required-hours", type=int, default=72, help="Soak threshold in hours")
    set_parser.add_argument("--skip-soak", action="store_true", help="Do not recompute soak in scheduled runs")
    set_parser.add_argument("--disabled", action="store_true", help="Persist the schedule as disabled")
    set_parser.add_argument("--start-immediately", action="store_true", help="Make the first scheduled run due immediately")
    set_parser.add_argument("--json", action="store_true", help="Print the schedule as JSON")

    run_parser = subparsers.add_parser("run", help="Execute the schedule only if it is due")
    run_parser.add_argument("--artifact-root", default="artifacts", help="Artifact root directory")
    run_parser.add_argument("--run-id", help="Run id to schedule. Defaults to the latest run.")
    run_parser.add_argument("--force", action="store_true", help="Run even if the schedule is not due yet")
    run_parser.add_argument("--json", action="store_true", help="Print the schedule run report as JSON")
    return parser


def _render_schedule(payload: dict) -> str:
    lines = [
        f"Operations schedule: run={payload.get('run_id')}",
        f"Enabled: {payload.get('enabled')}",
        f"Interval seconds: {payload.get('interval_seconds')}",
        f"Next run at: {payload.get('next_run_at') or 'n/a'}",
    ]
    return "\n".join(lines)


def _render_run_report(payload: dict) -> str:
    lines = [
        f"Operations schedule run: run={payload.get('run_id')}",
        f"Due: {payload.get('due')} forced={payload.get('forced')} executed={payload.get('executed')}",
        f"Reason: {payload.get('reason')}",
    ]
    job = payload.get("job") or {}
    if job:
        lines.append(f"Latest release status: {job.get('latest_release_status')}")
        lines.append(f"Latest soak status: {job.get('latest_soak_status')}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "set":
            payload = persist_operations_schedule(
                artifact_root=args.artifact_root,
                run_id=args.run_id,
                interval_seconds=args.interval_seconds,
                limit=args.limit,
                required_hours=args.required_hours,
                include_soak=not args.skip_soak,
                enabled=not args.disabled,
                start_immediately=args.start_immediately,
            ).model_dump(mode="json")
            if args.json:
                print(json.dumps(payload, indent=2))
            else:
                print(_render_schedule(payload))
        else:
            payload = run_due_operations_schedule(
                artifact_root=args.artifact_root,
                run_id=args.run_id,
                force=args.force,
            ).model_dump(mode="json")
            if args.json:
                print(json.dumps(payload, indent=2))
            else:
                print(_render_run_report(payload))
    except ValueError as exc:
        parser.exit(1, f"{exc}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
