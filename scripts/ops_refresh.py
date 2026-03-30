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

from ops.refresh import persist_operations_refresh


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persist a full operational refresh bundle for a run")
    parser.add_argument("--artifact-root", default="artifacts", help="Artifact root directory")
    parser.add_argument("--run-id", help="Run id to refresh. Defaults to the latest run.")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of runs in release board and soak window")
    parser.add_argument("--required-hours", type=int, default=72, help="Soak threshold in hours")
    parser.add_argument("--skip-soak", action="store_true", help="Do not recompute soak during refresh")
    parser.add_argument("--json", action="store_true", help="Print the refresh report as JSON")
    return parser


def _render_text(report: dict) -> str:
    lines = [
        f"Operations refresh: run={report.get('run_id')}",
        f"Readyz: {((report.get('readyz') or {}).get('status'))}",
        f"Release gate: {((report.get('release_gate') or {}).get('status'))}",
        f"Release board runs: {((report.get('release_board') or {}).get('total_runs'))}",
        f"Soak status: {((report.get('soak') or {}).get('soak_gate') or {}).get('status', 'skipped')}",
    ]
    for name, artifact in (report.get("artifacts") or {}).items():
        lines.append(f"- {name}: {artifact.get('path')}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = persist_operations_refresh(
            artifact_root=args.artifact_root,
            run_id=args.run_id,
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
