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

from ops.health import build_release_gate_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate release readiness from persisted run artifacts")
    parser.add_argument("--artifact-root", default="artifacts", help="Artifact root directory")
    parser.add_argument("--run-id", help="Run id to inspect. Defaults to the latest run.")
    parser.add_argument("--json", action="store_true", help="Print the release-gate report as JSON")
    return parser


def _render_text(report: dict) -> str:
    lines = [
        f"Release gate: ok={report.get('ok')} status={report.get('status')}",
        f"Artifact root: {report.get('artifact_root')}",
        f"Latest run: {report.get('latest_run_id') or 'n/a'}",
    ]
    for name, value in (report.get("checks") or {}).items():
        lines.append(f"- {name}: {value}")
    for reason in report.get("reasons") or []:
        lines.append(f"- reason: {reason}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    report = build_release_gate_report(args.artifact_root, args.run_id).model_dump()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(_render_text(report))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
