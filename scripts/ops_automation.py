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

from ops.automation import persist_operations_automation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize cron/systemd automation artifacts for an ops schedule")
    parser.add_argument("--artifact-root", default="artifacts", help="Artifact root directory")
    parser.add_argument("--run-id", help="Run id to automate. Defaults to the latest run.")
    parser.add_argument("--python-bin", default=".venv/bin/python", help="Python interpreter used by generated artifacts")
    parser.add_argument("--json", action="store_true", help="Print the automation bundle as JSON")
    return parser


def _render_text(payload: dict) -> str:
    lines = [
        f"Operations automation: run={payload.get('run_id')}",
        f"Generated at: {payload.get('generated_at')}",
        f"Command: {payload.get('command')}",
        f"Cron: {payload.get('cron_expression')}",
    ]
    for name in ("shell_artifact", "cron_artifact", "systemd_service_artifact", "systemd_timer_artifact", "artifact"):
        artifact = payload.get(name) or {}
        if artifact.get("path"):
            lines.append(f"- {name}: {artifact.get('path')}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = persist_operations_automation(
            artifact_root=args.artifact_root,
            run_id=args.run_id,
            python_bin=args.python_bin,
        ).model_dump(mode="json")
    except ValueError as exc:
        parser.exit(1, f"{exc}\n")

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(_render_text(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
