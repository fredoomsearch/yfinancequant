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

from ops.soak import persist_soak_gate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persist a soak gate and release summary from recent runs")
    parser.add_argument("--artifact-root", default="artifacts", help="Artifact root directory")
    parser.add_argument("--run-id", help="Run id to update. Defaults to the latest run.")
    parser.add_argument("--required-hours", type=int, default=72, help="Required observed soak window in hours")
    parser.add_argument("--limit", type=int, default=50, help="Maximum number of recent runs to inspect")
    parser.add_argument("--json", action="store_true", help="Print the persisted soak result as JSON")
    return parser


def _render_text(payload: dict) -> str:
    soak = payload.get("soak_gate") or {}
    release = payload.get("release_summary") or {}
    lines = [
        f"Soak gate for {payload.get('run_id')}",
        f"- status={soak.get('status')} ok={soak.get('ok')} observed_hours={soak.get('observed_hours')}",
        f"- sampled_runs={soak.get('sampled_runs')} ready_runs={soak.get('ready_runs')}",
        f"- release_stage={release.get('release_stage')} release_ok={release.get('ok')}",
    ]
    for reason in soak.get("reasons") or []:
        lines.append(f"- reason: {reason}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = persist_soak_gate(
            artifact_root=args.artifact_root,
            run_id=args.run_id,
            required_hours=args.required_hours,
            limit=args.limit,
        )
    except ValueError as exc:
        parser.exit(1, f"{exc}\n")
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(_render_text(payload))
    return 0 if (payload.get("soak_gate") or {}).get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
