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

from adaptive.promote import persist_promotion_execution


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mark a prepared adaptive promotion package as applied")
    parser.add_argument("--artifact-root", default="artifacts", help="Artifact root directory")
    parser.add_argument("--run-id", help="Run id to promote. Defaults to the latest run.")
    parser.add_argument("--operator", required=True, help="Operator applying the promotion package")
    parser.add_argument("--notes", default="", help="Optional operator notes")
    parser.add_argument("--json", action="store_true", help="Print the promotion execution artifact as JSON")
    return parser


def _render_text(payload: dict) -> str:
    lines = [
        f"Adaptive promotion: run={payload.get('run_id')} status={payload.get('status')}",
        f"Operator: {payload.get('applied_by')}",
        f"Applied at: {payload.get('applied_at')}",
        f"Config version: {payload.get('config_version')}",
    ]
    if payload.get("notes"):
        lines.append(f"Notes: {payload.get('notes')}")
    for reason in payload.get("reasons") or []:
        lines.append(f"- reason: {reason}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = persist_promotion_execution(
            artifact_root=args.artifact_root,
            run_id=args.run_id,
            operator=args.operator,
            notes=args.notes,
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
