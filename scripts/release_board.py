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

from ops.health import build_release_board


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize recent runs in a release board")
    parser.add_argument("--artifact-root", default="artifacts", help="Artifact root directory")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of runs to include")
    parser.add_argument("--json", action="store_true", help="Print the release board as JSON")
    return parser


def _render_text(board: dict) -> str:
    lines = [
        f"Release board: total_runs={board.get('total_runs')} latest_run={board.get('latest_run_id') or 'n/a'}",
    ]
    for entry in board.get("entries") or []:
        lines.append(
            f"- {entry.get('run_id')}: status={entry.get('status')} "
            f"decision={entry.get('final_decision')} verify_ok={entry.get('verify_ok')} "
            f"readyz={entry.get('readiness_status')} release={entry.get('release_status')} "
            f"review={entry.get('manual_review_status')} promotion={entry.get('promotion_lifecycle_status') or entry.get('promotion_application_status')} "
            f"duration_ms={entry.get('run_duration_ms')}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    board = build_release_board(args.artifact_root, args.limit).model_dump()
    if args.json:
        print(json.dumps(board, indent=2))
    else:
        print(_render_text(board))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
