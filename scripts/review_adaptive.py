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

from adaptive.review import persist_promotion_review


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persist a manual promotion review for an adaptive candidate")
    parser.add_argument("--artifact-root", default="artifacts", help="Artifact root directory")
    parser.add_argument("--run-id", help="Run id to review. Defaults to the latest run.")
    parser.add_argument("--reviewer", required=True, help="Reviewer name or identifier")
    parser.add_argument("--decision", required=True, choices=["approve", "reject"], help="Manual promotion decision")
    parser.add_argument("--notes", default="", help="Optional reviewer notes")
    parser.add_argument("--json", action="store_true", help="Print the persisted review as JSON")
    return parser


def _render_text(review: dict) -> str:
    lines = [
        f"Adaptive review: run={review.get('run_id')} status={review.get('status')} approved={review.get('approved')}",
        f"Reviewer: {review.get('reviewer')}",
        f"Reviewed at: {review.get('reviewed_at')}",
        f"Policy status: {review.get('policy_status')}",
        f"Promotion mode: {review.get('promotion_mode')}",
    ]
    if review.get("notes"):
        lines.append(f"Notes: {review.get('notes')}")
    for reason in review.get("reasons") or []:
        lines.append(f"- reason: {reason}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        review = persist_promotion_review(
            artifact_root=args.artifact_root,
            run_id=args.run_id,
            reviewer=args.reviewer,
            decision=args.decision,
            notes=args.notes,
        ).model_dump(mode="json")
    except ValueError as exc:
        parser.exit(1, f"{exc}\n")

    if args.json:
        print(json.dumps(review, indent=2))
    else:
        print(_render_text(review))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
