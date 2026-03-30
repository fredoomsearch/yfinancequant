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

from ops.dashboard import build_operations_dashboard, persist_operations_dashboard, render_operations_dashboard_html


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render an operations dashboard from persisted artifacts")
    parser.add_argument("--artifact-root", default="artifacts", help="Artifact root directory")
    parser.add_argument("--run-id", help="Run id to focus. Defaults to the latest run.")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of runs to include in the board")
    parser.add_argument("--json", action="store_true", help="Print dashboard data as JSON")
    parser.add_argument("--html-out", help="Optional output path for rendered HTML")
    parser.add_argument("--persist", action="store_true", help="Persist dashboard JSON and HTML under the run ops directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.persist:
        dashboard = persist_operations_dashboard(args.artifact_root, args.run_id, limit=args.limit)
    else:
        dashboard = build_operations_dashboard(args.artifact_root, args.run_id, limit=args.limit)
    if args.json:
        print(json.dumps(dashboard.model_dump(mode="json"), indent=2))
    else:
        html = render_operations_dashboard_html(dashboard)
        if args.html_out:
            path = Path(args.html_out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(html)
            print(path)
        else:
            print(html)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
