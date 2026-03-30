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

from assistant.scorecard import build_assistant_scorecard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render the assistant architecture scorecard")
    parser.add_argument("--json", action="store_true", help="Print the scorecard as JSON")
    return parser


def _render_text(report: dict) -> str:
    lines = [
        f"Assistant scorecard generated_at={report.get('generated_at')}",
        (
            "Local-first score: "
            f"runtime_ready={report.get('local_first_runtime_ready_score_pct')}% "
            f"implementation={report.get('local_first_implementation_score_pct')}%"
        ),
        (
            "Hybrid score: "
            f"runtime_ready={report.get('hybrid_runtime_ready_score_pct')}% "
            f"implementation={report.get('hybrid_implementation_score_pct')}%"
        ),
        (
            "Runtime web config: "
            f"configured={((report.get('runtime') or {}).get('web_retriever_configured'))} "
            f"method={((report.get('runtime') or {}).get('web_method'))} "
            f"results_path={((report.get('runtime') or {}).get('web_results_path') or 'n/a')}"
        ),
    ]
    for key, layer in (report.get("layers") or {}).items():
        lines.append(
            (
                f"- {key}: runtime_ready={layer.get('runtime_ready_score_pct')}% "
                f"implementation={layer.get('implementation_score_pct')}% "
                f"checks={layer.get('passed_checks')}/{layer.get('total_checks')}"
            )
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    report = build_assistant_scorecard().to_dict()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(_render_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
