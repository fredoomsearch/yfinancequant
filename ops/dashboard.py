from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Optional

from ops.health import build_readyz_report, build_release_board, build_release_gate_report, build_run_verification_report
from schemas.pipeline import ArtifactRef, OperationsDashboard


def _artifact(path: Path, kind: str) -> ArtifactRef:
    return ArtifactRef(
        name=path.name,
        path=str(path),
        kind=kind,
        size_bytes=path.stat().st_size,
    )


def build_operations_dashboard(
    artifact_root: str = "artifacts",
    run_id: Optional[str] = None,
    *,
    limit: int = 10,
) -> OperationsDashboard:
    readyz = build_readyz_report(artifact_root, run_id)
    latest_run_id = run_id or readyz.latest_run_id
    release_gate = build_release_gate_report(artifact_root, latest_run_id)
    release_board = build_release_board(artifact_root, limit)
    latest_run_verify = {}
    latest_run_observability = {}
    if latest_run_id:
        try:
            run_verify = build_run_verification_report(artifact_root, latest_run_id)
            latest_run_verify = run_verify
            latest_run_observability = run_verify.get("observability") or {}
        except ValueError:
            latest_run_verify = {}
            latest_run_observability = {}
    return OperationsDashboard(
        artifact_root=artifact_root,
        latest_run_id=latest_run_id,
        readyz=readyz,
        release_gate=release_gate,
        release_board=release_board,
        latest_run_verify=latest_run_verify,
        latest_run_observability=latest_run_observability,
    )


def persist_operations_dashboard(
    artifact_root: str = "artifacts",
    run_id: Optional[str] = None,
    *,
    limit: int = 10,
) -> OperationsDashboard:
    dashboard = build_operations_dashboard(artifact_root, run_id, limit=limit)
    if not dashboard.latest_run_id:
        raise ValueError("No completed or persisted runs were found under the artifact root.")

    ops_dir = Path(artifact_root) / "runs" / dashboard.latest_run_id / "ops"
    ops_dir.mkdir(parents=True, exist_ok=True)

    json_path = ops_dir / "operations_dashboard.json"
    html_path = ops_dir / "operations_dashboard.html"

    json_path.write_text(json.dumps(dashboard.model_dump(mode="json"), indent=2))
    html_path.write_text(render_operations_dashboard_html(dashboard))

    dashboard.artifacts = {
        "json": _artifact(json_path, "operations_dashboard"),
        "html": _artifact(html_path, "operations_dashboard_html"),
    }
    json_path.write_text(json.dumps(dashboard.model_dump(mode="json"), indent=2))
    return dashboard


def render_operations_dashboard_html(dashboard: OperationsDashboard) -> str:
    readyz = dashboard.readyz
    release_gate = dashboard.release_gate
    release_board = dashboard.release_board
    observability = dashboard.latest_run_observability or {}
    stage_rows = []
    for stage, status in (observability.get("stage_statuses") or {}).items():
        duration = (observability.get("stage_durations_ms") or {}).get(stage, 0)
        stage_rows.append(
            f"<tr><td>{escape(stage)}</td><td>{escape(str(status))}</td><td>{escape(str(duration))}</td></tr>"
        )
    board_rows = []
    for entry in release_board.entries:
        board_rows.append(
            "<tr>"
            f"<td>{escape(entry.run_id)}</td>"
            f"<td>{escape(entry.status)}</td>"
            f"<td>{escape(', '.join(entry.tickers))}</td>"
            f"<td>{escape(str(entry.final_decision))}</td>"
            f"<td>{escape(str(entry.verify_ok))}</td>"
            f"<td>{escape(entry.readiness_status)}</td>"
            f"<td>{escape(entry.release_status)}</td>"
            f"<td>{escape(str(entry.manual_review_status or 'n/a'))}</td>"
            f"<td>{escape(str(entry.promotion_lifecycle_status or entry.promotion_application_status or 'n/a'))}</td>"
            f"<td>{escape(str(entry.run_duration_ms))}</td>"
            "</tr>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>IAC Ops Dashboard</title>
  <style>
    :root {{
      --bg: #f5f1e8;
      --panel: #fffdf8;
      --ink: #1f2a37;
      --muted: #6b7280;
      --line: #d6d3d1;
      --good: #146c43;
      --warn: #a16207;
      --bad: #b91c1c;
      --accent: #0f766e;
    }}
    body {{
      margin: 0;
      padding: 24px;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #efe7d7 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    h1, h2 {{ margin: 0 0 12px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 16px;
      margin-bottom: 20px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
    }}
    .metric {{
      font-size: 28px;
      font-weight: 700;
      margin-top: 6px;
    }}
    .good {{ color: var(--good); }}
    .warn {{ color: var(--warn); }}
    .bad {{ color: var(--bad); }}
    .muted {{ color: var(--muted); }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border-radius: 14px;
      overflow: hidden;
      border: 1px solid var(--line);
      margin-bottom: 20px;
    }}
    th, td {{
      text-align: left;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }}
    th {{
      background: #e7efe9;
      color: var(--accent);
    }}
    .section-title {{
      margin: 20px 0 10px;
    }}
  </style>
</head>
<body>
  <h1>IAC Operations Dashboard</h1>
  <p class="muted">artifact_root={escape(dashboard.artifact_root)} | latest_run={escape(str(dashboard.latest_run_id or 'n/a'))}</p>

  <div class="grid">
    <div class="card">
      <div class="muted">Readyz</div>
      <div class="metric {'good' if readyz.ok else 'warn'}">{escape(readyz.status)}</div>
      <div class="muted">ok={escape(str(readyz.ok))}</div>
    </div>
    <div class="card">
      <div class="muted">Release Gate</div>
      <div class="metric {'good' if release_gate.ok else 'warn'}">{escape(release_gate.status)}</div>
      <div class="muted">ok={escape(str(release_gate.ok))}</div>
    </div>
    <div class="card">
      <div class="muted">Release Board</div>
      <div class="metric">{escape(str(release_board.total_runs))}</div>
      <div class="muted">runs tracked</div>
    </div>
    <div class="card">
      <div class="muted">Latest Decision</div>
      <div class="metric">{escape(str((dashboard.latest_run_verify.get('decision') or {}).get('final') or 'n/a'))}</div>
      <div class="muted">source={escape(str((dashboard.latest_run_verify.get('decision') or {}).get('source') or 'n/a'))}</div>
      <div class="muted">manual_review={escape(str((dashboard.latest_run_verify.get('adaptive') or {}).get('manual_review_status') or 'n/a'))}</div>
      <div class="muted">promotion={escape(str((dashboard.latest_run_verify.get('adaptive') or {}).get('promotion_lifecycle_status') or 'n/a'))}</div>
      <div class="muted">next_ops_run={escape(str((dashboard.latest_run_verify.get('operations') or {}).get('schedule_next_run_at') or 'n/a'))}</div>
    </div>
  </div>

  <h2 class="section-title">Latest Run Observability</h2>
  <table>
    <thead>
      <tr><th>Stage</th><th>Status</th><th>Duration ms</th></tr>
    </thead>
    <tbody>
      {''.join(stage_rows) or '<tr><td colspan="3">No observability data available.</td></tr>'}
    </tbody>
  </table>

  <h2 class="section-title">Release Board</h2>
  <table>
    <thead>
      <tr>
        <th>Run</th><th>Status</th><th>Tickers</th><th>Decision</th><th>Verify</th><th>Readyz</th><th>Release</th><th>Review</th><th>Promotion</th><th>Duration ms</th>
      </tr>
    </thead>
    <tbody>
      {''.join(board_rows) or '<tr><td colspan="10">No runs available.</td></tr>'}
    </tbody>
  </table>
</body>
</html>"""
