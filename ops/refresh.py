from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from ops.dashboard import persist_operations_dashboard
from ops.health import build_readyz_report, build_release_board, build_release_gate_report
from ops.soak import persist_soak_gate
from schemas.pipeline import ArtifactRef, OperationsRefreshReport


def _artifact(path: Path, kind: str) -> ArtifactRef:
    return ArtifactRef(
        name=path.name,
        path=str(path),
        kind=kind,
        size_bytes=path.stat().st_size,
    )


def _write_json(path: Path, payload: Dict) -> ArtifactRef:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return _artifact(path, path.stem)


def persist_operations_refresh(
    artifact_root: str = "artifacts",
    run_id: Optional[str] = None,
    *,
    limit: int = 10,
    required_hours: int = 72,
    include_soak: bool = True,
) -> OperationsRefreshReport:
    dashboard = persist_operations_dashboard(artifact_root, run_id, limit=limit)
    target_run_id = dashboard.latest_run_id
    if not target_run_id:
        raise ValueError("No completed or persisted runs were found under the artifact root.")

    soak_payload: Dict = {}
    if include_soak:
        soak_payload = persist_soak_gate(
            artifact_root=artifact_root,
            run_id=target_run_id,
            required_hours=required_hours,
            limit=max(limit, 1),
        )

    readyz = build_readyz_report(artifact_root, target_run_id)
    release_gate = build_release_gate_report(artifact_root, target_run_id)
    release_board = build_release_board(artifact_root, limit)

    ops_dir = Path(artifact_root) / "runs" / target_run_id / "ops"
    artifacts = {
        "readyz": _write_json(ops_dir / "readyz.json", readyz.model_dump(mode="json")),
        "release_gate": _write_json(ops_dir / "release_gate_report.json", release_gate.model_dump(mode="json")),
        "release_board": _write_json(ops_dir / "release_board.json", release_board.model_dump(mode="json")),
    }
    artifacts.update(dashboard.artifacts)

    return OperationsRefreshReport(
        artifact_root=artifact_root,
        run_id=target_run_id,
        readyz=readyz,
        release_gate=release_gate,
        release_board=release_board,
        dashboard=dashboard,
        soak=soak_payload,
        artifacts=artifacts,
    )
