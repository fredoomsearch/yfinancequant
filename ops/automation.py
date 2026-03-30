from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pipeline.orchestrator import PipelineOrchestrator
from schemas.pipeline import ArtifactRef, OperationsAutomationBundle


def _artifact(path: Path, kind: str) -> ArtifactRef:
    return ArtifactRef(
        name=path.name,
        path=str(path),
        kind=kind,
        size_bytes=path.stat().st_size,
    )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def persist_operations_automation(
    artifact_root: str = "artifacts",
    run_id: Optional[str] = None,
    *,
    python_bin: str = ".venv/bin/python",
) -> OperationsAutomationBundle:
    orchestrator = PipelineOrchestrator(artifact_root=artifact_root)
    target_run_id = run_id or orchestrator.latest_run_id()
    if not target_run_id:
        raise ValueError("No completed or persisted runs were found under the artifact root.")
    bundle = orchestrator.load_run(target_run_id)
    if not bundle.get("manifest"):
        raise ValueError(f"Run {target_run_id} was not found under {artifact_root}.")

    ops_dir = Path(artifact_root) / "runs" / target_run_id / "ops"
    schedule_path = ops_dir / "schedule.json"
    if not schedule_path.exists():
        raise ValueError(f"Run {target_run_id} does not contain an operations schedule.")

    generated_at = _utc_now_iso()
    root_dir = Path(artifact_root).resolve().parent
    command = (
        f"cd {root_dir} && {python_bin} scripts/ops_schedule.py run "
        f"--artifact-root {artifact_root} --run-id {target_run_id}"
    )
    timer_name = f"iactest-ops-{target_run_id}"

    shell_path = ops_dir / "run_ops_schedule.sh"
    shell_path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + command + "\n")
    shell_path.chmod(0o755)

    cron_path = ops_dir / "ops_schedule.crontab"
    cron_path.write_text(f"* * * * * {command}\n")

    service_path = ops_dir / f"{timer_name}.service"
    service_path.write_text(
        "\n".join(
            [
                "[Unit]",
                f"Description=IAC ops schedule runner for {target_run_id}",
                "",
                "[Service]",
                "Type=oneshot",
                f"WorkingDirectory={root_dir}",
                f"ExecStart={root_dir / python_bin} scripts/ops_schedule.py run --artifact-root {artifact_root} --run-id {target_run_id}",
                "",
            ]
        )
        + "\n"
    )

    timer_path = ops_dir / f"{timer_name}.timer"
    timer_path.write_text(
        "\n".join(
            [
                "[Unit]",
                f"Description=IAC ops schedule timer for {target_run_id}",
                "",
                "[Timer]",
                "OnCalendar=*-*-* *:*:00",
                "Persistent=true",
                f"Unit={timer_name}.service",
                "",
                "[Install]",
                "WantedBy=timers.target",
                "",
            ]
        )
    )

    shell_artifact = _artifact(shell_path, "operations_schedule_shell")
    cron_artifact = _artifact(cron_path, "operations_schedule_cron")
    service_artifact = _artifact(service_path, "operations_schedule_service")
    timer_artifact = _artifact(timer_path, "operations_schedule_timer")

    payload = {
        "artifact_root": artifact_root,
        "run_id": target_run_id,
        "generated_at": generated_at,
        "command": command,
        "cron_expression": "* * * * *",
        "shell_artifact": shell_artifact.model_dump(mode="json"),
        "cron_artifact": cron_artifact.model_dump(mode="json"),
        "systemd_service_artifact": service_artifact.model_dump(mode="json"),
        "systemd_timer_artifact": timer_artifact.model_dump(mode="json"),
    }
    bundle_path = ops_dir / "automation_bundle.json"
    bundle_path.write_text(json.dumps(payload, indent=2))
    bundle_artifact = _artifact(bundle_path, "operations_automation_bundle")

    return OperationsAutomationBundle(
        artifact_root=artifact_root,
        run_id=target_run_id,
        generated_at=generated_at,
        command=command,
        cron_expression="* * * * *",
        shell_artifact=shell_artifact,
        cron_artifact=cron_artifact,
        systemd_service_artifact=service_artifact,
        systemd_timer_artifact=timer_artifact,
        artifact=bundle_artifact,
    )
