from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from assistant.runtime import AssistantRuntime
from adaptive.apply import persist_promotion_application
from adaptive.promote import persist_promotion_execution
from adaptive.review import persist_promotion_review
from ops.automation import persist_operations_automation
from ops.dashboard import build_operations_dashboard, persist_operations_dashboard, render_operations_dashboard_html
from ops.health import build_readyz_report, build_release_board, build_release_gate_report, build_run_verification_report
from ops.job_runner import persist_operations_job
from ops.refresh import persist_operations_refresh
from ops.scheduler import persist_operations_schedule, run_due_operations_schedule
from ops.soak import persist_soak_gate
from pipeline.orchestrator import PipelineOrchestrator
from schemas.pipeline import PipelineRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="IAC MoneyLab Quant Pipeline", version="1.0.0")


def _orchestrator(artifact_root: str = "artifacts") -> PipelineOrchestrator:
    return PipelineOrchestrator(artifact_root=artifact_root)


def _assistant_runtime(artifact_root: str = "artifacts", session_id: str = "default") -> AssistantRuntime:
    return AssistantRuntime(artifact_root=artifact_root, session_id=session_id)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readyz(artifact_root: str = "artifacts", run_id: str | None = None) -> Dict[str, Any]:
    return build_readyz_report(artifact_root, run_id).model_dump()


@app.get("/releasez")
def releasez(artifact_root: str = "artifacts", run_id: str | None = None) -> Dict[str, Any]:
    return build_release_gate_report(artifact_root, run_id).model_dump()


@app.get("/release-board")
def release_board(artifact_root: str = "artifacts", limit: int = 10) -> Dict[str, Any]:
    return build_release_board(artifact_root, limit).model_dump()


@app.get("/ops/dashboard")
def ops_dashboard(artifact_root: str = "artifacts", run_id: str | None = None, limit: int = 10) -> Dict[str, Any]:
    return build_operations_dashboard(artifact_root, run_id, limit=limit).model_dump(mode="json")


@app.get("/ops/dashboard/html", response_class=HTMLResponse)
def ops_dashboard_html(artifact_root: str = "artifacts", run_id: str | None = None, limit: int = 10) -> HTMLResponse:
    dashboard = build_operations_dashboard(artifact_root, run_id, limit=limit)
    return HTMLResponse(render_operations_dashboard_html(dashboard))


@app.post("/pipeline/runs/{run_id}/dashboard")
def persist_run_dashboard(run_id: str, artifact_root: str = "artifacts", limit: int = 10) -> Dict[str, Any]:
    try:
        return persist_operations_dashboard(artifact_root, run_id, limit=limit).model_dump(mode="json")
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/pipeline/runs/{run_id}/adaptive/review")
def review_adaptive_candidate(
    run_id: str,
    reviewer: str,
    decision: str,
    notes: str = "",
    artifact_root: str = "artifacts",
) -> Dict[str, Any]:
    try:
        return persist_promotion_review(
            artifact_root=artifact_root,
            run_id=run_id,
            reviewer=reviewer,
            decision=decision,
            notes=notes,
        ).model_dump(mode="json")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/pipeline/runs/{run_id}/adaptive/apply")
def apply_adaptive_candidate(
    run_id: str,
    operator: str,
    notes: str = "",
    artifact_root: str = "artifacts",
) -> Dict[str, Any]:
    try:
        return persist_promotion_application(
            artifact_root=artifact_root,
            run_id=run_id,
            operator=operator,
            notes=notes,
        ).model_dump(mode="json")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/pipeline/runs/{run_id}/adaptive/promote")
def promote_adaptive_candidate(
    run_id: str,
    operator: str,
    notes: str = "",
    artifact_root: str = "artifacts",
) -> Dict[str, Any]:
    try:
        return persist_promotion_execution(
            artifact_root=artifact_root,
            run_id=run_id,
            operator=operator,
            notes=notes,
        ).model_dump(mode="json")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/pipeline/runs/{run_id}/ops/refresh")
def refresh_run_operations(
    run_id: str,
    artifact_root: str = "artifacts",
    limit: int = 10,
    required_hours: int = 72,
    include_soak: bool = True,
) -> Dict[str, Any]:
    try:
        return persist_operations_refresh(
            artifact_root=artifact_root,
            run_id=run_id,
            limit=limit,
            required_hours=required_hours,
            include_soak=include_soak,
        ).model_dump(mode="json")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/pipeline/runs/{run_id}/ops/run")
def run_operations_job(
    run_id: str,
    artifact_root: str = "artifacts",
    cycles: int = 1,
    interval_seconds: int = 0,
    limit: int = 10,
    required_hours: int = 72,
    include_soak: bool = True,
) -> Dict[str, Any]:
    try:
        return persist_operations_job(
            artifact_root=artifact_root,
            run_id=run_id,
            cycles=cycles,
            interval_seconds=interval_seconds,
            limit=limit,
            required_hours=required_hours,
            include_soak=include_soak,
        ).model_dump(mode="json")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/pipeline/runs/{run_id}/ops/schedule")
def schedule_run_operations(
    run_id: str,
    artifact_root: str = "artifacts",
    interval_seconds: int = 300,
    limit: int = 10,
    required_hours: int = 72,
    include_soak: bool = True,
    enabled: bool = True,
    start_immediately: bool = False,
) -> Dict[str, Any]:
    try:
        return persist_operations_schedule(
            artifact_root=artifact_root,
            run_id=run_id,
            interval_seconds=interval_seconds,
            limit=limit,
            required_hours=required_hours,
            include_soak=include_soak,
            enabled=enabled,
            start_immediately=start_immediately,
        ).model_dump(mode="json")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/pipeline/runs/{run_id}/ops/schedule/run")
def run_due_scheduled_operations(
    run_id: str,
    artifact_root: str = "artifacts",
    force: bool = False,
) -> Dict[str, Any]:
    try:
        return run_due_operations_schedule(
            artifact_root=artifact_root,
            run_id=run_id,
            force=force,
        ).model_dump(mode="json")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/pipeline/runs/{run_id}/ops/automation")
def persist_ops_automation(
    run_id: str,
    artifact_root: str = "artifacts",
    python_bin: str = ".venv/bin/python",
) -> Dict[str, Any]:
    try:
        return persist_operations_automation(
            artifact_root=artifact_root,
            run_id=run_id,
            python_bin=python_bin,
        ).model_dump(mode="json")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/pipeline/runs/{run_id}/soak")
def run_soak_gate(run_id: str, artifact_root: str = "artifacts", required_hours: int = 72, limit: int = 50) -> Dict[str, Any]:
    try:
        return persist_soak_gate(
            artifact_root=artifact_root,
            run_id=run_id,
            required_hours=required_hours,
            limit=limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/pipeline/run")
def run_pipeline(request: PipelineRequest) -> Dict[str, Any]:
    orchestrator = _orchestrator(request.artifact_root)
    try:
        result = orchestrator.run(request)
        return result.model_dump()
    except Exception as exc:
        logger.exception("Pipeline run failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/pipeline/runs")
def list_runs(artifact_root: str = "artifacts") -> List[Dict[str, Any]]:
    orchestrator = _orchestrator(artifact_root)
    return orchestrator.list_runs()


@app.get("/pipeline/runs/{run_id}")
def get_run(run_id: str, artifact_root: str = "artifacts") -> Dict[str, Any]:
    orchestrator = _orchestrator(artifact_root)
    run = orchestrator.load_run(run_id)
    if not run["manifest"]:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.get("/pipeline/runs/{run_id}/results")
def get_results(run_id: str, artifact_root: str = "artifacts") -> Dict[str, Any]:
    orchestrator = _orchestrator(artifact_root)
    run = orchestrator.load_run(run_id)
    if not run["manifest"]:
        raise HTTPException(status_code=404, detail="Run not found")
    result = run.get("result")
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    return result


@app.get("/pipeline/runs/{run_id}/verify")
def verify_run(run_id: str, artifact_root: str = "artifacts") -> Dict[str, Any]:
    try:
        return build_run_verification_report(artifact_root, run_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/pipeline/runs/{run_id}/observability")
def run_observability(run_id: str, artifact_root: str = "artifacts") -> Dict[str, Any]:
    orchestrator = _orchestrator(artifact_root)
    run = orchestrator.load_run(run_id)
    if not run["manifest"]:
        raise HTTPException(status_code=404, detail="Run not found")
    operations = (run.get("summary") or {}).get("operations") or (run.get("result") or {}).get("operations") or (run.get("manifest") or {}).get("operations") or {}
    observability = operations.get("observability")
    if not observability:
        raise HTTPException(status_code=404, detail="Observability report not found")
    return observability


@app.get("/pipeline/runs/{run_id}/logs")
def get_logs(run_id: str, artifact_root: str = "artifacts") -> List[Dict[str, Any]]:
    orchestrator = _orchestrator(artifact_root)
    run = orchestrator.load_run(run_id)
    if not run["manifest"]:
        raise HTTPException(status_code=404, detail="Run not found")
    return run.get("logs", [])


@app.get("/models")
def list_models(artifact_root: str = "artifacts") -> List[Dict[str, Any]]:
    orchestrator = _orchestrator(artifact_root)
    return orchestrator.list_models()


@app.get("/models/{model_name}")
def get_model(model_name: str, artifact_root: str = "artifacts") -> Dict[str, Any]:
    orchestrator = _orchestrator(artifact_root)
    matches = [model for model in orchestrator.list_models() if model["name"] == model_name]
    if not matches:
        raise HTTPException(status_code=404, detail="Model not found")
    return matches[-1]


@app.post("/assistant/chat")
def assistant_chat(payload: Dict[str, Any]) -> Dict[str, Any]:
    message = str(payload.get("message") or payload.get("text") or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    artifact_root = str(payload.get("artifact_root") or "artifacts")
    session_id = str(payload.get("session_id") or "default")
    runtime = _assistant_runtime(artifact_root=artifact_root, session_id=session_id)

    try:
        answer = runtime.ask(message)
        return {
            "answer": answer,
            "state": runtime.get_state().to_dict(),
        }
    except Exception as exc:
        logger.exception("Assistant chat failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/assistant/state")
def assistant_state(session_id: str = "default", artifact_root: str = "artifacts") -> Dict[str, Any]:
    runtime = _assistant_runtime(artifact_root=artifact_root, session_id=session_id)
    return runtime.get_state().to_dict()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
