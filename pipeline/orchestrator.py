
from __future__ import annotations

import json
import logging
import re
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from adaptive import (
    AdaptivePolicyEngine,
    AdaptiveSelector,
    AdaptiveValidator,
    DriftDetector,
    FeatureRegistry,
    PromotionGate,
    RetrainingScheduler,
    RuntimeFingerprintBuilder,
    ShadowRunner,
)
from agents import CleaningAgent, ExtractionAgent, ModelingAgent
from ops import OperationsReportBuilder
from providers import GroqAdvisor, GroqBrain, GroqReviewer
from scripts.compare_sources import compare_frames, normalize_binance, normalize_yfinance
from utils.legacy_btc import build_legacy_btc_analysis
from schemas.pipeline import (
    AdaptiveReport,
    AgentLog,
    ArtifactRef,
    ErrorPayload,
    OperationsReport,
    ModelFamily,
    PipelineRequest,
    PipelineResult,
    ReviewerPacket,
    RunManifest,
    RunState,
    SourceComparison,
    StageBrief,
)

logger = logging.getLogger(__name__)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump()
    path.write_text(json.dumps(payload, indent=2, default=str))


def _read_json(path: Path):
    return json.loads(path.read_text())


def _normalize_direction(value: Any, fallback: str) -> str:
    candidate = str(value or "").strip().lower()
    if candidate in {"long", "short", "hold"}:
        return candidate
    return fallback


class PipelineOrchestrator:
    def __init__(
        self,
        artifact_root: str = "artifacts",
        reviewer: Optional[GroqReviewer] = None,
        advisor: Optional[GroqAdvisor] = None,
        brain: Optional[GroqBrain] = None,
        drift_detector: Optional[DriftDetector] = None,
        adaptive_selector: Optional[AdaptiveSelector] = None,
        adaptive_policy: Optional[AdaptivePolicyEngine] = None,
        adaptive_validator: Optional[AdaptiveValidator] = None,
        promotion_gate: Optional[PromotionGate] = None,
        shadow_runner: Optional[ShadowRunner] = None,
        feature_registry: Optional[FeatureRegistry] = None,
        retraining_scheduler: Optional[RetrainingScheduler] = None,
        runtime_fingerprint_builder: Optional[RuntimeFingerprintBuilder] = None,
        operations_builder: Optional[OperationsReportBuilder] = None,
    ) -> None:
        self.artifact_root = Path(artifact_root)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.runs_root = self.artifact_root / "runs"
        self.models_root = self.artifact_root / "models"
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.models_root.mkdir(parents=True, exist_ok=True)
        self.reviewer = reviewer or GroqReviewer()
        self.advisor = advisor or GroqAdvisor()
        self.brain = brain or GroqBrain()
        self.drift_detector = drift_detector or DriftDetector()
        self.adaptive_selector = adaptive_selector or AdaptiveSelector()
        self.adaptive_policy = adaptive_policy or AdaptivePolicyEngine()
        self.adaptive_validator = adaptive_validator or AdaptiveValidator()
        self.promotion_gate = promotion_gate or PromotionGate()
        self.shadow_runner = shadow_runner or ShadowRunner()
        self.feature_registry = feature_registry or FeatureRegistry()
        self.retraining_scheduler = retraining_scheduler or RetrainingScheduler()
        self.runtime_fingerprint_builder = runtime_fingerprint_builder or RuntimeFingerprintBuilder()
        self.operations_builder = operations_builder or OperationsReportBuilder()

    def _resolve_run_mode(
        self,
        manifest: RunManifest,
        legacy_analysis: Optional[Dict[str, Any]],
        brain_used: bool,
    ) -> str:
        comparison_requested = bool(manifest.request.compare_binance)
        legacy_enabled = bool(legacy_analysis and legacy_analysis.get("enabled", False))
        reviewer_enabled = manifest.reviewer_used
        brain_enabled = bool(manifest.request.experimental_groq_brain)
        if comparison_requested and legacy_enabled and brain_enabled:
            return "local_plus_binance_legacy_groq_brain"
        if comparison_requested and brain_enabled:
            return "local_plus_binance_groq_brain"
        if brain_enabled:
            return "local_only_groq_brain"
        if comparison_requested and legacy_enabled and reviewer_enabled:
            return "local_plus_binance_legacy_plus_reviewer"
        if comparison_requested and legacy_enabled:
            return "local_plus_binance_legacy"
        if comparison_requested and reviewer_enabled:
            return "local_plus_binance_plus_reviewer"
        if comparison_requested:
            return "local_plus_binance"
        if reviewer_enabled:
            return "local_plus_reviewer"
        return "local_only"

    def _run_dir(self, run_id: str) -> Path:
        return self.runs_root / run_id

    def _next_run_id(self) -> str:
        pattern = re.compile(r"^run_(\d+)$")
        next_suffix = 1
        for path in self.runs_root.iterdir():
            if not path.is_dir():
                continue
            match = pattern.match(path.name)
            if match:
                next_suffix = max(next_suffix, int(match.group(1)) + 1)
        run_id = f"run_{next_suffix:04d}"
        while (self.runs_root / run_id).exists():
            next_suffix += 1
            run_id = f"run_{next_suffix:04d}"
        return run_id

    def _ensure_layout(self, run_dir: Path) -> None:
        for subdir in ("raw", "cleaned", "models", "logs"):
            (run_dir / subdir).mkdir(parents=True, exist_ok=True)

    def _save_manifest(self, manifest: RunManifest, run_dir: Path) -> None:
        manifest.updated_at = datetime.now(timezone.utc)
        _write_json(run_dir / "manifest.json", manifest)

    def _append_log(
        self,
        manifest: RunManifest,
        run_dir: Path,
        agent: str,
        message: str,
        status: RunState = RunState.running,
        **details,
    ) -> None:
        entry = AgentLog(agent=agent, status=status, message=message, details=details)
        manifest.logs.append(entry)
        self._save_manifest(manifest, run_dir)

    def _finalize(self, manifest: RunManifest, run_dir: Path, result: PipelineResult) -> None:
        _write_json(run_dir / "result.json", result)
        _write_json(run_dir / "logs.json", [log.model_dump() for log in manifest.logs])
        self._save_manifest(manifest, run_dir)

    def _build_source_comparison(self, request: PipelineRequest) -> Optional[SourceComparison]:
        if not request.compare_binance:
            return None
        if not request.comparison_asset or not request.comparison_yfinance_ticker or not request.comparison_binance_symbol:
            return SourceComparison(
                enabled=False,
                source_1="yfinance",
                source_2="binance",
                asset=request.comparison_asset,
                timeframe=request.interval,
                note="Binance comparison requested but comparison inputs were incomplete.",
            )

        yf_frame = normalize_yfinance(
            request.comparison_yfinance_ticker,
            request.start.isoformat(),
            request.end.isoformat(),
            request.interval,
        )
        bn_frame = normalize_binance(
            request.comparison_binance_symbol,
            request.start.isoformat(),
            request.end.isoformat(),
            request.interval,
        )
        comparison = compare_frames(yf_frame, bn_frame)
        return SourceComparison(
            enabled=True,
            source_1="yfinance",
            source_2="binance",
            asset=request.comparison_asset,
            timeframe=request.interval,
            date_range={"start": request.start.isoformat(), "end": request.end.isoformat()},
            coverage=comparison["coverage"],
            missing_fields=comparison["missing_fields"],
            row_counts=comparison["row_counts"],
            close_price_alignment=comparison["close_price_alignment"],
            note=(
                "Optional manual comparison between yfinance and Binance for the same asset and timeframe."
                if request.language == "en"
                else "Comparación manual opcional entre yfinance y Binance para el mismo activo y horizonte temporal."
            ),
        )

    def _build_summary(
        self,
        manifest: RunManifest,
        extraction,
        cleaning,
        modeling,
        reviewer_result,
        run_dir: Path,
        source_comparison: Optional[SourceComparison] = None,
        legacy_analysis: Optional[Dict[str, Any]] = None,
        orchestrator_brief: Optional[StageBrief] = None,
        brain_result: Optional[ReviewerResult] = None,
        deterministic_decision: Optional[str] = None,
        deterministic_confidence: Optional[float] = None,
        brain_used: bool = False,
        decision_source: Optional[str] = None,
        adaptive_report: Optional[AdaptiveReport] = None,
        operations_report: Optional[OperationsReport] = None,
    ) -> Dict[str, Any]:
        raw_columns = list(extraction.raw_data.columns)
        feature_columns = list(cleaning.feature_columns)
        derived_columns = [column for column in feature_columns if column not in raw_columns]
        trained_models = [model.model_name for model in modeling.models]
        raw_rows = extraction.rows
        clean_rows = cleaning.rows_out
        retention_pct = round((clean_rows / raw_rows * 100.0), 2) if raw_rows else 0.0
        es = manifest.request.language == "es"

        if manifest.reviewer_used and reviewer_result:
            selection_strategy = "reviewer_override"
            selection_reason = (
                f"Groq reviewer selected the final decision: {reviewer_result.explanation}"
                if not es
                else f"El revisor de Groq seleccionó la decisión final: {reviewer_result.explanation}"
            )
        elif manifest.request.model_choice != ModelFamily.auto:
            if manifest.request.model_choice == ModelFamily.ensemble:
                selection_strategy = "explicit_ensemble"
                selection_reason = (
                    "The request explicitly asked for the ensemble result."
                    if not es
                    else "La solicitud pidió explícitamente el resultado del ensamble."
                )
            else:
                selection_strategy = f"explicit_{manifest.request.model_choice.value}"
                selection_reason = (
                    f"The request explicitly selected {manifest.request.model_choice.value}."
                    if not es
                    else f"La solicitud seleccionó explícitamente {manifest.request.model_choice.value}."
                )
        elif modeling.disagreement:
            selection_strategy = "majority_vote"
            selection_reason = (
                "The local models disagreed, so the orchestrator used the majority candidate "
                "and kept the decision conservative."
                if not es
                else "Los modelos locales discreparon, así que el orquestador usó la mayoría y mantuvo la decisión conservadora."
            )
        else:
            selection_strategy = "ensemble"
            selection_reason = (
                "The local models were aligned, so the ensemble was used."
                if not es
                else "Los modelos locales estuvieron alineados, así que se usó el ensamble."
            )

        model_metrics = []
        model_details = []
        vote_counts = {"long": 0, "short": 0}
        model_probabilities = []
        for model in modeling.models:
            vote_counts[model.latest_prediction] = vote_counts.get(model.latest_prediction, 0) + 1
            model_probabilities.append(model.latest_probability)
            model_metrics.append(
                {
                    "model": model.model_name,
                    "prediction": model.latest_prediction,
                    "probability": round(model.latest_probability, 6),
                    "confidence": round(model.confidence, 6),
                    "accuracy": round(model.validation_metrics.get("accuracy", 0.0), 6),
                    "roc_auc": round(model.validation_metrics.get("roc_auc", 0.0), 6),
                }
            )
            model_details.append(
                {
                    "model": model.model_name,
                    "prediction": model.latest_prediction,
                    "probability": round(model.latest_probability, 6),
                    "confidence": round(model.confidence, 6),
                    "summary": (
                        (
                        f"{model.model_name} predicted {model.latest_prediction} "
                        f"with {model.latest_probability:.2f} probability and {model.confidence:.2f} confidence."
                        if not es
                        else f"{model.model_name} predijo {model.latest_prediction} "
                        f"con {model.latest_probability:.2f} de probabilidad y {model.confidence:.2f} de confianza."
                    )
                    ),
                }
            )

        probability_spread = round((max(model_probabilities) - min(model_probabilities)), 6) if model_probabilities else 0.0
        extraction_health = "good" if not extraction.missing_columns and raw_rows > 0 else "needs_review"
        modeling_health = "stable" if not modeling.disagreement and manifest.confidence and manifest.confidence >= manifest.request.confidence_threshold else "mixed"
        selection_summary = (
            f"{selection_strategy} chose {manifest.decision} because {selection_reason.lower()}"
            if not es
            else f"{selection_strategy} eligió {manifest.decision} porque {selection_reason.lower()}"
        )
        if modeling.disagreement:
            disagreement_reason = (
                f"The local model votes split as long={vote_counts['long']} and short={vote_counts['short']}; "
                f"probability spread was {probability_spread:.4f}."
                if not es
                else f"Las votaciones locales se dividieron en long={vote_counts['long']} y short={vote_counts['short']}; "
                f"la diferencia de probabilidad fue {probability_spread:.4f}."
            )
        else:
            disagreement_reason = (
                f"All three local models agreed on {modeling.ensemble_prediction}; "
                f"probability spread was {probability_spread:.4f}."
                if not es
                else f"Los tres modelos locales acordaron {modeling.ensemble_prediction}; "
                f"la diferencia de probabilidad fue {probability_spread:.4f}."
            )

        if manifest.reviewer_used and reviewer_result:
            decision_path = "reviewer_override"
        elif selection_strategy == "majority_vote":
            decision_path = "local_majority"
        elif selection_strategy == "ensemble":
            decision_path = "local_ensemble"
        else:
            decision_path = selection_strategy

        stage_briefs = {
            "extraction": extraction.groq_brief.model_dump() if getattr(extraction, "groq_brief", None) else None,
            "cleaning": cleaning.groq_brief.model_dump() if getattr(cleaning, "groq_brief", None) else None,
            "modeling": modeling.groq_brief.model_dump() if getattr(modeling, "groq_brief", None) else None,
            "orchestrator": orchestrator_brief.model_dump() if orchestrator_brief else None,
        }

        motor = {
            "requested": manifest.request.model_choice.value,
            "selected": modeling.selected_model,
            "decision": manifest.decision,
            "reviewer_used": manifest.reviewer_used,
            "reviewer_provider": manifest.reviewer_provider,
            "decision_path": decision_path,
            "compare_binance": bool(manifest.request.compare_binance),
            "legacy_enabled": bool(legacy_analysis and legacy_analysis.get("enabled", False)),
        }

        agent_notes = [
            {
                "agent": "extraction_agent",
                "motor": stage_briefs["extraction"]["motor"] if stage_briefs["extraction"] else "yfinance",
                "summary": (
                    f"Pulled {raw_rows} raw rows from yfinance across {len(manifest.request.tickers)} ticker(s). "
                    f"Missing columns: {', '.join(extraction.missing_columns) or 'none'}."
                    if not es
                    else f"Se extrajeron {raw_rows} filas crudas desde yfinance para {len(manifest.request.tickers)} símbolo(s). "
                    f"Columnas faltantes: {', '.join(extraction.missing_columns) or 'ninguna'}."
                ),
                "groq_brief": stage_briefs["extraction"],
            },
            {
                "agent": "cleaning_agent",
                "motor": stage_briefs["cleaning"]["motor"] if stage_briefs["cleaning"] else "feature_engineering",
                "summary": (
                    f"Cleaned {cleaning.rows_in} raw rows into {clean_rows} model-ready rows. "
                    f"Dropped duplicates: {cleaning.clean_data.quality_report.get('dropped_duplicates', 0)}; "
                    f"dropped missing rows: {cleaning.clean_data.quality_report.get('dropped_missing', 0)}."
                    if not es
                    else f"Se limpiaron {cleaning.rows_in} filas crudas y quedaron {clean_rows} filas listas para modelado. "
                    f"Duplicados eliminados: {cleaning.clean_data.quality_report.get('dropped_duplicates', 0)}; "
                    f"filas faltantes eliminadas: {cleaning.clean_data.quality_report.get('dropped_missing', 0)}."
                ),
                "groq_brief": stage_briefs["cleaning"],
            },
            {
                "agent": "modeling_agent",
                "motor": stage_briefs["modeling"]["motor"] if stage_briefs["modeling"] else modeling.selected_model,
                "summary": (
                    f"Trained {', '.join(trained_models)} using the latest sample set "
                    f"({modeling.latest_sample_count} row(s)) and selected {modeling.selected_model} "
                    f"as the deterministic local candidate."
                    if not es
                    else f"Se entrenaron {', '.join(trained_models)} usando la muestra más reciente "
                    f"({modeling.latest_sample_count} fila(s)) y se seleccionó {modeling.selected_model} "
                    f"como candidato local determinístico."
                ),
                "groq_brief": stage_briefs["modeling"],
            },
            {
                "agent": "orchestrator",
                "motor": motor["selected"],
                "summary": (
                    f"Combined model votes using {selection_strategy}; final decision was {manifest.decision} "
                    f"with confidence {round(manifest.confidence or 0.0, 4)}."
                    if not es
                    else f"Se combinaron las votaciones usando {selection_strategy}; la decisión final fue {manifest.decision} "
                    f"con confianza {round(manifest.confidence or 0.0, 4)}."
                ),
                "groq_brief": stage_briefs["orchestrator"],
            },
        ]
        if legacy_analysis:
            legacy_status = "failed" if legacy_analysis.get("error") else "succeeded"
            legacy_model_source = legacy_analysis.get("model_source", "n/a")
            legacy_trust = legacy_analysis.get("trust_score_pct", "n/a")
            agent_notes.append(
                {
                    "agent": "legacy_bridge",
                    "motor": legacy_analysis.get("selected_model", "n/a"),
                    "summary": (
                        f"Ran the legacy bridge with status {legacy_status}; selected {legacy_analysis.get('selected_model', 'n/a')} "
                        f"using {legacy_model_source} models and trust score {legacy_trust}."
                        if not es
                        else f"Se ejecutó el puente legacy con estado {legacy_status}; se seleccionó {legacy_analysis.get('selected_model', 'n/a')} "
                        f"usando modelos {legacy_model_source} y una confianza de {legacy_trust}."
                    ),
                    "groq_brief": None,
                }
            )

        comparison_requested = bool(manifest.request.compare_binance)
        brain_enabled = bool(manifest.request.experimental_groq_brain)
        run_mode = self._resolve_run_mode(manifest, legacy_analysis, brain_used)

        summary = {
            "run_id": manifest.run_id,
            "status": manifest.status.value,
            "created_at": manifest.created_at.isoformat(),
            "updated_at": manifest.updated_at.isoformat(),
            "tickers": list(manifest.request.tickers),
            "date_range": {
                "start": manifest.request.start.isoformat(),
                "end": manifest.request.end.isoformat(),
                "interval": manifest.request.interval,
            },
            "rows": {
                "raw": extraction.rows,
                "cleaned": cleaning.rows_out,
            },
            "files": {
                "run_dir": str(run_dir),
                "raw": str(run_dir / "raw" / "raw_market_data.csv"),
                "cleaned": str(run_dir / "cleaned" / "clean_market_data.csv"),
                "summary": str(run_dir / "summary.json"),
                "result": str(run_dir / "result.json"),
            },
            "models": {
                "trained": trained_models,
                "selected": modeling.selected_model,
                "final_decision": manifest.decision,
                "confidence": round(manifest.confidence or 0.0, 6),
                "deterministic_decision": deterministic_decision or manifest.deterministic_decision or modeling.ensemble_prediction,
                "deterministic_confidence": round(
                    deterministic_confidence if deterministic_confidence is not None else manifest.deterministic_confidence or modeling.ensemble_probability,
                    6,
                ),
                "disagreement": modeling.disagreement,
                "disagreement_reason": disagreement_reason,
                "reviewer_used": manifest.reviewer_used,
                "reviewer_provider": manifest.reviewer_provider,
                "groq_brain_used": brain_used,
                "groq_brain_provider": manifest.groq_brain_provider,
                "decision_source": decision_source or manifest.decision_source or ("groq_brain" if brain_used else decision_path),
                "details": model_details,
                "vote_counts": vote_counts,
            },
            "selection": {
                "strategy": selection_strategy,
                "reason": selection_reason,
            },
            "comparison": {
                "extraction_health": extraction_health,
                "modeling_health": modeling_health,
                "raw_to_clean_retention_pct": retention_pct,
                "extraction_vs_models": (
                    f"{raw_rows} raw rows became {clean_rows} cleaned rows, "
                    f"and the 3 models produced a {selection_strategy} decision."
                    if not es
                    else f"{raw_rows} filas crudas se convirtieron en {clean_rows} filas limpias, "
                    f"y los 3 modelos produjeron una decisión {selection_strategy}."
                ),
                "decision_alignment": (
                    selection_summary
                    if not es
                    else selection_summary
                ),
                "decision_path": decision_path,
            },
            "data": {
                "raw_columns": raw_columns,
                "feature_columns": feature_columns,
                "derived_columns": derived_columns,
                "target_column": cleaning.target_column,
            },
            "agents": agent_notes,
            "overview": (
                f"{', '.join(manifest.request.tickers)} over {manifest.request.interval} went through extraction, cleaning, "
                f"3-model training, and orchestrator selection."
                if not es
                else f"{', '.join(manifest.request.tickers)} sobre {manifest.request.interval} pasó por extracción, limpieza, "
                f"entrenamiento de 3 modelos y selección del orquestador."
            ),
            "language": manifest.request.language,
            "metrics": model_metrics,
            "motor": motor,
            "stage_briefs": stage_briefs,
            "brain": {
                "enabled": brain_enabled,
                "used": brain_used,
                "provider": manifest.groq_brain_provider,
                "decision_source": decision_source or manifest.decision_source or ("groq_brain" if brain_used else decision_path),
                "deterministic_decision": deterministic_decision or manifest.deterministic_decision or modeling.ensemble_prediction,
                "deterministic_confidence": round(
                    deterministic_confidence if deterministic_confidence is not None else manifest.deterministic_confidence or modeling.ensemble_probability,
                    6,
                ),
                "decision": manifest.decision,
                "confidence": round(manifest.confidence or 0.0, 6),
                "rationale": manifest.rationale,
                "risks": brain_result.risks if brain_result else [],
                "explanation": brain_result.explanation if brain_result else "",
            },
            "artifacts_count": len(manifest.artifacts),
            "logs_count": len(manifest.logs),
            "rationale": manifest.rationale,
            "reviewer": reviewer_result.model_dump() if reviewer_result else None,
            "groq_brain": brain_result.model_dump() if brain_result else None,
            "source_comparison": source_comparison.model_dump() if source_comparison else None,
            "legacy_analysis": legacy_analysis,
            "adaptive": adaptive_report.model_dump() if adaptive_report else None,
            "operations": operations_report.model_dump() if operations_report else None,
            "review_mode": manifest.request.review_mode,
            "run_mode": run_mode,
        }
        return summary

    def _write_summary(
        self,
        manifest: RunManifest,
        extraction,
        cleaning,
        modeling,
        reviewer_result,
        run_dir: Path,
        source_comparison: Optional[SourceComparison] = None,
        legacy_analysis: Optional[Dict[str, Any]] = None,
        orchestrator_brief: Optional[StageBrief] = None,
        brain_result: Optional[ReviewerResult] = None,
        deterministic_decision: Optional[str] = None,
        deterministic_confidence: Optional[float] = None,
        brain_used: bool = False,
        decision_source: Optional[str] = None,
        adaptive_report: Optional[AdaptiveReport] = None,
        operations_report: Optional[OperationsReport] = None,
    ) -> Dict[str, Any]:
        summary = self._build_summary(
            manifest,
            extraction,
            cleaning,
            modeling,
            reviewer_result,
            run_dir,
            source_comparison,
            legacy_analysis,
            orchestrator_brief,
            brain_result,
            deterministic_decision,
            deterministic_confidence,
            brain_used,
            decision_source,
            adaptive_report,
            operations_report,
        )
        _write_json(run_dir / "summary.json", summary)
        return summary

    def _build_adaptive_report(
        self,
        *,
        request: PipelineRequest,
        extraction,
        cleaning,
        modeling,
        run_dir: Path,
        source_comparison: Optional[SourceComparison] = None,
        legacy_analysis: Optional[Dict[str, Any]] = None,
    ) -> AdaptiveReport:
        drift = self.drift_detector.assess(
            request=request,
            extraction=extraction,
            cleaning=cleaning,
            modeling=modeling,
            source_comparison=source_comparison,
            legacy_analysis=legacy_analysis,
        )
        selection = self.adaptive_selector.select(
            request=request,
            modeling=modeling,
            drift=drift,
        )
        shadow = self.shadow_runner.run(
            request=request,
            modeling=modeling,
            drift=drift,
            selection=selection,
            run_dir=run_dir,
        )
        feature_registry = self.feature_registry.snapshot(
            cleaning=cleaning,
            run_dir=run_dir,
        )
        validation = self.adaptive_validator.validate(
            drift=drift,
            selection=selection,
            shadow=shadow,
        )
        retraining = self.retraining_scheduler.plan(
            request=request,
            drift=drift,
            selection=selection,
            validation=validation,
            run_dir=run_dir,
        )
        approval = self.adaptive_policy.evaluate(
            request=request,
            drift=drift,
            selection=selection,
            feature_registry=feature_registry,
            shadow=shadow,
            validation=validation,
            retraining=retraining,
            run_dir=run_dir,
        )
        promotion = self.promotion_gate.decide(
            drift=drift,
            approval=approval,
            selection=selection,
            validation=validation,
            shadow=shadow,
        )
        runtime_fingerprint = self.runtime_fingerprint_builder.build(
            request=request,
            modeling=modeling,
            cleaning=cleaning,
            run_dir=run_dir,
        )
        summary = (
            f"Adaptive control recorded drift={drift.level} ({drift.score:.4f}) and recommended "
            f"{selection.recommended_action} in {selection.mode} mode; shadow={shadow.status}; "
            f"approval={approval.status}; promotion={promotion.mode}; retraining={retraining.status}."
        )

        adaptive_dir = run_dir / "adaptive"
        adaptive_dir.mkdir(parents=True, exist_ok=True)
        report_path = adaptive_dir / "adaptive_report.json"
        payload = {
            "enabled": True,
            "mode": selection.mode,
            "drift": drift.model_dump(),
            "selection": selection.model_dump(),
            "shadow": shadow.model_dump(),
            "feature_registry": feature_registry.model_dump(),
            "retraining": retraining.model_dump(),
            "runtime_fingerprint": runtime_fingerprint.model_dump(),
            "validation": validation.model_dump(),
            "approval": approval.model_dump(),
            "promotion": promotion.model_dump(),
            "applied": False,
            "summary": summary,
        }
        _write_json(report_path, payload)
        artifact = ArtifactRef(
            name=report_path.name,
            path=str(report_path),
            kind="adaptive_report",
            size_bytes=report_path.stat().st_size,
        )
        return AdaptiveReport(
            enabled=True,
            mode=selection.mode,
            artifact=artifact,
            drift=drift,
            selection=selection,
            shadow=shadow,
            feature_registry=feature_registry,
            retraining=retraining,
            runtime_fingerprint=runtime_fingerprint,
            validation=validation,
            approval=approval,
            promotion=promotion,
            applied=False,
            summary=summary,
        )

    def _mirror_model_artifact(self, source: Path, run_id: str) -> Path:
        mirrored_dir = self.models_root / run_id
        mirrored_dir.mkdir(parents=True, exist_ok=True)
        destination = mirrored_dir / source.name
        shutil.copy2(source, destination)
        return destination

    def run(self, request: PipelineRequest) -> PipelineResult:
        run_id = self._next_run_id()
        run_dir = self._run_dir(run_id)
        self._ensure_layout(run_dir)

        manifest = RunManifest(run_id=run_id, request=request, status=RunState.running)
        self._save_manifest(manifest, run_dir)
        self._append_log(manifest, run_dir, "orchestrator", "run_started", tickers=request.tickers)

        extraction_agent = ExtractionAgent(advisor=self.advisor)
        cleaning_agent = CleaningAgent(advisor=self.advisor)
        modeling_agent = ModelingAgent(advisor=self.advisor)
        source_comparison = None
        legacy_analysis = None
        orchestrator_brief = None

        try:
            def _record_stage_failure(stage: str, agent: str, event: str, exc: Exception, **details: Any) -> None:
                error = ErrorPayload(
                    stage=stage,
                    message=str(exc),
                    traceback=traceback.format_exc(),
                    details={"run_id": run_id, **details},
                )
                manifest.status = RunState.failed
                manifest.error = error
                self._append_log(
                    manifest,
                    run_dir,
                    agent,
                    event,
                    status=RunState.failed,
                    error=str(exc),
                    **details,
                )
                self._save_manifest(manifest, run_dir)
                _write_json(run_dir / "error.json", error)

            self._append_log(manifest, run_dir, "extraction_agent", "starting_extraction")

            try:
                source_comparison = self._build_source_comparison(request)
            except Exception as exc:
                source_comparison = SourceComparison(
                    enabled=False,
                    source_1="yfinance",
                    source_2="binance",
                    asset=request.comparison_asset,
                    timeframe=request.interval,
                    date_range={"start": request.start.isoformat(), "end": request.end.isoformat()},
                    note=(
                        "Binance comparison failed inside the orchestrator flow."
                        if request.language == "en"
                        else "La comparación con Binance falló dentro del flujo del orquestador."
                    ),
                    error=str(exc),
                )
                manifest.source_comparison = source_comparison
                self._append_log(
                    manifest,
                    run_dir,
                    "orchestrator",
                    "source_comparison_failed",
                    status=RunState.failed,
                    error=str(exc),
                )

            try:
                extraction = extraction_agent.run(request, run_dir)
            except Exception as exc:
                _record_stage_failure(
                    "extraction",
                    "extraction_agent",
                    "extraction_failed",
                    exc,
                    tickers=request.tickers,
                    start=request.start.isoformat(),
                    end=request.end.isoformat(),
                    interval=request.interval,
                    source="yfinance",
                )
                raise
            manifest.artifacts.append(extraction.raw_data.artifact)
            self._append_log(
                manifest,
                run_dir,
                "extraction_agent",
                "extraction_finished",
                status=RunState.succeeded,
                rows=extraction.rows,
                missing_columns=extraction.missing_columns,
            )

            if source_comparison is not None and source_comparison.enabled:
                manifest.source_comparison = source_comparison
                source_comparison_path = run_dir / "source_comparison.json"
                _write_json(source_comparison_path, source_comparison)
                manifest.artifacts.append(
                    ArtifactRef(
                        name="source_comparison.json",
                        path=str(source_comparison_path),
                        kind="source_comparison",
                        size_bytes=source_comparison_path.stat().st_size,
                    )
                )
                self._append_log(
                    manifest,
                    run_dir,
                    "orchestrator",
                    "source_comparison_completed",
                    status=RunState.succeeded,
                    source_1=source_comparison.source_1,
                    source_2=source_comparison.source_2,
                    asset=source_comparison.asset,
                )
            elif source_comparison is not None:
                manifest.source_comparison = source_comparison
                if not source_comparison.error:
                    self._append_log(
                        manifest,
                        run_dir,
                        "orchestrator",
                        "source_comparison_skipped",
                        status=RunState.running,
                        reason=source_comparison.note,
                    )

            self._append_log(manifest, run_dir, "cleaning_agent", "starting_cleaning")
            try:
                cleaning = cleaning_agent.run(request, extraction, run_dir)
            except Exception as exc:
                _record_stage_failure(
                    "cleaning",
                    "cleaning_agent",
                    "cleaning_failed",
                    exc,
                    tickers=request.tickers,
                    rows_in=getattr(extraction, "rows", None),
                    raw_columns=getattr(getattr(extraction, "raw_data", None), "columns", []),
                    target_column="target_direction",
                )
                raise
            manifest.artifacts.append(cleaning.clean_data.artifact)
            self._append_log(
                manifest,
                run_dir,
                "cleaning_agent",
                "cleaning_finished",
                status=RunState.succeeded,
                rows_in=cleaning.rows_in,
                rows_out=cleaning.rows_out,
            )

            self._append_log(manifest, run_dir, "modeling_agent", "starting_modeling")
            try:
                modeling = modeling_agent.run(request, cleaning, run_dir)
            except Exception as exc:
                _record_stage_failure(
                    "modeling",
                    "modeling_agent",
                    "modeling_failed",
                    exc,
                    tickers=request.tickers,
                    rows_in=getattr(cleaning, "rows_in", None),
                    rows_out=getattr(cleaning, "rows_out", None),
                    feature_columns=getattr(cleaning, "feature_columns", []),
                    target_column=getattr(cleaning, "target_column", "target_direction"),
                )
                raise
            manifest.artifacts.append(modeling.artifact)
            self._mirror_model_artifact(Path(modeling.artifact.path), run_id)
            for model in modeling.models:
                manifest.artifacts.append(model.artifact)
                self._mirror_model_artifact(Path(model.artifact.path), run_id)
            self._append_log(
                manifest,
                run_dir,
                "modeling_agent",
                "modeling_finished",
                status=RunState.succeeded,
                selected_model=modeling.selected_model,
                disagreement=modeling.disagreement,
                ensemble_prediction=modeling.ensemble_prediction,
                majority_prediction=modeling.majority_prediction,
            )

            if request.compare_binance:
                legacy_asset = request.comparison_asset or request.comparison_yfinance_ticker or (request.tickers[0] if len(request.tickers) == 1 else None)
                if legacy_asset:
                    self._append_log(
                        manifest,
                        run_dir,
                        "legacy_bridge",
                        "starting_legacy_bridge",
                        asset=legacy_asset,
                    )
                    try:
                        raw_df = pd.read_csv(extraction.raw_data.artifact.path)
                        legacy_analysis = build_legacy_btc_analysis(
                            raw_df,
                            run_dir,
                            asset=legacy_asset,
                            language=request.language,
                            source_comparison=source_comparison.model_dump() if source_comparison else None,
                            cleaning=cleaning,
                            modeling=modeling,
                        )
                        manifest.legacy_analysis = legacy_analysis
                        legacy_artifact = legacy_analysis.get("artifact")
                        if legacy_artifact:
                            manifest.artifacts.append(ArtifactRef.model_validate(legacy_artifact))
                        self._append_log(
                            manifest,
                            run_dir,
                            "legacy_bridge",
                            "legacy_bridge_completed",
                            status=RunState.succeeded,
                            selected_model=legacy_analysis.get("selected_model"),
                            disagreement=legacy_analysis.get("disagreement"),
                            modeling_health=legacy_analysis.get("modeling_health"),
                            trust_score_pct=legacy_analysis.get("trust_score_pct"),
                        )
                    except Exception as exc:
                        legacy_analysis = {
                            "enabled": False,
                            "comparison_mode": "legacy_bridge",
                            "asset": legacy_asset,
                            "error": str(exc),
                            "created_at": datetime.now(timezone.utc).isoformat(),
                        }
                        manifest.legacy_analysis = legacy_analysis
                        self._append_log(
                            manifest,
                            run_dir,
                            "legacy_bridge",
                            "legacy_bridge_failed",
                            status=RunState.failed,
                            error=str(exc),
                        )

            deterministic_decision = modeling.ensemble_prediction
            deterministic_confidence = modeling.ensemble_probability
            rationale = modeling.rationale
            final_decision = deterministic_decision
            final_confidence = deterministic_confidence
            reviewer_result = None
            reviewer_used = False
            brain_result = None
            brain_used = False
            adaptive_report = None
            decision_source = "local_ensemble"

            if modeling.disagreement:
                decision_source = "local_majority"

            if request.model_choice != ModelFamily.auto:
                if request.model_choice == ModelFamily.ensemble:
                    final_decision = modeling.ensemble_prediction
                    final_confidence = modeling.ensemble_probability
                    rationale = "Explicit ensemble selection from request."
                    decision_source = "explicit_ensemble"
                else:
                    selected = next((m for m in modeling.models if m.model_name == request.model_choice.value), None)
                    if selected:
                        final_decision = selected.latest_prediction
                        final_confidence = selected.confidence
                        rationale = f"Explicit model selection: {request.model_choice.value}"
                        decision_source = f"explicit_{request.model_choice.value}"

            should_review = False
            if request.review_mode == "on":
                should_review = True
            elif request.review_mode == "auto" and (modeling.disagreement or final_confidence < request.confidence_threshold):
                should_review = True

            if request.experimental_groq_brain:
                should_review = False

            if should_review:
                packet = ReviewerPacket(
                    run_id=run_id,
                    candidate_decision=final_decision,
                    confidence=final_confidence,
                    disagreement=modeling.disagreement,
                    language=request.language,
                    summary={
                        "ensemble_prediction": modeling.ensemble_prediction,
                        "majority_prediction": modeling.majority_prediction,
                        "ensemble_probability": modeling.ensemble_probability,
                        "selected_model": modeling.selected_model,
                        "deterministic_decision": deterministic_decision,
                        "deterministic_confidence": deterministic_confidence,
                        "experimental_groq_brain": bool(request.experimental_groq_brain),
                    },
                    model_results=[m.model_dump() for m in modeling.models],
                    logs=[log.model_dump() for log in manifest.logs],
                )
                reviewer_result = self.reviewer.review(packet) if request.use_reviewer else None
                if reviewer_result:
                    reviewer_used = True
                    final_decision = reviewer_result.decision
                    final_confidence = reviewer_result.confidence
                    rationale = reviewer_result.explanation
                    decision_source = "reviewer_override"
                    manifest.reviewer_provider = reviewer_result.provider
                    self._append_log(
                        manifest,
                        run_dir,
                        "groq_reviewer",
                        "review_completed",
                        status=RunState.succeeded,
                        decision=reviewer_result.decision,
                        confidence=reviewer_result.confidence,
                    )
                else:
                    final_decision = "hold" if final_confidence < request.confidence_threshold else final_decision
                    decision_source = f"{decision_source}_review_fallback"
                    if request.review_mode == "off":
                        rationale = (
                            "Deterministic fallback used because reviewer mode was off."
                            if request.language == "en"
                            else "Se usó un respaldo determinístico porque el modo revisor estaba desactivado."
                        )
                    else:
                        rationale = (
                            "Deterministic fallback used because reviewer was disabled or unavailable."
                            if request.language == "en"
                            else "Se usó un respaldo determinístico porque el revisor estaba deshabilitado o no disponible."
                        )

            if request.experimental_groq_brain:
                brain_packet = ReviewerPacket(
                    run_id=run_id,
                    candidate_decision=final_decision,
                    confidence=final_confidence,
                    disagreement=modeling.disagreement,
                    language=request.language,
                    summary={
                        "ensemble_prediction": modeling.ensemble_prediction,
                        "majority_prediction": modeling.majority_prediction,
                        "ensemble_probability": modeling.ensemble_probability,
                        "selected_model": modeling.selected_model,
                        "deterministic_decision": deterministic_decision,
                        "deterministic_confidence": deterministic_confidence,
                        "reviewer_used": reviewer_used,
                        "reviewer_provider": manifest.reviewer_provider,
                        "experimental_groq_brain": True,
                        "compare_binance": bool(request.compare_binance),
                        "legacy_enabled": bool(legacy_analysis and legacy_analysis.get("enabled", False)),
                    },
                    model_results=[m.model_dump() for m in modeling.models],
                    logs=[log.model_dump() for log in manifest.logs],
                )
                brain_enabled = bool(getattr(self.brain, "enabled", False))
                if not hasattr(self.brain, "enabled") and callable(getattr(self.brain, "decide", None)):
                    brain_enabled = True
                brain_result = self.brain.decide(brain_packet) if brain_enabled else None
                if brain_result:
                    brain_used = True
                    final_decision = _normalize_direction(brain_result.decision, deterministic_decision)
                    final_confidence = brain_result.confidence
                    rationale = brain_result.explanation
                    decision_source = "groq_brain"
                    manifest.groq_brain_provider = brain_result.provider
                    self._append_log(
                        manifest,
                        run_dir,
                        "groq_brain",
                        "brain_completed",
                        status=RunState.succeeded,
                        decision=final_decision,
                        confidence=final_confidence,
                    )
                else:
                    final_decision = "hold" if final_confidence < request.confidence_threshold else final_decision
                    decision_source = "groq_brain_unavailable"
                    rationale = (
                        "Groq brain was unavailable, so the deterministic result was kept."
                        if request.language == "en"
                        else "Groq brain no estuvo disponible, así que se conservó el resultado determinístico."
                    )

            adaptive_report = self._build_adaptive_report(
                request=request,
                extraction=extraction,
                cleaning=cleaning,
                modeling=modeling,
                run_dir=run_dir,
                source_comparison=source_comparison,
                legacy_analysis=legacy_analysis,
            )
            manifest.adaptive = adaptive_report
            if adaptive_report.artifact:
                manifest.artifacts.append(adaptive_report.artifact)
            if adaptive_report.shadow.artifact:
                manifest.artifacts.append(adaptive_report.shadow.artifact)
            if adaptive_report.feature_registry.artifact:
                manifest.artifacts.append(adaptive_report.feature_registry.artifact)
            if adaptive_report.retraining.artifact:
                manifest.artifacts.append(adaptive_report.retraining.artifact)
            if adaptive_report.runtime_fingerprint.artifact:
                manifest.artifacts.append(adaptive_report.runtime_fingerprint.artifact)
            if adaptive_report.approval.artifact:
                manifest.artifacts.append(adaptive_report.approval.artifact)
            self._append_log(
                manifest,
                run_dir,
                "adaptive_control",
                "adaptive_report_generated",
                status=RunState.succeeded,
                drift_level=adaptive_report.drift.level,
                drift_score=adaptive_report.drift.score,
                recommended_action=adaptive_report.selection.recommended_action,
                shadow_status=adaptive_report.shadow.status,
                shadow_ready=adaptive_report.shadow.ready_for_promotion,
                approved_features_pct=adaptive_report.feature_registry.approved_pct,
                retraining_status=adaptive_report.retraining.status,
                fingerprint_id=adaptive_report.runtime_fingerprint.fingerprint_id,
                validation_status=adaptive_report.validation.status,
                approval_status=adaptive_report.approval.status,
                promotion_mode=adaptive_report.promotion.mode,
                promotion_eligible=adaptive_report.promotion.eligible,
            )

            manifest.motor = {
                "requested": request.model_choice.value,
                "selected": modeling.selected_model,
                "decision": final_decision,
                "reviewer_used": reviewer_used,
                "reviewer_provider": manifest.reviewer_provider,
                "brain_enabled": bool(request.experimental_groq_brain),
                "brain_used": brain_used,
                "decision_source": decision_source,
                "deterministic_decision": deterministic_decision,
                "deterministic_confidence": round(deterministic_confidence, 6),
                "decision_path": (
                    "groq_brain_override"
                    if brain_used
                    else (
                        "reviewer_override"
                        if reviewer_used
                        else ("local_majority" if modeling.disagreement else "local_ensemble")
                    )
                ),
                "compare_binance": bool(request.compare_binance),
                "legacy_enabled": bool(legacy_analysis and legacy_analysis.get("enabled", False)),
            }

            if self.advisor.enabled:
                orchestrator_brief = self.advisor.brief(
                    "orchestrator",
                    {
                        "motor": modeling.selected_model,
                        "requested_motor": request.model_choice.value,
                        "decision": final_decision,
                        "confidence": round(final_confidence or 0.0, 6),
                        "reviewer_used": reviewer_used,
                        "reviewer_provider": manifest.reviewer_provider,
                        "review_mode": request.review_mode,
                        "compare_binance": bool(request.compare_binance),
                        "legacy_enabled": bool(legacy_analysis and legacy_analysis.get("enabled", False)),
                        "selected_model": modeling.selected_model,
                        "majority_prediction": modeling.majority_prediction,
                        "ensemble_prediction": modeling.ensemble_prediction,
                        "language": request.language,
                    },
                )
                manifest.orchestrator_brief = orchestrator_brief
                if orchestrator_brief and orchestrator_brief.motor:
                    manifest.motor["groq_brief_motor"] = orchestrator_brief.motor

            manifest.experimental_groq_brain = bool(request.experimental_groq_brain)
            manifest.deterministic_decision = deterministic_decision
            manifest.deterministic_confidence = deterministic_confidence
            manifest.groq_brain_used = brain_used
            manifest.decision_source = decision_source
            manifest.status = RunState.succeeded
            manifest.decision = final_decision
            manifest.confidence = final_confidence
            manifest.rationale = rationale
            manifest.reviewer_used = reviewer_used
            self._append_log(manifest, run_dir, "orchestrator", "run_finished", status=RunState.succeeded, decision=final_decision)

            result = PipelineResult(
                run_id=run_id,
                status=manifest.status,
                manifest=manifest,
                extraction=extraction,
                cleaning=cleaning,
                modeling=modeling,
                reviewer=reviewer_result,
                groq_brain=brain_result,
                final_decision=final_decision,
                final_confidence=final_confidence,
                rationale=rationale,
                artifact_root=str(self.artifact_root),
                source_comparison=source_comparison,
                legacy_analysis=legacy_analysis,
                adaptive=adaptive_report,
                motor=manifest.motor,
                deterministic_decision=deterministic_decision,
                deterministic_confidence=deterministic_confidence,
                decision_source=decision_source,
                orchestrator_brief=orchestrator_brief,
            )
            run_mode = self._resolve_run_mode(manifest, legacy_analysis, brain_used)
            operations_report = self.operations_builder.build(
                manifest=manifest,
                run_dir=run_dir,
                run_mode=run_mode,
                extraction=extraction,
                cleaning=cleaning,
                modeling=modeling,
                adaptive_report=adaptive_report,
            )
            manifest.operations = operations_report
            if operations_report.observability.artifact:
                manifest.artifacts.append(operations_report.observability.artifact)
            if operations_report.verify_gate.artifact:
                manifest.artifacts.append(operations_report.verify_gate.artifact)
            if operations_report.soak_gate.artifact:
                manifest.artifacts.append(operations_report.soak_gate.artifact)
            if operations_report.release_summary.artifact:
                manifest.artifacts.append(operations_report.release_summary.artifact)
            self._append_log(
                manifest,
                run_dir,
                "operations_control",
                "operations_report_generated",
                status=RunState.succeeded,
                verify_ok=operations_report.verify_gate.ok,
                soak_ok=operations_report.soak_gate.ok,
                release_ok=operations_report.release_summary.ok,
                release_stage=operations_report.release_summary.release_stage,
            )
            result.operations = operations_report
            self._write_summary(
                manifest,
                extraction,
                cleaning,
                modeling,
                reviewer_result,
                run_dir,
                source_comparison,
                legacy_analysis,
                orchestrator_brief,
                brain_result,
                deterministic_decision,
                deterministic_confidence,
                brain_used,
                decision_source,
                adaptive_report,
                operations_report,
            )
            self._finalize(manifest, run_dir, result)
            return result
        except Exception as exc:
            if manifest.error and getattr(manifest.error, "stage", "") in {"extraction", "cleaning", "modeling"}:
                raise
            error = ErrorPayload(
                stage="orchestrator",
                message=str(exc),
                traceback=traceback.format_exc(),
                details={"run_id": run_id},
            )
            manifest.status = RunState.failed
            manifest.error = error
            self._append_log(manifest, run_dir, "orchestrator", "run_failed", status=RunState.failed, error=str(exc))
            self._save_manifest(manifest, run_dir)
            _write_json(run_dir / "error.json", error)
            raise

    def load_run(self, run_id: str) -> Dict[str, Any]:
        run_dir = self._run_dir(run_id)
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            return {"manifest": None, "result": None, "logs": [], "summary": None}
        manifest = _read_json(manifest_path)
        result_path = run_dir / "result.json"
        logs_path = run_dir / "logs.json"
        summary_path = run_dir / "summary.json"
        result = _read_json(result_path) if result_path.exists() else None
        logs = _read_json(logs_path) if logs_path.exists() else []
        summary = _read_json(summary_path) if summary_path.exists() else None
        return {"manifest": manifest, "result": result, "logs": logs, "summary": summary}

    def latest_run_id(self) -> Optional[str]:
        runs = [path.name for path in self.runs_root.iterdir() if path.is_dir()]
        if not runs:
            return None
        return sorted(runs)[-1]

    def list_runs(self) -> List[Dict[str, Any]]:
        runs: List[Dict[str, Any]] = []
        for run_dir in sorted([path for path in self.runs_root.iterdir() if path.is_dir()]):
            summary_path = run_dir / "summary.json"
            manifest_path = run_dir / "manifest.json"
            if summary_path.exists():
                runs.append(_read_json(summary_path))
            elif manifest_path.exists():
                manifest = _read_json(manifest_path)
                runs.append(
                    {
                        "run_id": manifest.get("run_id"),
                        "status": manifest.get("status"),
                        "tickers": manifest.get("request", {}).get("tickers", []),
                        "date_range": {
                            "start": manifest.get("request", {}).get("start"),
                            "end": manifest.get("request", {}).get("end"),
                            "interval": manifest.get("request", {}).get("interval"),
                        },
                        "final_decision": manifest.get("decision"),
                        "confidence": manifest.get("confidence"),
                        "artifacts_count": len(manifest.get("artifacts", [])),
                        "logs_count": len(manifest.get("logs", [])),
                    }
                )
        return runs

    def list_models(self) -> List[Dict[str, Any]]:
        models: List[Dict[str, Any]] = []
        for model_path in sorted(self.models_root.rglob("*.joblib")):
            models.append(
                {
                    "name": model_path.stem,
                    "path": str(model_path),
                    "size_bytes": model_path.stat().st_size,
                }
            )
        return models
