from __future__ import annotations

from datetime import datetime, date, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class RunState(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class ModelFamily(str, Enum):
    auto = "auto"
    ensemble = "ensemble"
    logistic_regression = "logistic_regression"
    random_forest = "random_forest"
    gradient_boosting = "gradient_boosting"


class ArtifactRef(BaseModel):
    name: str
    path: str
    kind: str
    size_bytes: Optional[int] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AgentLog(BaseModel):
    agent: str
    status: RunState = RunState.running
    level: str = "info"
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: Optional[datetime] = None


class ErrorPayload(BaseModel):
    stage: str
    message: str
    traceback: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class PipelineRequest(BaseModel):
    tickers: List[str] = Field(default_factory=lambda: ["AAPL"])
    start: date
    end: date
    interval: str = "1d"
    model_choice: ModelFamily = ModelFamily.auto
    confidence_threshold: float = 0.60
    use_reviewer: bool = True
    review_mode: Literal["auto", "off", "on"] = "auto"
    reviewer_provider: Literal["groq"] = "groq"
    artifact_root: str = "artifacts"
    compare_binance: bool = False
    comparison_asset: Optional[str] = None
    comparison_yfinance_ticker: Optional[str] = None
    comparison_binance_symbol: Optional[str] = None
    language: Literal["en", "es"] = "en"
    experimental_groq_brain: bool = False


class RawMarketData(BaseModel):
    ticker: str
    start: date
    end: date
    interval: str
    rows: int
    columns: List[str]
    missing_columns: List[str] = Field(default_factory=list)
    artifact: ArtifactRef


class CleanMarketData(BaseModel):
    rows_in: int
    rows_out: int
    feature_columns: List[str]
    target_column: str
    artifact: ArtifactRef
    quality_report: Dict[str, Any] = Field(default_factory=dict)


class SourceComparison(BaseModel):
    enabled: bool = False
    source_1: str = "yfinance"
    source_2: str = "binance"
    asset: Optional[str] = None
    timeframe: Optional[str] = None
    date_range: Dict[str, str] = Field(default_factory=dict)
    coverage: Dict[str, Any] = Field(default_factory=dict)
    missing_fields: Dict[str, List[str]] = Field(default_factory=dict)
    row_counts: Dict[str, Any] = Field(default_factory=dict)
    close_price_alignment: Dict[str, Any] = Field(default_factory=dict)
    note: str = ""
    error: Optional[str] = None


class ModelResult(BaseModel):
    model_name: str
    validation_metrics: Dict[str, float] = Field(default_factory=dict)
    latest_probability: float
    latest_prediction: str
    confidence: float
    artifact: ArtifactRef


class ModelingResult(BaseModel):
    models: List[ModelResult] = Field(default_factory=list)
    ensemble_probability: float
    ensemble_prediction: str
    majority_prediction: str
    disagreement: bool
    selected_model: str
    rationale: str
    latest_sample_count: int
    artifact: ArtifactRef
    groq_brief: Optional[StageBrief] = None


class ReviewerPacket(BaseModel):
    run_id: str
    candidate_decision: str
    confidence: float
    disagreement: bool
    language: Literal["en", "es"] = "en"
    summary: Dict[str, Any] = Field(default_factory=dict)
    model_results: List[Dict[str, Any]] = Field(default_factory=list)
    logs: List[Dict[str, Any]] = Field(default_factory=list)


class ReviewerResult(BaseModel):
    provider: str
    decision: str
    confidence: float
    explanation: str
    risks: List[str] = Field(default_factory=list)
    raw_response: Dict[str, Any] = Field(default_factory=dict)


class StageBrief(BaseModel):
    stage: str
    provider: str = "groq"
    enabled: bool = False
    motor: Optional[str] = None
    summary: str = ""
    key_points: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    raw_response: Dict[str, Any] = Field(default_factory=dict)


class DriftAssessment(BaseModel):
    level: Literal["stable", "watch", "drifted"] = "stable"
    score: float = 0.0
    reasons: List[str] = Field(default_factory=list)


class AdaptiveSelection(BaseModel):
    mode: Literal["observe_only", "manual_gate", "auto_apply"] = "observe_only"
    recommended_action: str = "keep_current_strategy"
    recommended_model: Optional[str] = None
    recommended_confidence_threshold: Optional[float] = None
    recommended_review_mode: Optional[str] = None
    reasons: List[str] = Field(default_factory=list)


class AdaptiveValidation(BaseModel):
    passed: bool = False
    status: Literal["observe_only", "shadow_only", "candidate_ready", "blocked"] = "observe_only"
    reasons: List[str] = Field(default_factory=list)


class AdaptiveApproval(BaseModel):
    status: Literal["observe_only", "blocked", "manual_review", "approved_for_promotion_review"] = "observe_only"
    artifact: Optional[ArtifactRef] = None
    requires_manual_signoff: bool = True
    auto_apply_allowed: bool = False
    proposed_changes: Dict[str, Any] = Field(default_factory=dict)
    checks: Dict[str, bool] = Field(default_factory=dict)
    reasons: List[str] = Field(default_factory=list)


class AdaptivePromotionReview(BaseModel):
    run_id: str = ""
    status: Literal["approved", "rejected"] = "rejected"
    approved: bool = False
    reviewer: str = ""
    reviewed_at: str = ""
    notes: str = ""
    policy_status: str = ""
    promotion_mode: str = ""
    proposed_changes: Dict[str, Any] = Field(default_factory=dict)
    reasons: List[str] = Field(default_factory=list)
    artifact: Optional[ArtifactRef] = None


class AdaptiveCandidateConfig(BaseModel):
    version: str = ""
    run_id: str = ""
    created_at: str = ""
    source_review_status: str = ""
    source_reviewer: str = ""
    proposed_changes: Dict[str, Any] = Field(default_factory=dict)
    runtime_fingerprint_id: str = ""
    feature_registry_version: str = ""
    artifact: Optional[ArtifactRef] = None


class AdaptivePromotionApplication(BaseModel):
    run_id: str = ""
    status: Literal["prepared"] = "prepared"
    prepared_by: str = ""
    prepared_at: str = ""
    notes: str = ""
    config_version: str = ""
    reasons: List[str] = Field(default_factory=list)
    config_artifact: Optional[ArtifactRef] = None
    artifact: Optional[ArtifactRef] = None


class AdaptivePromotionExecution(BaseModel):
    run_id: str = ""
    status: Literal["applied"] = "applied"
    applied_by: str = ""
    applied_at: str = ""
    notes: str = ""
    config_version: str = ""
    reasons: List[str] = Field(default_factory=list)
    application_artifact: Optional[ArtifactRef] = None
    artifact: Optional[ArtifactRef] = None


class PromotionDecision(BaseModel):
    eligible: bool = False
    mode: Literal["manual_only", "shadow_only", "promotion_ready"] = "manual_only"
    reasons: List[str] = Field(default_factory=list)


class ShadowRunReport(BaseModel):
    executed: bool = False
    status: Literal["not_needed", "planned", "completed"] = "not_needed"
    candidate_model: Optional[str] = None
    baseline_score: Optional[float] = None
    candidate_score: Optional[float] = None
    improvement: Optional[float] = None
    ready_for_promotion: bool = False
    reasons: List[str] = Field(default_factory=list)
    artifact: Optional[ArtifactRef] = None


class FeatureRegistrySnapshot(BaseModel):
    version: str = "v1"
    approved_features: List[str] = Field(default_factory=list)
    observed_features: List[str] = Field(default_factory=list)
    missing_features: List[str] = Field(default_factory=list)
    extra_features: List[str] = Field(default_factory=list)
    approved_pct: float = 100.0
    artifact: Optional[ArtifactRef] = None


class RetrainingPlan(BaseModel):
    status: Literal["monitor", "scheduled", "immediate"] = "monitor"
    recommended_within_hours: Optional[int] = None
    candidate_model: Optional[str] = None
    recommended_review_mode: Optional[str] = None
    recommended_confidence_threshold: Optional[float] = None
    reasons: List[str] = Field(default_factory=list)
    artifact: Optional[ArtifactRef] = None


class RuntimeFingerprint(BaseModel):
    version: str = "v1"
    fingerprint_id: str = ""
    python_version: str = ""
    platform: str = ""
    request: Dict[str, Any] = Field(default_factory=dict)
    selected_model: Optional[str] = None
    feature_count: int = 0
    artifact: Optional[ArtifactRef] = None


class AdaptiveReport(BaseModel):
    enabled: bool = True
    mode: Literal["observe_only", "manual_gate", "auto_apply"] = "observe_only"
    artifact: Optional[ArtifactRef] = None
    drift: DriftAssessment = Field(default_factory=DriftAssessment)
    selection: AdaptiveSelection = Field(default_factory=AdaptiveSelection)
    shadow: ShadowRunReport = Field(default_factory=ShadowRunReport)
    feature_registry: FeatureRegistrySnapshot = Field(default_factory=FeatureRegistrySnapshot)
    retraining: RetrainingPlan = Field(default_factory=RetrainingPlan)
    runtime_fingerprint: RuntimeFingerprint = Field(default_factory=RuntimeFingerprint)
    validation: AdaptiveValidation = Field(default_factory=AdaptiveValidation)
    approval: AdaptiveApproval = Field(default_factory=AdaptiveApproval)
    promotion: PromotionDecision = Field(default_factory=PromotionDecision)
    applied: bool = False
    summary: str = ""


class RuntimeObservability(BaseModel):
    status: Literal["captured", "partial"] = "captured"
    artifact: Optional[ArtifactRef] = None
    runtime_fingerprint_id: str = ""
    run_started_at: str = ""
    run_finished_at: str = ""
    run_duration_ms: int = 0
    log_count: int = 0
    artifact_count: int = 0
    decision_path: str = "n/a"
    review_mode: str = "auto"
    run_mode: str = "local_only"
    status_counts: Dict[str, int] = Field(default_factory=dict)
    agent_event_counts: Dict[str, int] = Field(default_factory=dict)
    artifact_kind_counts: Dict[str, int] = Field(default_factory=dict)
    stage_statuses: Dict[str, str] = Field(default_factory=dict)
    stage_durations_ms: Dict[str, int] = Field(default_factory=dict)
    data_profile: Dict[str, Any] = Field(default_factory=dict)
    decision_profile: Dict[str, Any] = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list)


class VerifyGate(BaseModel):
    ok: bool = False
    artifact: Optional[ArtifactRef] = None
    checks: Dict[str, bool] = Field(default_factory=dict)
    reasons: List[str] = Field(default_factory=list)


class SoakGate(BaseModel):
    ok: bool = False
    executed: bool = False
    artifact: Optional[ArtifactRef] = None
    status: Literal["not_executed", "pending", "passed", "failed"] = "not_executed"
    required_hours: int = 72
    observed_hours: float = 0.0
    sampled_runs: int = 0
    ready_runs: int = 0
    window_start: str = ""
    window_end: str = ""
    reasons: List[str] = Field(default_factory=list)


class ReleaseSummary(BaseModel):
    ok: bool = False
    artifact: Optional[ArtifactRef] = None
    release_stage: Literal["dev_ready", "ops_pending", "release_ready"] = "dev_ready"
    reasons: List[str] = Field(default_factory=list)


class OperationsReport(BaseModel):
    observability: RuntimeObservability = Field(default_factory=RuntimeObservability)
    verify_gate: VerifyGate = Field(default_factory=VerifyGate)
    soak_gate: SoakGate = Field(default_factory=SoakGate)
    release_summary: ReleaseSummary = Field(default_factory=ReleaseSummary)


class ReadinessReport(BaseModel):
    ok: bool = False
    status: Literal["ready", "degraded", "not_ready"] = "not_ready"
    artifact_root: str = "artifacts"
    latest_run_id: Optional[str] = None
    checks: Dict[str, bool] = Field(default_factory=dict)
    reasons: List[str] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)


class ReleaseGateReport(BaseModel):
    ok: bool = False
    status: Literal["blocked", "pending_soak", "pending_apply", "release_ready", "release_applied"] = "blocked"
    artifact_root: str = "artifacts"
    latest_run_id: Optional[str] = None
    checks: Dict[str, bool] = Field(default_factory=dict)
    reasons: List[str] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)


class ReleaseBoardEntry(BaseModel):
    run_id: str
    status: str = ""
    tickers: List[str] = Field(default_factory=list)
    final_decision: Optional[str] = None
    confidence: Optional[float] = None
    run_mode: Optional[str] = None
    decision_path: Optional[str] = None
    verify_ok: bool = False
    readiness_status: str = "not_ready"
    release_status: str = "blocked"
    release_stage: Optional[str] = None
    soak_status: Optional[str] = None
    manual_review_status: Optional[str] = None
    promotion_application_status: Optional[str] = None
    promotion_application_version: Optional[str] = None
    promotion_lifecycle_status: Optional[str] = None
    run_duration_ms: int = 0


class ReleaseBoard(BaseModel):
    artifact_root: str = "artifacts"
    latest_run_id: Optional[str] = None
    total_runs: int = 0
    entries: List[ReleaseBoardEntry] = Field(default_factory=list)


class OperationsDashboard(BaseModel):
    artifact_root: str = "artifacts"
    latest_run_id: Optional[str] = None
    readyz: ReadinessReport = Field(default_factory=ReadinessReport)
    release_gate: ReleaseGateReport = Field(default_factory=ReleaseGateReport)
    release_board: ReleaseBoard = Field(default_factory=ReleaseBoard)
    latest_run_verify: Dict[str, Any] = Field(default_factory=dict)
    latest_run_observability: Dict[str, Any] = Field(default_factory=dict)
    artifacts: Dict[str, ArtifactRef] = Field(default_factory=dict)


class OperationsRefreshReport(BaseModel):
    artifact_root: str = "artifacts"
    run_id: Optional[str] = None
    readyz: ReadinessReport = Field(default_factory=ReadinessReport)
    release_gate: ReleaseGateReport = Field(default_factory=ReleaseGateReport)
    release_board: ReleaseBoard = Field(default_factory=ReleaseBoard)
    dashboard: OperationsDashboard = Field(default_factory=OperationsDashboard)
    soak: Dict[str, Any] = Field(default_factory=dict)
    artifacts: Dict[str, ArtifactRef] = Field(default_factory=dict)


class OperationsJobCycle(BaseModel):
    cycle_index: int = 0
    run_id: Optional[str] = None
    readyz_status: str = "not_ready"
    release_status: str = "blocked"
    soak_status: str = "skipped"
    started_at: str = ""
    finished_at: str = ""


class OperationsJobReport(BaseModel):
    artifact_root: str = "artifacts"
    run_id: Optional[str] = None
    cycles_requested: int = 1
    cycles_completed: int = 0
    interval_seconds: int = 0
    include_soak: bool = True
    latest_release_status: str = "blocked"
    latest_soak_status: str = "skipped"
    cycles: List[OperationsJobCycle] = Field(default_factory=list)
    artifact: Optional[ArtifactRef] = None


class OperationsSchedule(BaseModel):
    artifact_root: str = "artifacts"
    run_id: Optional[str] = None
    enabled: bool = True
    interval_seconds: int = 300
    limit: int = 10
    required_hours: int = 72
    include_soak: bool = True
    created_at: str = ""
    updated_at: str = ""
    last_run_at: str = ""
    next_run_at: str = ""
    artifact: Optional[ArtifactRef] = None


class OperationsScheduleRunReport(BaseModel):
    artifact_root: str = "artifacts"
    run_id: Optional[str] = None
    forced: bool = False
    due: bool = False
    executed: bool = False
    checked_at: str = ""
    reason: str = ""
    schedule: Optional[OperationsSchedule] = None
    job: Optional[OperationsJobReport] = None
    artifact: Optional[ArtifactRef] = None


class OperationsAutomationBundle(BaseModel):
    artifact_root: str = "artifacts"
    run_id: Optional[str] = None
    generated_at: str = ""
    command: str = ""
    cron_expression: str = "* * * * *"
    shell_artifact: Optional[ArtifactRef] = None
    cron_artifact: Optional[ArtifactRef] = None
    systemd_service_artifact: Optional[ArtifactRef] = None
    systemd_timer_artifact: Optional[ArtifactRef] = None
    artifact: Optional[ArtifactRef] = None


class RunManifest(BaseModel):
    run_id: str
    status: RunState = RunState.queued
    request: PipelineRequest
    artifacts: List[ArtifactRef] = Field(default_factory=list)
    logs: List[AgentLog] = Field(default_factory=list)
    decision: Optional[str] = None
    confidence: Optional[float] = None
    rationale: Optional[str] = None
    reviewer_used: bool = False
    reviewer_provider: Optional[str] = None
    deterministic_decision: Optional[str] = None
    deterministic_confidence: Optional[float] = None
    experimental_groq_brain: bool = False
    groq_brain_used: bool = False
    groq_brain_provider: Optional[str] = None
    decision_source: Optional[str] = None
    error: Optional[ErrorPayload] = None
    source_comparison: Optional[SourceComparison] = None
    legacy_analysis: Optional[Dict[str, Any]] = None
    adaptive: Optional[AdaptiveReport] = None
    operations: Optional[OperationsReport] = None
    motor: Dict[str, Any] = Field(default_factory=dict)
    orchestrator_brief: Optional[StageBrief] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ExtractionResult(BaseModel):
    rows: int
    tickers: List[str]
    missing_columns: List[str] = Field(default_factory=list)
    raw_data: RawMarketData
    groq_brief: Optional[StageBrief] = None


class CleaningResult(BaseModel):
    rows_in: int
    rows_out: int
    feature_columns: List[str]
    target_column: str
    clean_data: CleanMarketData
    groq_brief: Optional[StageBrief] = None


class PipelineResult(BaseModel):
    run_id: str
    status: RunState
    manifest: RunManifest
    extraction: ExtractionResult
    cleaning: CleaningResult
    modeling: ModelingResult
    reviewer: Optional[ReviewerResult] = None
    groq_brain: Optional[ReviewerResult] = None
    final_decision: str
    final_confidence: float
    rationale: str
    artifact_root: str
    source_comparison: Optional[SourceComparison] = None
    legacy_analysis: Optional[Dict[str, Any]] = None
    adaptive: Optional[AdaptiveReport] = None
    operations: Optional[OperationsReport] = None
    motor: Dict[str, Any] = Field(default_factory=dict)
    deterministic_decision: Optional[str] = None
    deterministic_confidence: Optional[float] = None
    decision_source: Optional[str] = None
    orchestrator_brief: Optional[StageBrief] = None
