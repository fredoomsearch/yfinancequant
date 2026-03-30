# Technical Note

## Propósito

Este documento resume la arquitectura final del sistema en términos de control plane, execution plane, capa adaptativa y operación.

La meta no es solo ejecutar un pipeline cuantitativo. La meta es operar un sistema trazable, explicable y gobernado.

## Arquitectura por planos

### Control plane

```text
Usuario
-> AssistantRouter
-> AssistantPlanner
-> Policy
-> Executor
-> Runtime
```

Responsabilidad:

- comprender intención
- extraer `run_id`, etapa, modo, ticker y fechas
- convertir lenguaje natural en una acción estructurada
- limitar tools y riesgo

### Execution plane

```text
PipelineOrchestrator
-> ExtractionAgent
-> CleaningAgent
-> ModelingAgent
-> artifacts por run
```

Responsabilidad:

- ejecutar extracción, limpieza y modelado
- consolidar la decisión final
- persistir `manifest/result/summary/logs/error`

### Capa adaptativa

```text
drift -> selector -> validator -> policy approval -> shadow -> manual review -> promotion gate
```

Responsabilidad:

- detectar drift y cambio de régimen
- proponer estrategia o modelo recomendado
- validar candidato vs baseline
- aprobar o bloquear cambios adaptativos con checks explícitos
- correr shadow benchmarking
- registrar signoff humano antes de cualquier promoción trazada
- marcar `observe_only`, `shadow_only` o `promotion_ready`

### Capa operativa

```text
observability -> verify -> readyz -> release gate -> release board -> dashboard -> soak gate -> ops refresh -> ops runner -> ops schedule
```

Responsabilidad:

- medir el run
- validar artifacts obligatorios
- reportar readiness
- bloquear o habilitar release
- resumir runs recientes
- materializar snapshots operativos persistidos
- materializar una ventana de soak
- ejecutar ciclos operativos periódicos sobre runs persistidos
- ejecutar schedules persistidos solo cuando el próximo ciclo está due

## Artifacts por corrida

Cada corrida puede persistir:

- `manifest.json`
- `summary.json`
- `result.json`
- `logs.json`
- `error.json` si falla
- `adaptive/adaptive_report.json`
- `adaptive/approval_decision.json`
- `adaptive/promotion_review.json`
- `adaptive/candidate_config_<run_id>-vNNN.json`
- `adaptive/promotion_application.json`
- `adaptive/promotion_execution.json`
- `adaptive/shadow_run.json`
- `adaptive/feature_registry.json`
- `adaptive/retraining_plan.json`
- `adaptive/runtime_fingerprint.json`
- `ops/runtime_observability.json`
- `ops/verify_gate.json`
- `ops/operations_dashboard.json`
- `ops/operations_dashboard.html`
- `ops/readyz.json`
- `ops/release_gate_report.json`
- `ops/release_board.json`
- `ops/soak_gate.json`
- `ops/release_summary.json`
- `ops/job_runner_report.json`
- `ops/schedule.json`
- `ops/schedule_run_report.json`
- `ops/automation_bundle.json`
- `ops/run_ops_schedule.sh`
- `ops/ops_schedule.crontab`
- `ops/iactest-ops-<run_id>.service`
- `ops/iactest-ops-<run_id>.timer`

## Contratos clave

### Assistant

- `AssistantRoute`
- `AssistantState`
- `AssistantPlan`
- `PlannerStep`

### Pipeline

- `PipelineRequest`
- `RunManifest`
- `PipelineResult`
- `AdaptiveReport`
- `OperationsReport`

### Operación

- `ReadinessReport`
- `ReleaseGateReport`
- `ReleaseBoard`
- `ReleaseBoardEntry`

## Política operativa

### Lo que el sistema sí permite

- responder grounded en artifacts
- observar drift
- proponer ajustes
- validar candidatos
- persistir gates y snapshots operativos

### Lo que el sistema no autoaplica libremente

- cambiar columnas raw
- redefinir target
- modificar pipeline productivo
- promover cambios sin validación
- promover cambios sin evidence trail

## Observabilidad actual

`RuntimeObservability` ya persiste:

- `run_started_at`
- `run_finished_at`
- `run_duration_ms`
- `status_counts`
- `agent_event_counts`
- `artifact_kind_counts`
- `stage_statuses`
- `stage_durations_ms`
- `data_profile`
- `decision_profile`

## Gates actuales

### Verify gate

Valida presencia de:

- manifest
- result
- summary
- logs
- adaptive report
- runtime observability

### Release gate

Valida además:

- run exitoso
- etapas core completas
- cero logs fallidos
- mínimos de datos y modelos
- decision path conocido
- selected model conocido
- signoff manual cuando un candidato adaptativo ya está en `promotion_ready`
- paquete `promotion_application` preparado cuando ya hubo aprobación manual
- estado `applied` trazado cuando el paquete preparado se promociona manualmente

Estados:

- `blocked`
- `pending_soak`
- `pending_apply`
- `release_ready`
- `release_applied`

### Soak gate

Se ejecuta externamente sobre runs persistidos.

Evalúa:

- ventana observada
- runs muestreados
- runs listos
- verify gate verde en la ventana

Estados:

- `not_executed`
- `pending`
- `passed`
- `failed`

## Comandos operativos

```bash
.venv/bin/python scripts/verify_run.py --artifact-root artifacts --run-id run_0001 --json
.venv/bin/python scripts/observability_report.py --artifact-root artifacts --run-id run_0001 --json
.venv/bin/python scripts/readyz.py --artifact-root artifacts --json
.venv/bin/python scripts/release_gate.py --artifact-root artifacts --json
.venv/bin/python scripts/release_board.py --artifact-root artifacts --limit 10 --json
.venv/bin/python scripts/ops_dashboard.py --artifact-root artifacts --run-id run_0001 --persist --json
.venv/bin/python scripts/review_adaptive.py --artifact-root artifacts --run-id run_0001 --reviewer ops-lead --decision approve --notes "manual signoff" --json
.venv/bin/python scripts/apply_adaptive.py --artifact-root artifacts --run-id run_0001 --operator release-operator --notes "prepare package" --json
.venv/bin/python scripts/promote_adaptive.py --artifact-root artifacts --run-id run_0001 --operator prod-operator --notes "apply package" --json
.venv/bin/python scripts/ops_refresh.py --artifact-root artifacts --run-id run_0001 --json
.venv/bin/python scripts/ops_runner.py --artifact-root artifacts --run-id run_0001 --cycles 3 --interval-seconds 0 --json
.venv/bin/python scripts/ops_schedule.py set --artifact-root artifacts --run-id run_0001 --interval-seconds 300 --start-immediately --json
.venv/bin/python scripts/ops_schedule.py run --artifact-root artifacts --run-id run_0001 --json
.venv/bin/python scripts/ops_automation.py --artifact-root artifacts --run-id run_0001 --json
.venv/bin/python scripts/soak_gate.py --artifact-root artifacts --run-id run_0001 --required-hours 72 --json
```

## Endpoints

- `GET /health`
- `GET /readyz`
- `GET /releasez`
- `GET /release-board`
- `GET /ops/dashboard`
- `GET /ops/dashboard/html`
- `GET /pipeline/runs/{run_id}/verify`
- `GET /pipeline/runs/{run_id}/observability`
- `POST /pipeline/runs/{run_id}/dashboard`
- `POST /pipeline/runs/{run_id}/adaptive/review`
- `POST /pipeline/runs/{run_id}/adaptive/apply`
- `POST /pipeline/runs/{run_id}/adaptive/promote`
- `POST /pipeline/runs/{run_id}/ops/refresh`
- `POST /pipeline/runs/{run_id}/ops/run`
- `POST /pipeline/runs/{run_id}/ops/schedule`
- `POST /pipeline/runs/{run_id}/ops/schedule/run`
- `POST /pipeline/runs/{run_id}/ops/automation`
- `POST /pipeline/runs/{run_id}/soak`

## Estado de madurez

Lectura aproximada después del cierre actual:

- arquitectura total: `84% - 87%`
- control plane LLM-céntrico: `64% - 70%`
- capa adaptativa cuantitativa gobernada: `62% - 68%`
- calidad operativa / observabilidad / release / soak: `72% - 78%`

## Siguiente paso natural

El siguiente bloque ya no es otra capa nueva. Es cierre de producto:

- consolidar README final
- unificar nota ejecutiva
- automatizar soak programado si se quiere operación continua
