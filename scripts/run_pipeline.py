
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env_loader import load_project_env
load_project_env()

from pipeline.orchestrator import PipelineOrchestrator
from schemas.pipeline import ModelFamily, PipelineRequest


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the IAC MoneyLab quant pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Mode examples:\n"
            "  deterministic / local only:\n"
            "    python3 scripts/run_pipeline.py --tickers AAPL --start 2024-01-01 --end 2025-01-01\n"
            "  Binance comparison:\n"
            "    python3 scripts/run_pipeline.py --tickers BTC-USD --start 2024-01-01 --end 2025-01-01 --compare-binance\n"
            "    legacy bridge is only available through compare-binance for BTC/ETH/LTC aliases\n"
            "  experimental Groq brain:\n"
            "    python3 scripts/run_pipeline.py --tickers AAPL --start 2024-01-01 --end 2025-01-01 --groq-brain\n"
            "  comparison + experimental brain:\n"
            "    python3 scripts/run_pipeline.py --tickers BTC-USD --start 2024-01-01 --end 2025-01-01 --compare-binance --groq-brain\n"
        ),
    )
    parser.add_argument("--tickers", nargs="+", default=["AAPL"], help="Tickers to analyze")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--interval", default="1d", help="yfinance interval")
    parser.add_argument("--artifact-root", default="artifacts", help="Artifact root directory")
    parser.add_argument(
        "--model-choice",
        default="auto",
        choices=[member.value for member in ModelFamily],
        help="Final model choice strategy",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.60, help="Reviewer trigger threshold")
    parser.add_argument("--no-reviewer", action="store_true", help="Disable Groq reviewer even if available")
    parser.add_argument("--review-mode", choices=["auto", "off", "on"], default="auto", help="When to use Groq reviewer")
    parser.add_argument("--compare-binance", action="store_true", help="Add Binance bonus comparison")
    parser.add_argument("--comparison-asset", help="Asset label used in the source comparison report")
    parser.add_argument("--comparison-yfinance-ticker", help="yfinance ticker for the comparison source")
    parser.add_argument("--comparison-binance-symbol", help="Binance symbol for the comparison source")
    parser.add_argument("--groq-brain", action="store_true", help="Experimental Groq brain final-decision mode")
    parser.add_argument("--language", choices=["en", "es"], default="en", help="Console output language")
    return parser


def _print_summary(summary: dict) -> None:
    lang = summary.get("language", "en")
    es = lang == "es"
    labels = {
        "title": "=== Resumen de ejecución IAC MoneyLab ===" if es else "=== IAC MoneyLab Run Summary ===",
        "run_id": "ID de ejecución" if es else "Run ID",
        "status": "Estado" if es else "Status",
        "tickers": "Símbolos" if es else "Tickers",
        "date_range": "Rango de fechas" if es else "Date range",
        "rows": "Filas" if es else "Rows",
        "models_trained": "Modelos entrenados" if es else "Models trained",
        "selected_mode": "Modo seleccionado" if es else "Selected mode",
        "run_mode": "Modo de ejecución" if es else "Run mode",
        "motor": "Motor" if es else "Motor",
        "overview": "Resumen" if es else "Overview",
        "selection_why": "Por qué" if es else "Selection why",
        "selection_strat": "Estrategia" if es else "Selection strat",
        "extraction_vs_models": "Extracción vs modelos" if es else "Extraction vs models",
        "extraction_health": "Salud de extracción" if es else "Extraction health",
        "modeling_health": "Salud del modelado" if es else "Modeling health",
        "decision_path": "Ruta de decisión" if es else "Decision path",
        "source_comparison": "Comparación de fuente" if es else "Source comparison",
        "legacy_analysis": "Análisis legacy" if es else "Legacy analysis",
        "legacy_mode": "Modo legacy" if es else "Legacy mode",
        "legacy_rows": "Filas legacy" if es else "Legacy rows",
        "legacy_models": "Modelos legacy" if es else "Legacy models",
        "legacy_rationale": "Razón legacy" if es else "Legacy rationale",
        "legacy_trust": "Confianza legacy" if es else "Legacy trust",
        "legacy_alignment": "Alineación legacy" if es else "Legacy alignment",
        "legacy_source": "Fuente de modelos" if es else "Model source",
        "final_decision": "Decisión final" if es else "Final decision",
        "confidence": "Confianza" if es else "Confidence",
        "reviewer_used": "Revisor usado" if es else "Reviewer used",
        "reviewer_mode": "Modo del revisor" if es else "Reviewer mode",
        "brain_mode": "Modo Groq brain" if es else "Groq brain mode",
        "brain_decision": "Decisión brain" if es else "Brain decision",
        "deterministic_decision": "Decisión determinista" if es else "Deterministic decision",
        "decision_source": "Fuente de decisión" if es else "Decision source",
        "artifacts": "Artefactos" if es else "Artifacts",
        "logs": "Registros" if es else "Logs",
        "model_detail": "Detalle de modelos" if es else "Model detail",
        "agent_notes": "Notas de agentes" if es else "Agent notes",
        "why_disagreed": "Por qué discreparon los modelos" if es else "Why the models disagreed",
        "data_columns": "Columnas de datos" if es else "Data columns",
        "files": "Archivos" if es else "Files",
        "summary": "Resumen" if es else "Summary",
        "result": "Resultado" if es else "Result",
        "rationale": "Razón" if es else "Rationale",
    }

    print(f"\n{labels['title']}")
    print(f"{labels['run_id']}:        {summary['run_id']}")
    print(f"{labels['status']}:        {summary['status']}")
    print(f"{labels['tickers']}:       {', '.join(summary['tickers'])}")
    print(
        f"{labels['date_range']}:    "
        f"{summary['date_range']['start']} -> {summary['date_range']['end']} "
        f"({summary['date_range']['interval']})"
    )
    print(f"{labels['rows']}:          raw={summary['rows']['raw']}  cleaned={summary['rows']['cleaned']}")
    print(f"{labels['models_trained']}: {', '.join(summary['models']['trained'])}")
    print(f"{labels['selected_mode']}:  {summary['models']['selected']}")
    print(f"{labels['run_mode']}:       {summary.get('run_mode', 'local_only')}")
    if summary.get("motor"):
        motor = summary["motor"]
        print(
            f"{labels['motor']}:         requested={motor.get('requested', 'n/a')} "
            f"selected={motor.get('selected', 'n/a')} decision={motor.get('decision', 'n/a')}"
        )
    if 'review_mode' in summary:
        print(f"{labels['reviewer_mode']}:  {summary['review_mode']}")
    brain = summary.get("brain") or {}
    if brain:
        print(
            f"{labels['brain_mode']}:    enabled={brain.get('enabled')} used={brain.get('used')} "
            f"decision_source={brain.get('decision_source')}"
        )
        if brain.get("decision") is not None:
            print(
                f"{labels['brain_decision']}:  {brain.get('decision')} ({brain.get('confidence', 0.0):.4f})"
            )
            print(
                f"{labels['deterministic_decision']}: {brain.get('deterministic_decision')} "
                f"({brain.get('deterministic_confidence', 0.0):.4f})"
            )
    if 'overview' in summary:
        print(f"{labels['overview']}:      {summary['overview']}")
    if 'selection' in summary:
        print(f"{labels['selection_why']}:   {summary['selection']['reason']}")
        print(f"{labels['selection_strat']}: {summary['selection']['strategy']}")
    if 'comparison' in summary:
        print(f"{labels['extraction_vs_models']}: {summary['comparison']['extraction_vs_models']}")
        print(f"{labels['extraction_health']}:  {summary['comparison']['extraction_health']}")
        print(f"{labels['modeling_health']}:    {summary['comparison']['modeling_health']}")
        print(f"{labels['decision_path']}:      {summary['comparison'].get('decision_path', 'n/a')}")
    source_comparison = summary.get('source_comparison')
    if source_comparison:
        is_enabled = bool(source_comparison.get('enabled'))
        print(f"{labels['source_comparison']}:   {'enabled' if is_enabled else 'disabled'}")
        if source_comparison.get('error'):
            print(f"- error: {source_comparison.get('error')}")
        if source_comparison.get('note'):
            print(f"- note: {source_comparison.get('note')}")
        if is_enabled:
            print(
                f"- {source_comparison['source_1']} vs {source_comparison['source_2']} "
                f"for {source_comparison.get('asset') or 'the selected asset'} ({source_comparison.get('timeframe')})"
            )
            if source_comparison.get('coverage'):
                coverage = source_comparison['coverage']
                print(
                    f"- coverage overlap={coverage.get('overlap_rows')} "
                    f"yfinance_only={coverage.get('yfinance_only_rows')} "
                    f"binance_only={coverage.get('binance_only_rows')}"
                )
            if source_comparison.get('close_price_alignment'):
                close_alignment = source_comparison['close_price_alignment']
                print(
                    f"- close alignment mae={close_alignment.get('mae')} "
                    f"mape_pct={close_alignment.get('mape_pct')} "
                    f"corr={close_alignment.get('correlation')}"
                )
    legacy_analysis = summary.get('legacy_analysis')
    if legacy_analysis:
        legacy_enabled = bool(legacy_analysis.get('enabled', True))
        print(f"\n{labels['legacy_analysis']}:   {'enabled' if legacy_enabled else 'disabled'}")
        if legacy_analysis.get('error'):
            print(f"- error: {legacy_analysis.get('error')}")
        if legacy_analysis.get('bridge_note'):
            print(f"- note: {legacy_analysis.get('bridge_note')}")
        if legacy_enabled:
            print(f"- {labels['legacy_mode']}: {legacy_analysis.get('comparison_mode')}")
            print(f"- asset: {legacy_analysis.get('asset')}")
            if legacy_analysis.get('requested_asset'):
                print(f"- requested asset: {legacy_analysis.get('requested_asset')}")
            if legacy_analysis.get('matched_asset'):
                print(f"- matched asset: {legacy_analysis.get('matched_asset')}")
            if legacy_analysis.get('asset_aliases'):
                print(f"- aliases: {', '.join(legacy_analysis.get('asset_aliases', []))}")
            print(f"- {labels['legacy_rows']}: raw={legacy_analysis.get('rows')} clean={legacy_analysis.get('clean_rows')}")
            print(f"- selected: {legacy_analysis.get('selected_model')}")
            print(f"- health: {legacy_analysis.get('modeling_health')}")
            if legacy_analysis.get('model_source'):
                print(f"- {labels['legacy_source']}: {legacy_analysis.get('model_source')}")
            if legacy_analysis.get('schema_alignment_pct') is not None:
                print(f"- {labels['legacy_alignment']}: {legacy_analysis.get('schema_alignment_pct')}%")
            if legacy_analysis.get('trust_score_pct') is not None:
                print(f"- {labels['legacy_trust']}: {legacy_analysis.get('trust_score_pct')}%")
            if legacy_analysis.get('rationale'):
                print(f"- {labels['legacy_rationale']}: {legacy_analysis.get('rationale')}")
            if legacy_analysis.get('source_comparison'):
                legacy_sc = legacy_analysis['source_comparison']
                if legacy_sc.get('coverage'):
                    coverage = legacy_sc['coverage']
                    print(
                        f"- legacy source overlap={coverage.get('overlap_rows')} "
                        f"yfinance_only={coverage.get('yfinance_only_rows')} "
                        f"binance_only={coverage.get('binance_only_rows')}"
                    )
            print(f"\n{labels['legacy_models']}")
            for item in legacy_analysis.get('models', []):
                direction = 'above' if item.get('latest_probability', 0.0) >= 0.5 else 'below'
                if es:
                    direction_text = 'por encima' if direction == 'above' else 'por debajo'
                    print(
                        f"- {item.get('model_name')}: predijo {item.get('latest_prediction')} porque la probabilidad fue {item.get('latest_probability', 0.0):.4f} "
                        f"({direction_text} de 0.5000); confianza={item.get('confidence', 0.0):.4f}; "
                        f"acc={item.get('validation_metrics', {}).get('accuracy', 0.0):.4f}; roc_auc={item.get('validation_metrics', {}).get('roc_auc', 0.0):.4f}"
                    )
                else:
                    print(
                        f"- {item.get('model_name')}: predicted {item.get('latest_prediction')} because probability was {item.get('latest_probability', 0.0):.4f} "
                        f"({direction} 0.5000); confidence={item.get('confidence', 0.0):.4f}; "
                        f"acc={item.get('validation_metrics', {}).get('accuracy', 0.0):.4f}; roc_auc={item.get('validation_metrics', {}).get('roc_auc', 0.0):.4f}"
                    )
    print(f"{labels['final_decision']}: {summary['models']['final_decision']}")
    print(f"{labels['confidence']}:     {summary['models']['confidence']:.4f}")
    print(f"{labels['reviewer_used']}:  {summary['models']['reviewer_used']}")
    print(f"{labels['artifacts']}:      {summary['artifacts_count']} files")
    print(f"{labels['logs']}:           {summary['logs_count']} entries")
    print(f"\n{labels['model_detail']}")
    for item in summary['metrics']:
        direction = "above" if item['probability'] >= 0.5 else "below"
        if es:
            direction_text = "por encima" if direction == "above" else "por debajo"
            print(
                f"- {item['model']}: predijo {item['prediction']} porque la probabilidad fue {item['probability']:.4f} "
                f"({direction_text} de 0.5000); confianza={item['confidence']:.4f}; "
                f"acc={item['accuracy']:.4f}; roc_auc={item['roc_auc']:.4f}"
            )
        else:
            print(
                f"- {item['model']}: predicted {item['prediction']} because probability was {item['probability']:.4f} "
                f"({direction} 0.5000); confidence={item['confidence']:.4f}; "
                f"acc={item['accuracy']:.4f}; roc_auc={item['roc_auc']:.4f}"
            )
    if summary.get('agents'):
        print(f"\n{labels['agent_notes']}")
        for item in summary['agents']:
            print(f"- {item['agent']}: {item['summary']}")
    if summary.get('models', {}).get('disagreement_reason'):
        print(f"\n{labels['why_disagreed']}: {summary['models']['disagreement_reason']}")
    print(f"\n{labels['data_columns']}")
    print(f"- raw:     {', '.join(summary['data']['raw_columns'])}")
    print(f"- feature: {', '.join(summary['data']['feature_columns'])}")
    if summary['data']['derived_columns']:
        print(f"- derived: {', '.join(summary['data']['derived_columns'])}")
    print(f"- target:  {summary['data']['target_column']}")
    print(f"\n{labels['files']}:      {summary['files']['run_dir']}")
    print(f"{labels['summary']}:    {summary['files']['summary']}")
    print(f"{labels['result']}:     {summary['files']['result']}")
    print(f"{labels['rationale']}:  {summary['rationale']}")
def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    parser = build_parser()
    args = parser.parse_args()

    request = PipelineRequest(
        tickers=args.tickers,
        start=_parse_date(args.start),
        end=_parse_date(args.end),
        interval=args.interval,
        artifact_root=args.artifact_root,
        model_choice=ModelFamily(args.model_choice),
        confidence_threshold=args.confidence_threshold,
        use_reviewer=not args.no_reviewer,
        review_mode="off" if args.no_reviewer else args.review_mode,
        compare_binance=args.compare_binance,
        comparison_asset=args.comparison_asset,
        comparison_yfinance_ticker=args.comparison_yfinance_ticker,
        comparison_binance_symbol=args.comparison_binance_symbol,
        language=args.language,
        experimental_groq_brain=args.groq_brain,
    )

    orchestrator = PipelineOrchestrator(artifact_root=args.artifact_root)
    result = orchestrator.run(request)
    summary = orchestrator.load_run(result.run_id)["summary"]

    if summary is None:
        print(json.dumps(result.model_dump(), indent=2, default=str))
    else:
        _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
