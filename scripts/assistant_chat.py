from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env_loader import load_project_env
load_project_env()

from assistant.runtime import AssistantRuntime, _format_session_context_line


_ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "bright_red": "\033[91m",
    "bright_green": "\033[92m",
    "bright_yellow": "\033[93m",
    "bright_blue": "\033[94m",
    "bright_magenta": "\033[95m",
    "bright_cyan": "\033[96m",
    "bright_white": "\033[97m",
}

_LABEL_ORDER = [
    "Run",
    "In plain language",
    "Decision:",
    "Mode:",
    "Rows:",
    "Reviewer mode:",
    "Decision path:",
    "Selection:",
    "Health:",
    "Models disagreed:",
    "Data:",
    "Motor:",
]

_RUN_PREFIX_RE = re.compile(r"^Run\s+(?P<run_id>run_\d+)(?:\s+completed)?\.?\s*(?P<body>.*)$", re.S)
_AGENT_CARD_RE = re.compile(r"^__agent_card__:(?P<stage>[a-z_]+):(?P<lang>en|es)\n(?P<body>.*)$", re.S)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chat with the quant pipeline assistant")
    parser.add_argument("--artifact-root", default="artifacts", help="Artifact root directory")
    parser.add_argument("--session-id", default="default", help="Assistant session id")
    parser.add_argument("--message", help="Run a single assistant turn and exit")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors and terminal styling")
    return parser


def _use_color(disabled: bool) -> bool:
    if disabled:
        return False
    if os.getenv("NO_COLOR"):
        return False
    if not sys.stdout.isatty():
        return False
    return os.getenv("TERM", "dumb").lower() != "dumb"


def _paint(text: str, color: str, enabled: bool, *, bold: bool = False, dim: bool = False) -> str:
    if not enabled:
        return text
    pieces = []
    if bold:
        pieces.append(_ANSI["bold"])
    if dim:
        pieces.append(_ANSI["dim"])
    pieces.append(_ANSI.get(color, ""))
    pieces.append(text)
    pieces.append(_ANSI["reset"])
    return "".join(pieces)


def _wrap(text: str, width: int, indent: str = "") -> str:
    return textwrap.fill(
        text,
        width=width,
        initial_indent=indent,
        subsequent_indent=indent,
        break_long_words=False,
        replace_whitespace=False,
    )


def _strip_run_prefix(text: str) -> tuple[Optional[str], str]:
    raw = (text or "").strip()
    match = _RUN_PREFIX_RE.match(raw)
    if not match:
        return None, raw
    run_id = match.group("run_id")
    body = (match.group("body") or "").strip()
    return run_id, body


def _render_card(
    title: str,
    body_lines: list[str],
    *,
    enabled: bool,
    width: int,
    color: str = "white",
    subtitle: str | None = None,
) -> str:
    inner_width = max(48, width - 2)
    left_pad = " " * max(0, (width - (inner_width + 2)) // 2)
    flat_lines: list[str] = []
    for line in body_lines:
        text = (line or "").strip()
        if not text:
            flat_lines.append("")
            continue
        flat_lines.extend(textwrap.wrap(text, width=inner_width, break_long_words=False, replace_whitespace=False) or [""])

    label = title if not subtitle else f"{title} · {subtitle}"
    if not enabled:
        pieces = [f"{left_pad}[{label}]"]
        pieces.extend(f"{left_pad}  {line}" if line else "" for line in flat_lines)
        return "\n".join(pieces).rstrip()

    top = f"{left_pad}╭─ {_paint(label, color, enabled, bold=True)}"
    top = top + "─" * max(0, inner_width - len(label) - 1)
    body = [f"{left_pad}│ {line.ljust(inner_width - 2)} │" if line else f"{left_pad}│ {' ' * (inner_width - 2)} │" for line in flat_lines]
    bottom = f"{left_pad}╰" + "─" * (inner_width) + "╯"
    return "\n".join([top, *body, bottom])


def _render_kv_lines(lines: list[str], width: int) -> list[str]:
    rendered: list[str] = []
    for line in lines:
        for chunk in re.split(r"\n+", line or ""):
            text = (chunk or "").strip()
            if not text:
                rendered.append("")
                continue
            rendered.extend(textwrap.wrap(text, width=width, break_long_words=False, replace_whitespace=False) or [""])
    return rendered


def _table_widths(width: int, columns: int) -> tuple[list[int], int]:
    inner_width = max(72, width - 2)
    content_width = max(48, inner_width - 4)
    gap = 2
    base = max(10, (content_width - gap * (columns - 1)) // columns)
    widths = [base for _ in range(columns)]
    remainder = content_width - gap * (columns - 1) - base * columns
    for index in range(remainder):
        widths[index % columns] += 1
    return widths, inner_width


def _format_table_rows(
    rows: list[tuple[str, ...]],
    widths: list[int],
    *,
    enabled: bool,
    colors: list[str] | None = None,
    bold_first: bool = False,
) -> list[str]:
    rendered: list[str] = []
    for row in rows:
        wrapped_cells: list[list[str]] = []
        for index, cell in enumerate(row):
            wrapped = textwrap.wrap(
                (cell or "").strip(),
                width=widths[index],
                break_long_words=False,
                replace_whitespace=False,
            ) or [""]
            wrapped_cells.append(wrapped)
        row_height = max((len(cell_lines) for cell_lines in wrapped_cells), default=1)
        for line_index in range(row_height):
            pieces: list[str] = []
            for col_index, cell_lines in enumerate(wrapped_cells):
                text = cell_lines[line_index] if line_index < len(cell_lines) else ""
                text = text.ljust(widths[col_index])
                if enabled and colors and col_index < len(colors):
                    color = colors[col_index]
                    if color:
                        text = _paint(text, color, enabled, bold=bold_first and col_index == 0)
                pieces.append(text)
            rendered.append(f"│ {'  '.join(pieces)} │")
    return rendered


def _section_title(text: str, width: int, enabled: bool, color: str) -> str:
    inner_width = max(72, width - 2) - 4
    centered = text.center(inner_width)
    return f"│ {_paint(centered, color, enabled, bold=True)} │"


def _friendly_mode(mode: str, *, es: bool) -> str:
    value = (mode or "local_only").strip() or "local_only"
    has_groq_brain = "groq_brain" in value
    has_binance = "binance" in value
    if has_groq_brain and has_binance:
        return "compare-binance + Groq brain experimental" if not es else "compare-binance + Groq brain experimental"
    if has_groq_brain:
        return "Groq brain experimental"
    if "reviewer" in value:
        return "deterministic + reviewer" if not es else "determinista + revisor"
    if has_binance:
        return "deterministic + compare-binance" if not es else "determinista + compare-binance"
    labels = {
        "local_only": "determinista" if es else "deterministic",
        "local_plus_reviewer": "determinista" if es else "deterministic",
        "local_plus_binance": "determinista" if es else "deterministic",
        "local_plus_binance_legacy": "determinista" if es else "deterministic",
        "local_only_groq_brain": "Groq brain experimental" if not es else "Groq brain experimental",
        "local_plus_binance_groq_brain": "Groq brain experimental" if not es else "Groq brain experimental",
        "local_plus_binance_legacy_groq_brain": "Groq brain experimental" if not es else "Groq brain experimental",
    }
    return labels.get(value, value)


def _banner_rows(language: str) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    es = str(language or "en").lower().startswith("es")
    quick_rows = [
        ("1", "Agents", "menu"),
        ("2", "Extraction", "AAPL / BTC-USD / EURUSD=X"),
        ("3", "Cleaning", "what metrics are in the cleaned data?"),
        ("4", "Modeling", "what did the model predict?"),
        ("5", "Orchestrator", "why did it decide that way?"),
        ("6", "Status", "status"),
        ("7", "Full status", "status full"),
        ("8", "Clean data", "what does this row say?"),
        ("9", "Modes", "modes / compare-binance"),
        ("A", "English", "switch to English"),
        ("B", "Spanish", "switch to Spanish"),
        ("H", "Help", "what can you do?"),
    ]
    stage_rows = [
        ("Extraction", "what symbol was used?", "AAPL / MSFT / BTC-USD / EURUSD=X"),
        ("Cleaning", "what metrics are in the cleaned data?", "schema / exact row / row analysis"),
        ("Modeling", "what did the model predict?", "votes / confidence / long-short-hold"),
        ("Orchestrator", "why did it decide that way?", "final truth / compare-binance / brain"),
    ]
    mode_rows = [
        ("local_only", "default", "deterministic local path"),
        ("compare-binance", "comparison", "yfinance vs Binance"),
        ("--groq-brain", "experimental", "Groq can decide the final run"),
        ("compare-binance + --groq-brain", "combined", "comparison plus experimental brain"),
    ]
    if es:
        quick_rows = [
            ("1", "Agentes", "menú"),
            ("2", "Extracción", "AAPL / BTC-USD / EURUSD=X"),
            ("3", "Limpieza", "qué métricas hay en los datos limpios?"),
            ("4", "Modelado", "qué predijo el modelo?"),
            ("5", "Orquestador", "por qué decidió eso?"),
            ("6", "Estado", "estado"),
            ("7", "Estado completo", "estado completo"),
            ("8", "Datos limpios", "qué dice esta fila?"),
            ("9", "Modos", "modos / compare-binance"),
            ("A", "Inglés", "switch to English"),
            ("B", "Español", "switch to Spanish"),
            ("H", "Ayuda", "qué puedes hacer?"),
        ]
        stage_rows = [
            ("Extracción", "qué símbolo se usó?", "AAPL / MSFT / BTC-USD / EURUSD=X"),
            ("Limpieza", "qué métricas hay en los datos limpios?", "esquema / fila exacta / análisis de fila"),
            ("Modelado", "qué predijo el modelo?", "votos / confianza / long-short-hold"),
            ("Orquestador", "por qué decidió eso?", "verdad final / compare-binance / brain"),
        ]
        mode_rows = [
            ("local_only", "default", "ruta local determinista"),
            ("compare-binance", "comparación", "yfinance vs Binance"),
            ("--groq-brain", "experimental", "Groq puede decidir la corrida final"),
            ("compare-binance + --groq-brain", "combinado", "comparación más cerebro experimental"),
        ]
    return quick_rows, stage_rows, mode_rows


def _banner(
    enabled: bool,
    width: int,
    language: str = "en",
    active_mode: str = "local_only",
    context_line: str = "",
) -> str:
    es = str(language or "en").lower().startswith("es")
    inner_width = max(72, width - 2)
    text_width = inner_width - 4

    def line(text: str, color: str = "white", *, bold: bool = False, dim: bool = False) -> str:
        centered = text.center(text_width)
        return f"│ {_paint(centered, color, enabled, bold=bold, dim=dim)} │"

    top = _paint("╔" + "═" * inner_width + "╗", "cyan", enabled)
    bottom = _paint("╚" + "═" * inner_width + "╝", "cyan", enabled)
    divider = _paint("╟" + "─" * inner_width + "╢", "cyan", enabled)

    return "\n".join(
        [
            top,
            line("YFINANCE", "cyan", bold=True),
            line("QUANT ASSISTANT", "cyan", bold=True),
            line(
                "Menú principal"
                if es
                else "Main menu",
                "bright_yellow",
                bold=True,
            ),
            line(
                "1 agentes | 2 extracción | 3 limpieza | 4 modelado | 5 orquestador"
                if es
                else "1 agents | 2 extraction | 3 cleaning | 4 modeling | 5 orchestrator",
                "bright_white",
            ),
            line(
                "6 estado | 7 completo | 8 datos limpios | 9 modos | A inglés | B español | H ayuda"
                if es
                else "6 status | 7 full | 8 clean data | 9 modes | A English | B Spanish | H help",
                "bright_magenta",
            ),
            line(
                "Entrada natural: AAPL | metricas | fila | qué predijo el modelo | por qué decidió eso"
                if es
                else "Natural input: AAPL | metrics | row | what did the model predict | why did it decide that way",
                "bright_green",
            ),
            line(context_line or ("Contexto: run=n/a | símbolo=n/a | etapa=n/a" if es else "Context: run=n/a | symbol=n/a | stage=n/a"), "bright_white", dim=True),
            line(
                "Abre agents o modes para ver el detalle completo."
                if es
                else "Open agents or modes to see the full detail.",
                "bright_white",
                dim=True,
            ),
            divider,
            line(
                "Conversación: extracción -> limpieza -> modelado -> orquestador"
                if es
                else "Conversation: extraction -> cleaning -> modeling -> orchestrator",
                "bright_white",
                dim=True,
            ),
            line(
                "Escribe un número, una letra, un ticker, un nombre de etapa o una pregunta natural."
                if es
                else "Type a number, a letter, a ticker, a stage name, or a natural question.",
                "bright_white",
                dim=True,
            ),
            line(
                f"{'Modo activo' if es else 'Active mode'}: {_friendly_mode(active_mode, es=es)}",
                "bright_white",
                dim=True,
            ),
            bottom,
        ]
    )


def _label_color(label: str) -> str:
    mapping = {
        "Run": "cyan",
        "In plain language": "green",
        "Decision:": "magenta",
        "Mode:": "blue",
        "Rows:": "yellow",
        "Reviewer mode:": "cyan",
        "Decision path:": "cyan",
        "Selection:": "green",
        "Health:": "green",
        "Models disagreed:": "yellow",
        "Data:": "cyan",
        "Motor:": "blue",
    }
    return mapping.get(label, "white")


def _format_assistant_output(text: str, enabled: bool, width: int) -> str:
    raw = (text or "").strip()
    if not raw:
        return _paint("(empty response)", "yellow", enabled)

    agent_card_match = _AGENT_CARD_RE.match(raw)
    if agent_card_match:
        stage = agent_card_match.group("stage")
        lang = agent_card_match.group("lang")
        body = (agent_card_match.group("body") or "").strip()
        title_map = {
            ("extraction", "en"): "Extraction",
            ("extraction", "es"): "Extracción",
            ("cleaning", "en"): "Cleaning",
            ("cleaning", "es"): "Limpieza",
            ("modeling", "en"): "Modeling",
            ("modeling", "es"): "Modelado",
        }
        color_map = {
            "extraction": "blue",
            "cleaning": "green",
            "modeling": "yellow",
        }
        title = title_map.get((stage, lang), stage.title())
        subtitle = "agent" if lang == "en" else "agente"
        body_lines = [part.strip() for part in re.split(r"\n+", body) if part.strip()]
        return _render_card(
            title,
            body_lines,
            enabled=enabled,
            width=width,
            color=color_map.get(stage, "white"),
            subtitle=subtitle,
        )

    if raw.startswith("Session status full") or raw.startswith("Estado de sesión completo") or raw.startswith("Estado de sesion completo"):
        title = "Session Status" if raw.startswith("Session status full") else "Estado de sesión"
        subtitle = "full" if raw.startswith("Session status full") else "completo"
        body_lines = [part.strip() for part in re.split(r"\n+", raw or "")[1:] if part.strip()]
        return _render_card(
            title,
            body_lines,
            enabled=enabled,
            width=width,
            color="cyan",
            subtitle=subtitle,
        )
    if raw.startswith("Help hub") or raw.startswith("Centro de ayuda"):
        title = "Help hub" if raw.startswith("Help hub") else "Centro de ayuda"
        body_lines = [part.strip() for part in re.split(r"\n+", raw or "") if part.strip()]
        return _render_card(
            title,
            body_lines,
            enabled=enabled,
            width=width,
            color="bright_white",
        )
    if raw.startswith("Mode detail") or raw.startswith("Detalle de modo"):
        title = "Mode detail" if raw.startswith("Mode detail") else "Detalle de modo"
        body_lines = [part.strip() for part in re.split(r"\n+", raw or "") if part.strip()]
        return _render_card(
            title,
            body_lines,
            enabled=enabled,
            width=width,
            color="magenta",
        )
    if (
        raw.startswith("Extraction error")
        or raw.startswith("Error de extracción")
        or raw.startswith("Cleaning error")
        or raw.startswith("Error de limpieza")
        or raw.startswith("Modeling error")
        or raw.startswith("Error de modelado")
        or raw.startswith("Source comparison error")
        or raw.startswith("Error de comparación de fuentes")
        or raw.startswith("Legacy bridge error")
        or raw.startswith("Error del puente legacy")
        or raw.startswith("Run error")
        or raw.startswith("Error de ejecución")
    ):
        if raw.startswith("Extraction error"):
            title = "Extraction error"
        elif raw.startswith("Error de extracción"):
            title = "Error de extracción"
        elif raw.startswith("Cleaning error"):
            title = "Cleaning error"
        elif raw.startswith("Error de limpieza"):
            title = "Error de limpieza"
        elif raw.startswith("Modeling error"):
            title = "Modeling error"
        elif raw.startswith("Error de modelado"):
            title = "Error de modelado"
        elif raw.startswith("Source comparison error"):
            title = "Source comparison error"
        elif raw.startswith("Error de comparación de fuentes"):
            title = "Error de comparación de fuentes"
        elif raw.startswith("Legacy bridge error"):
            title = "Legacy bridge error"
        elif raw.startswith("Error del puente legacy"):
            title = "Error del puente legacy"
        elif raw.startswith("Run error"):
            title = "Run error"
        else:
            title = "Error de ejecución"
        body_lines = [part.strip() for part in re.split(r"\n+", raw or "") if part.strip()]
        return _render_card(
            title,
            body_lines,
            enabled=enabled,
            width=width,
            color="bright_red",
        )
    if (
        raw.startswith("Clean data hub")
        or raw.startswith("Centro de datos limpios")
        or raw.startswith("Clean market data overview")
        or raw.startswith("Resumen de clean_market_data")
        or raw.startswith("Clean market data analysis")
        or raw.startswith("Análisis de clean_market_data")
        or raw.startswith("Clean market data metrics")
        or raw.startswith("Métricas de clean_market_data")
    ):
        if raw.startswith("Clean market data analysis") or raw.startswith("Análisis de clean_market_data"):
            title = "Clean Data Analysis" if raw.startswith("Clean market data analysis") else "Análisis de datos limpios"
            color = "green"
        elif raw.startswith("Clean market data metrics") or raw.startswith("Métricas de clean_market_data"):
            title = "Clean Data Metrics" if raw.startswith("Clean market data metrics") else "Métricas de datos limpios"
            color = "yellow"
        elif raw.startswith("Clean data hub") or raw.startswith("Centro de datos limpios"):
            title = "Clean Data Hub" if raw.startswith("Clean data hub") else "Centro de datos limpios"
            color = "cyan"
        else:
            title = "Clean Data" if raw.startswith("Clean market data overview") else "Resumen de clean_market_data"
            color = "green"
        body_lines = [part.strip() for part in re.split(r"\n+", raw or "") if part.strip()]
        return _render_card(
            title,
            body_lines,
            enabled=enabled,
            width=width,
            color=color,
        )
    if raw.startswith("Source comparison") or raw.startswith("Comparación de fuentes"):
        title = "Source comparison" if raw.startswith("Source comparison") else "Comparación de fuentes"
        body_lines = [part.strip() for part in re.split(r"\n+", raw or "") if part.strip()]
        return _render_card(
            title,
            body_lines,
            enabled=enabled,
            width=width,
            color="yellow",
        )
    if raw.startswith("Decision") or raw.startswith("Decisión"):
        title = "Decision" if raw.startswith("Decision") else "Decisión"
        body_lines = [part.strip() for part in re.split(r"\n+", raw or "") if part.strip()]
        return _render_card(
            title,
            body_lines,
            enabled=enabled,
            width=width,
            color="magenta",
        )
    if raw.startswith("Asset") or raw.startswith("Activo") or raw.startswith("Last symbol used") or raw.startswith("Símbolo usado") or raw.startswith("Current asset"):
        title = "Asset" if raw.startswith("Asset") or raw.startswith("Last symbol used") or raw.startswith("Current asset") else "Activo"
        body_lines = [part.strip() for part in re.split(r"\n+", raw or "") if part.strip()]
        return _render_card(
            title,
            body_lines,
            enabled=enabled,
            width=width,
            color="cyan",
        )
    if raw.startswith("Continue") or raw.startswith("Continuar"):
        title = "Continue" if raw.startswith("Continue") else "Continuar"
        body_lines = [part.strip() for part in re.split(r"\n+", raw or "") if part.strip()]
        return _render_card(
            title,
            body_lines,
            enabled=enabled,
            width=width,
            color="white",
        )

    run_id, body = _strip_run_prefix(raw)

    # Structured run summaries are rendered as a short dashboard.
    if run_id:
        if body.startswith("I pulled"):
            title = "Extraction"
            body_lines = [body]
        elif body.startswith("Extraje") or body.startswith("En lenguaje simple, extraje"):
            title = "Extracción"
            body_lines = [body]
        elif body.startswith("I cleaned"):
            title = "Cleaning"
            body_lines = [body]
        elif body.startswith("Limpié") or body.startswith("En lenguaje simple, limpié"):
            title = "Limpieza"
            body_lines = [body]
        elif body.startswith("The models point"):
            title = "Modeling"
            body_lines = [body]
        elif body.startswith("Los modelos apuntan"):
            title = "Modelado"
            body_lines = [body]
        elif body.startswith("Short version:"):
            title = "Orchestrator"
            body_lines = [body]
        elif body.startswith("Versión corta:"):
            title = "Orquestador"
            body_lines = [body]
        elif body.startswith("Legacy"):
            title = "Legacy"
            body_lines = [body]
        elif body.startswith("Legacy habilitado") or body.startswith("Puente legacy"):
            title = "Legacy"
            body_lines = [body]
        elif body.startswith("Source comparison"):
            title = "Comparison"
            body_lines = [body]
        elif body.startswith("Comparación de fuentes"):
            title = "Comparación"
            body_lines = [body]
        elif body.startswith("Market metrics"):
            title = "Market Metrics"
            body_lines = [body]
        elif body.startswith("Métricas de mercado"):
            title = "Métricas de mercado"
            body_lines = [body]
        elif body.startswith("Groq status"):
            title = "Groq Status"
            body_lines = [body]
        elif body.startswith("Estado de Groq"):
            title = "Estado de Groq"
            body_lines = [body]
        elif body.startswith("Session status"):
            title = "Session Status"
            body_lines = [body]
        elif body.startswith("Estado de sesión") or body.startswith("Estado de sesion"):
            title = "Estado de sesión"
            body_lines = [body]
        elif body.startswith("Suggested symbols") or body.startswith("Símbolos sugeridos"):
            title = "Symbols"
            body_lines = [body]
        elif body.startswith("Last symbol used") or body.startswith("Símbolo usado") or body.startswith("Current asset"):
            title = "Asset"
            body_lines = [body]
        elif body.startswith("Clean data hub") or body.startswith("Centro de datos limpios"):
            title = "Clean Data Hub" if body.startswith("Clean data hub") else "Centro de datos limpios"
            body_lines = [body]
        elif body.startswith("Clean market data overview") or body.startswith("Resumen de clean_market_data"):
            title = "Clean Data"
            body_lines = [body]
        elif body.startswith("Clean market data analysis") or body.startswith("Análisis de clean_market_data"):
            title = "Clean Data Analysis" if body.startswith("Clean market data analysis") else "Análisis de datos limpios"
            body_lines = [part.strip() for part in re.split(r"\n+", body) if part.strip()]
        elif body.startswith("Clean market data metrics") or body.startswith("Métricas de clean_market_data"):
            title = "Clean Data Metrics" if body.startswith("Clean market data metrics") else "Métricas de datos limpios"
            body_lines = [body]
        elif body.startswith("The final decision was") or body.startswith("La decisión final fue"):
            title = "Decision"
            body_lines = [body]
        else:
            title = "Run Summary"
            body_lines = [part.strip() for part in re.split(r"\.\s+", body or raw) if part.strip()]
        return _render_card(title, body_lines, enabled=enabled, width=width, color="cyan", subtitle=run_id)

    # Human explanations get a lighter card-like layout.
    if raw.startswith("Market metrics") or raw.startswith("Métricas de mercado"):
        title = "Market Metrics" if raw.startswith("Market metrics") else "Métricas de mercado"
        stage_lines = _render_kv_lines([raw], max(44, width - 6))
        return _render_card(title, stage_lines, enabled=enabled, width=width, color="magenta")
    if raw.startswith("Agent menu") or raw.startswith("Menú de agentes") or raw.startswith("Agent hub") or raw.startswith("Centro de agentes"):
        if raw.startswith("Menú de agentes"):
            title = "Menú de agentes"
        elif raw.startswith("Agent menu"):
            title = "Agent menu"
        elif raw.startswith("Agent hub"):
            title = "Agent hub"
        else:
            title = "Centro de agentes"
        stage_lines = _render_kv_lines([raw], max(44, width - 6))
        return _render_card(title, stage_lines, enabled=enabled, width=width, color="cyan")
    if raw.startswith("Mode hub") or raw.startswith("Centro de modos"):
        title = "Mode hub" if raw.startswith("Mode hub") else "Centro de modos"
        stage_lines = _render_kv_lines([raw], max(44, width - 6))
        return _render_card(title, stage_lines, enabled=enabled, width=width, color="magenta")
    if raw.startswith("Groq status") or raw.startswith("Estado de Groq"):
        title = "Groq Status" if raw.startswith("Groq status") else "Estado de Groq"
        stage_lines = _render_kv_lines([raw], max(44, width - 6))
        return _render_card(title, stage_lines, enabled=enabled, width=width, color="cyan")
    if raw.startswith("Legacy") or raw.startswith("Puente legacy") or raw.startswith("El puente legacy"):
        title = "Legacy" if raw.startswith("Legacy") else "Puente legacy"
        stage_lines = _render_kv_lines([raw], max(44, width - 6))
        return _render_card(title, stage_lines, enabled=enabled, width=width, color="yellow")
    if raw.startswith("Session status") or raw.startswith("Estado de sesión") or raw.startswith("Estado de sesion"):
        title = "Session Status" if raw.startswith("Session status") else "Estado de sesión"
        stage_lines = [part.strip() for part in re.split(r"\.\s+", raw or "") if part.strip()]
        return _render_card(title, stage_lines, enabled=enabled, width=width, color="cyan")
    if raw.startswith("I pulled") or raw.startswith("I cleaned") or raw.startswith("The models point") or raw.startswith("Legacy") or raw.startswith("Source comparison"):
        title = "Extraction" if raw.startswith("I pulled") else "Cleaning" if raw.startswith("I cleaned") else "Modeling" if raw.startswith("The models point") else "Legacy" if raw.startswith("Legacy") else "Comparison"
        stage_lines = _render_kv_lines([body or raw], max(44, width - 6))
        return _render_card(title, stage_lines, enabled=enabled, width=width, color=_label_color("Motor:" if title == "Legacy" else "Decision:"), subtitle=run_id)

    if raw.startswith("Extraje") or raw.startswith("En lenguaje simple, extraje") or raw.startswith("Limpié") or raw.startswith("En lenguaje simple, limpié") or raw.startswith("Los modelos apuntan") or raw.startswith("Versión corta:") or raw.startswith("Símbolos sugeridos") or raw.startswith("Métricas de clean_market_data"):
        title = (
            "Extraction" if raw.startswith("Extraje") or raw.startswith("En lenguaje simple, extraje")
            else "Cleaning" if raw.startswith("Limpié") or raw.startswith("En lenguaje simple, limpié")
            else "Modeling" if raw.startswith("Los modelos apuntan")
            else "Orchestrator" if raw.startswith("Versión corta:")
            else "Symbols" if raw.startswith("Símbolos sugeridos")
            else "Clean Data Metrics"
        )
        stage_lines = _render_kv_lines([body or raw], max(44, width - 6))
        return _render_card(title, stage_lines, enabled=enabled, width=width, color="cyan", subtitle=run_id)

    # Fallback: wrapped plain text.
    return _render_card("Assistant", _render_kv_lines([raw], max(44, width - 6)), enabled=enabled, width=width, color="white")


def _print_assistant_response(text: str, *, no_color: bool) -> None:
    width = max(72, shutil.get_terminal_size((100, 20)).columns)
    enabled = _use_color(no_color)
    print(_paint("Assistant", "cyan", enabled, bold=True))
    print(_format_assistant_output(text, enabled, width))


def main() -> int:
    args = build_parser().parse_args()
    runtime = AssistantRuntime(artifact_root=args.artifact_root, session_id=args.session_id)

    if args.message:
        try:
            _print_assistant_response(runtime.ask(args.message), no_color=args.no_color)
        except Exception as exc:  # noqa: BLE001
            print(f"Error: {exc}")
            return 1
        return 0

    session_state = runtime.get_state()
    print(
        _banner(
            _use_color(args.no_color),
            shutil.get_terminal_size((100, 20)).columns,
            session_state.preferred_language,
            session_state.current_mode,
            _format_session_context_line(session_state.last_summary or {}, session_state, session_state.preferred_language or "en"),
        )
    )
    while True:
        try:
            prompt = _paint("> ", "green", _use_color(args.no_color), bold=True)
            message = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not message:
            continue
        if message.lower() in {"exit", "quit", "q"}:
            break
        try:
            _print_assistant_response(runtime.ask(message), no_color=args.no_color)
        except Exception as exc:  # noqa: BLE001
            print(_paint("Error", "red", _use_color(args.no_color), bold=True) + f": {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
