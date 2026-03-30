from __future__ import annotations

import calendar
import json
import logging
import os
import re
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

from assistant.contracts import AssistantRoute, AssistantState
from providers.groq_payloads import compact_for_groq

logger = logging.getLogger(__name__)

_MONTH_ALIASES: dict[str, int] = {
    "jan": 1,
    "january": 1,
    "ene": 1,
    "enero": 1,
    "feb": 2,
    "february": 2,
    "febrero": 2,
    "mar": 3,
    "march": 3,
    "marzo": 3,
    "apr": 4,
    "april": 4,
    "abril": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "junio": 6,
    "jul": 7,
    "july": 7,
    "julio": 7,
    "aug": 8,
    "august": 8,
    "agosto": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "setiembre": 9,
    "oct": 10,
    "october": 10,
    "octubre": 10,
    "nov": 11,
    "november": 11,
    "noviembre": 11,
    "dec": 12,
    "december": 12,
    "diciembre": 12,
}

_RUN_INTENTS = {
    "run_pipeline",
    "compare_sources",
}

_SHOW_INTENTS = {
    "show_semantic_lookup",
    "show_source_scope",
    "show_time_status",
    "show_latest_summary",
    "show_extraction",
    "show_cleaning",
    "show_clean_data",
    "show_market_metrics",
    "show_market_type",
    "show_prediction",
    "show_model_variables",
    "show_legacy_status",
    "show_symbol_guide",
    "show_asset_used",
    "show_decision_explanation",
    "show_stage_brief",
    "show_groq_status",
    "show_web_status",
    "show_session_status",
    "show_session_status_full",
    "show_mode_guide",
    "show_agent_guide",
    "show_agent_card",
}

_MENU_SHORTCUTS: dict[str, dict[str, str]] = {
    "1": {"intent": "show_agent_guide"},
    "2": {"intent": "show_agent_card", "stage": "extraction"},
    "3": {"intent": "show_agent_card", "stage": "cleaning"},
    "4": {"intent": "show_agent_card", "stage": "modeling"},
    "5": {"intent": "show_stage_brief", "stage": "orchestrator"},
    "6": {"intent": "show_session_status", "stage": "session_status"},
    "7": {"intent": "show_session_status_full", "stage": "session_status_full"},
    "8": {"intent": "show_clean_data", "stage": "clean_data"},
    "9": {"intent": "show_mode_guide", "stage": "mode_guide"},
    "a": {"intent": "show_agent_guide"},
    "e": {"intent": "show_agent_card", "stage": "extraction"},
    "c": {"intent": "show_agent_card", "stage": "cleaning"},
    "m": {"intent": "show_agent_card", "stage": "modeling"},
    "o": {"intent": "show_stage_brief", "stage": "orchestrator"},
    "s": {"intent": "show_session_status", "stage": "session_status"},
    "f": {"intent": "show_session_status_full", "stage": "session_status_full"},
    "d": {"intent": "show_clean_data", "stage": "clean_data"},
    "g": {"intent": "show_mode_guide", "stage": "mode_guide"},
    "n": {"intent": "set_language", "language": "en"},
    "b": {"intent": "set_language", "language": "es"},
    "p": {"intent": "set_language", "language": "es"},
    "h": {"intent": "help", "stage": "help"},
    "extract": {"intent": "show_agent_card", "stage": "extraction"},
    "clean": {"intent": "show_agent_card", "stage": "cleaning"},
    "model": {"intent": "show_agent_card", "stage": "modeling"},
    "agents": {"intent": "show_agent_guide"},
    "agentes": {"intent": "show_agent_guide"},
    "extraction": {"intent": "show_agent_card", "stage": "extraction"},
    "extraccion": {"intent": "show_agent_card", "stage": "extraction"},
    "extracción": {"intent": "show_agent_card", "stage": "extraction"},
    "cleaning": {"intent": "show_agent_card", "stage": "cleaning"},
    "limpieza": {"intent": "show_agent_card", "stage": "cleaning"},
    "modeling": {"intent": "show_agent_card", "stage": "modeling"},
    "modelado": {"intent": "show_agent_card", "stage": "modeling"},
    "orchestrator": {"intent": "show_stage_brief", "stage": "orchestrator"},
    "orquestador": {"intent": "show_stage_brief", "stage": "orchestrator"},
    "metrics": {"intent": "show_clean_data", "stage": "clean_data"},
    "metricas": {"intent": "show_clean_data", "stage": "clean_data"},
    "métricas": {"intent": "show_clean_data", "stage": "clean_data"},
    "metric": {"intent": "show_clean_data", "stage": "clean_data"},
    "fila": {"intent": "show_clean_data", "stage": "clean_data"},
    "filas": {"intent": "show_clean_data", "stage": "clean_data"},
    "row": {"intent": "show_clean_data", "stage": "clean_data"},
    "rows": {"intent": "show_clean_data", "stage": "clean_data"},
    "schema": {"intent": "show_clean_data", "stage": "clean_data"},
    "esquema": {"intent": "show_clean_data", "stage": "clean_data"},
    "status": {"intent": "show_session_status", "stage": "session_status"},
    "estado": {"intent": "show_session_status", "stage": "session_status"},
    "status full": {"intent": "show_session_status_full", "stage": "session_status_full"},
    "estado completo": {"intent": "show_session_status_full", "stage": "session_status_full"},
    "continue": {"intent": "continue_task", "stage": "continue"},
    "continuar": {"intent": "continue_task", "stage": "continue"},
    "sigue": {"intent": "continue_task", "stage": "continue"},
    "seguir": {"intent": "continue_task", "stage": "continue"},
    "clean data": {"intent": "show_clean_data", "stage": "clean_data"},
    "datos limpios": {"intent": "show_clean_data", "stage": "clean_data"},
    "volume": {"intent": "show_market_metrics", "stage": "metrics"},
    "volumen": {"intent": "show_market_metrics", "stage": "metrics"},
    "volatility": {"intent": "show_market_metrics", "stage": "metrics"},
    "volatilidad": {"intent": "show_market_metrics", "stage": "metrics"},
    "symbols": {"intent": "show_symbol_guide", "stage": "symbols"},
    "símbolos": {"intent": "show_symbol_guide", "stage": "symbols"},
    "simbolos": {"intent": "show_symbol_guide", "stage": "symbols"},
    "modes": {"intent": "show_mode_guide", "stage": "mode_guide"},
    "mode": {"intent": "show_mode_guide", "stage": "mode_guide"},
    "modos": {"intent": "show_mode_guide", "stage": "mode_guide"},
    "modo": {"intent": "show_mode_guide", "stage": "mode_guide"},
    "local_only": {"intent": "show_mode_guide", "stage": "local_only"},
    "local only": {"intent": "show_mode_guide", "stage": "local_only"},
    "local_plus_reviewer": {"intent": "show_mode_guide", "stage": "local_plus_reviewer"},
    "local_plus_binance": {"intent": "show_mode_guide", "stage": "local_plus_binance"},
    "local_plus_binance_legacy": {"intent": "show_mode_guide", "stage": "local_plus_binance_legacy"},
    "local_only_groq_brain": {"intent": "show_mode_guide", "stage": "local_only_groq_brain"},
    "local_plus_binance_groq_brain": {"intent": "show_mode_guide", "stage": "local_plus_binance_groq_brain"},
    "local_plus_binance_legacy_groq_brain": {"intent": "show_mode_guide", "stage": "local_plus_binance_legacy_groq_brain"},
    "--groq-brain": {"intent": "show_mode_guide", "stage": "groq_brain"},
    "groq brain": {"intent": "show_mode_guide", "stage": "groq_brain"},
    "compare-binance": {"intent": "show_mode_guide", "stage": "compare_binance"},
    "compare-binance + --groq-brain": {"intent": "show_mode_guide", "stage": "compare_binance_groq_brain"},
    "help": {"intent": "help", "stage": "help"},
    "ayuda": {"intent": "help", "stage": "help"},
}

_FOLLOW_UP_HINTS = (
    "continue",
    "same",
    "that",
    "this",
    "it",
    "latest",
    "previous",
    "again",
    "that one",
    "ese",
    "esa",
    "eso",
    "continuar",
    "sigue",
)

_GREETING_PATTERN = re.compile(
    r"^(?P<greeting>"
    r"hi|hello|hey|hola|hola\s+assistant|hello\s+assistant|hey\s+assistant|"
    r"buenas|buenos\s+d[ií]as|buenas\s+tardes|buenas\s+noches|"
    r"good\s+morning|good\s+afternoon|good\s+evening"
    r")(?P<rest>[\s,!.:;-].*)?$",
    re.IGNORECASE,
)

_CRYPTO_FAMILY_ALIASES = {
    "BTC",
    "ETH",
    "LTC",
    "XBT",
    "DOGE",
    "SOL",
    "ADA",
    "XRP",
    "BCH",
    "DOT",
    "AVAX",
    "MATIC",
    "LINK",
    "UNI",
}


def _default_date_range() -> tuple[date, date]:
    end = date.today()
    start = date.fromordinal(max(1, end.toordinal() - 365))
    return start, end


def _split_greeting_prefix(message: str) -> tuple[bool, str]:
    normalized = _normalize_whitespace(message)
    if not normalized:
        return False, normalized
    match = _GREETING_PATTERN.match(normalized)
    if not match:
        return False, normalized
    rest = _normalize_whitespace(match.group("rest") or "")
    return True, rest


def _default_ticker_for_message(message: str) -> Optional[str]:
    lowered = (message or "").lower()
    if any(hint in lowered for hint in ("forex", "fx", "currency", "exchange rate", "forex active")):
        return "EURUSD=X"
    if any(hint in lowered for hint in ("etf", "fund", "tracker")):
        return "SPY"
    if any(hint in lowered for hint in ("crypto", "bitcoin", "btc", "eth", "coin", "token", "crypto active")):
        return "BTC-USD"
    if any(
        hint in lowered
        for hint in (
            "stock",
            "stocks",
            "share",
            "shares",
            "equity",
            "action",
            "actions",
            "test",
            "tests",
            "demo",
            "select it you",
            "you choose",
            "make a test",
            "make tests",
            "not matter when",
            "whatever",
            "pick one",
            "surprise me",
            "any time",
            "any ticker",
            "just make a test",
        )
    ):
        return "AAPL"
    return None


def _should_default_dates(message: str) -> bool:
    lowered = (message or "").lower()
    return any(
        hint in lowered
        for hint in (
            "test",
            "tests",
            "demo",
            "select it you",
            "you choose",
            "make a test",
            "make tests",
            "not matter when",
            "whatever",
            "pick one",
            "surprise me",
            "any time",
            "any ticker",
            "just make a test",
            "forex active",
            "crypto active",
            "price of an forex active",
            "price of a stock",
        )
    )


def _normalize_whitespace(text: str) -> str:
    return " ".join((text or "").strip().split())


def _detect_language(text: str) -> str:
    lowered = (text or "").lower()
    spanish_hints = sum(
        1
        for hint in (
            "hola",
            "quien",
            "quién",
            "donde",
            "dónde",
            "buenas",
            "buenos días",
            "buenos dias",
            "buenas tardes",
            "buenas noches",
            "resumen",
            "ejecucion",
            "ejecución",
            "modelo",
            "revisor",
            "limpiar",
            "extraccion",
            "extracción",
            "comparacion",
            "comparación",
            "legado",
            "continuar",
            "muestra",
            "dime",
            "qué",
            "que",
            "cómo",
            "como",
            "hoy",
            "día",
            "dia",
            "fecha",
            "hora",
            "información",
            "informacion",
            "fuente",
            "fuentes",
            "buscar",
            "proviene",
            "viene",
            "pasa",
            "activo",
            "simbolo",
            "símbolo",
            "analisis",
            "análisis",
            "mercado",
        )
        if hint in lowered
    )
    english_hints = sum(
        1
        for hint in (
            "hello",
            "hi",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
            "summary",
            "run",
            "model",
            "reviewer",
            "clean",
            "extract",
            "comparison",
            "legacy",
            "continue",
            "latest",
        )
        if hint in lowered
    )
    return "es" if spanish_hints > english_hints else "en"


def _infer_requested_language(text: str) -> Optional[str]:
    lowered = (text or "").lower()
    if any(
        hint in lowered
        for hint in (
            "in spanish",
            "answer in spanish",
            "respond in spanish",
            "reply in spanish",
            "translate to spanish",
            "traduce al español",
            "traducir al español",
            "español",
            "espanol",
        )
    ):
        return "es"
    if any(
        hint in lowered
        for hint in (
            "in english",
            "answer in english",
            "respond in english",
            "reply in english",
            "translate to english",
            "traduce al inglés",
            "traducir al inglés",
            "ingles",
            "inglés",
        )
    ):
        return "en"
    if any(hint in lowered for hint in ("translate", "traduce", "traducir")):
        # Default to the opposite of the detected language when the user asks to translate,
        # so "translate this" still produces a visible transformation.
        detected = _detect_language(text)
        return "es" if detected == "en" else "en"
    return None


def _infer_language_switch(text: str) -> Optional[str]:
    lowered = (text or "").lower()
    if any(
        hint in lowered
        for hint in (
            "switch to spanish",
            "set language to spanish",
            "change language to spanish",
            "talk in spanish",
            "speak spanish",
            "habla en español",
            "hablar en español",
            "ponlo en español",
            "cambia a español",
            "español",
            "espanol",
        )
    ):
        return "es"
    if any(
        hint in lowered
        for hint in (
            "switch to english",
            "set language to english",
            "change language to english",
            "talk in english",
            "speak english",
            "habla en inglés",
            "hablar en inglés",
            "ponlo en inglés",
            "cambia a inglés",
            "ingles",
            "inglés",
        )
    ):
        return "en"
    return None


def _looks_like_explanation_request(text: str) -> bool:
    lowered = (text or "").lower()
    return any(
        hint in lowered
        for hint in (
            "share me",
            "share it",
            "in your words",
            "your words",
            "plain language",
            "simple language",
            "plain words",
            "simple words",
            "what happened",
            "why did",
            "why is",
            "explain",
            "tell me",
            "toll me",
            "what say groq",
            "what does groq say",
            "what groq says",
            "what does groq mean",
            "que dice groq",
            "qué dice groq",
            "groq here",
            "groq aca",
            "groq aquí",
            "groq aqui",
            "walk me through",
            "describe",
            "what mean",
            "what does it mean",
            "meaning",
            "what can you tell me",
            "what can u tell me",
            "what symbols can i use",
            "what symbols can u use",
            "supported symbols",
            "supported tickers",
        )
    )


def _looks_like_identity_request(text: str) -> bool:
    lowered = _normalize_whitespace(text).lower()
    if not lowered:
        return False
    return any(
        hint in lowered
        for hint in (
            "who are you",
            "who r u",
            "who is this",
            "what are you",
            "what is your name",
            "what's your name",
            "what can you do",
            "what can u do",
            "what are you capable of",
            "what are your capabilities",
            "what are your skills",
            "what is your role",
            "what's your role",
            "what is your role in this session",
            "what's your role in this session",
            "what do you do",
            "what do u do",
            "who am i talking to",
            "introduce yourself",
            "present yourself",
            "tell me about yourself",
            "about yourself",
            "quien eres",
            "quién eres",
            "que eres",
            "qué eres",
            "como te llamas",
            "cómo te llamas",
            "presentate",
            "preséntate",
            "presenta te",
        )
    )


def _looks_like_extraction_report_request(text: str) -> bool:
    lowered = re.sub(r"[?.!,;:]+$", "", _normalize_whitespace(text).lower()).strip()
    if not lowered:
        return False
    return any(
        token in lowered
        for token in (
            "what columns were extracted",
            "which columns were extracted",
            "what happened in extraction",
            "columns were extracted",
            "raw columns",
            "missing columns",
            "pulled rows",
            "rows were pulled",
            "data source",
            "timeframe",
            "interval",
            "columnas extraidas",
            "columnas extraídas",
            "columnas crudas",
            "columnas faltantes",
            "qué columnas se extrajeron",
            "que columnas se extrajeron",
            "qué pasó en la extracción",
            "que pasó en la extracción",
            "que paso en la extracción",
            "qué pasó en extraccion",
            "que pasó en extraccion",
            "qué pasó en extracción",
            "que pasó en extracción",
        )
    )


def _looks_like_clean_row_analysis_request(text: str) -> bool:
    lowered = (text or "").lower()
    if not any(
        hint in lowered
        for hint in (
            "analysis",
            "analyze",
            "analyse",
            "analize",
            "analiza",
            "analizar",
            "interpret",
            "interpretation",
            "break down",
            "row analysis",
            "clean row analysis",
            "cleaned row analysis",
            "what does this row mean",
            "how to read this row",
            "how do i read this row",
            "explain this row",
        "analiza la fila",
        "analiza la fila limpia",
        "analiza esta fila",
        "explica esta fila",
        "análisis de fila",
        "analisis de fila",
        "análisis de la fila",
        "analisis de la fila",
    )
    ):
        return False
    return any(
        hint in lowered
        for hint in (
            "clean",
            "cleaned",
            "clean row",
            "cleaned row",
            "clean market data",
            "cleaned data",
            "datos limpios",
            "clean_market_data",
            "row",
            "fila",
            "registro",
        )
    )


def _looks_like_clean_data_request(text: str) -> bool:
    lowered = re.sub(r"[?.!,;:]+$", "", _normalize_whitespace(text).lower()).strip()
    if not lowered:
        return False
    if re.search(
        r"\b(what is|what's|qué es|que es)\s+(?:(?:the|a|an|el|la|un|una)\s+)?(row|fila|schema|esquema|raw column|clean column|model variable)\b",
        lowered,
    ):
        return False
    if lowered in {
        "metricas",
        "métricas",
        "metrics",
        "metric",
        "fila",
        "filas",
        "row",
        "rows",
        "schema",
        "esquema",
        "clean data",
        "cleaned data",
        "clean_market_data",
        "datos limpios",
        "datos de limpieza",
        "symbols",
        "símbolos",
        "simbolos",
        "volume",
        "volumen",
        "volatility",
        "volatilidad",
    }:
        return True
    return any(
        hint in lowered
        for hint in (
            "what symbols are in the cleaned data",
            "symbols are in the cleaned data",
            "what metrics are in the cleaned data",
            "metrics are in the cleaned data",
            "what metrics does it contain",
            "what metrics are contained",
            "what metrics are included",
            "what is the clean data schema",
            "clean data schema",
            "clean data structure",
            "cleaned data structure",
            "clean data columns",
            "cleaned data columns",
            "what rows are in the cleaned data",
            "latest cleaned rows",
            "what does this clean row say",
            "what does this row say",
            "show me the clean row",
            "muestra la fila limpia",
            "qué dice esta fila",
            "que dice esta fila",
            "qué es esta fila",
            "que es esta fila",
            "qué métricas hay",
            "que metricas hay",
            "qué métricas contiene",
            "que metricas contiene",
            "qué métricas tiene",
            "que metricas tiene",
            "qué símbolos hay",
            "que simbolos hay",
            "qué simbolos hay",
        )
    )


def _looks_like_market_metrics_request(text: str) -> bool:
    lowered = (text or "").lower()
    if not any(hint in lowered for hint in ("volume", "volatility", "volatilidad", "volumen")):
        return False
    return any(
        hint in lowered
        for hint in (
            "cuál es",
            "cual es",
            "qué es",
            "que es",
            "cuánto es",
            "cuanto es",
            "volumen de",
            "volatilidad de",
            "volumen del",
            "volatilidad del",
            "dame",
            "muestra",
            "what is",
            "what's",
            "how much",
            "how volatile",
            "market",
            "clean",
            "data",
            "symbol",
            "ticker",
            "asset",
            "activo",
            "acción",
            "accion",
            "forex",
            "stock",
            "crypto",
            "etf",
        )
    ) or bool(_ticker_candidates(text))


def _looks_like_model_variables_request(text: str) -> bool:
    lowered = _normalize_whitespace(text).lower()
    if not lowered:
        return False
    variable_hints = (
        "variables usada",
        "variables usadas",
        "variables used",
        "variables explicativas",
        "explica las variables",
        "explain the variables",
        "explica las features",
        "explain the features",
        "explanatory variables",
        "model variables",
        "model inputs",
        "input variables",
        "inputs used",
        "features used",
        "features were used",
        "feature columns",
        "feature column",
        "what features",
        "which features",
        "what variables",
        "which variables",
        "qué variables",
        "que variables",
        "qué features",
        "que features",
        "qué columnas de modelo",
        "que columnas de modelo",
        "used to train",
        "before training",
        "antes de entrenar",
        "transformación",
        "transformacion",
        "transformaciones",
        "transformation",
        "transformations",
        "preprocessing",
        "preprocess",
        "target_direction",
    )
    context_hints = (
        "model",
        "modeling",
        "modelado",
        "prediction",
        "predijo",
        "predicted",
        "cleaning",
        "limpieza",
        "feature",
        "features",
        "variable",
        "variables",
        "columna",
        "columnas",
        "transform",
        "transforma",
        "used",
        "utilizaste",
        "usaste",
        "usa",
        "usó",
        "uso",
    )
    return any(hint in lowered for hint in variable_hints) and any(hint in lowered for hint in context_hints)


def _looks_like_asset_used_request(text: str) -> bool:
    lowered = _normalize_whitespace(text).lower()
    if not lowered:
        return False
    return any(
        hint in lowered
        for hint in (
            "what symbol was used",
            "which symbol was used",
            "what ticker was used",
            "which ticker was used",
            "what asset was used",
            "which asset was used",
            "what asset did the last run use",
            "last symbol used",
            "last ticker used",
            "selected symbol",
            "selected ticker",
            "used symbol",
            "used ticker",
            "used asset",
            "symbol used",
            "ticker used",
            "asset used",
            "que simbolo fue usado",
            "qué símbolo fue usado",
            "qué símbolo se usó",
            "que símbolo se usó",
            "que ticker fue usado",
            "qué ticker fue usado",
            "qué ticker se usó",
            "que ticker se usó",
            "que activo fue usado",
            "qué activo fue usado",
            "qué activo se usó",
            "que activo se usó",
            "ultimo simbolo usado",
            "último símbolo usado",
            "último simbolo usado",
            "ultimo ticker usado",
            "último ticker usado",
            "activo usado",
            "simbolo usado",
            "símbolo usado",
            "ticker usado",
            "activo para analisis",
            "activo para análisis",
            "activo del analisis",
            "activo del análisis",
        )
    )


def _looks_like_market_type_request(text: str) -> bool:
    lowered = _normalize_whitespace(text).lower()
    if not lowered:
        return False
    if re.search(r"\b(market type|tipo de mercado|asset class|asset type|clase de activo|tipo de activo)\b", lowered):
        return True
    return any(
        phrase in lowered
        for phrase in (
            "what market does",
            "which market does",
            "belong to market",
            "belongs to the market",
            "de que mercado hace parte",
            "de qué mercado hace parte",
            "a que mercado pertenece",
            "a qué mercado pertenece",
            "de que tipo de mercado hace parte",
            "de qué tipo de mercado hace parte",
        )
    )


def _looks_like_groq_status_request(text: str) -> bool:
    lowered = (text or "").lower()
    return any(
        hint in lowered
        for hint in (
            "groq status",
            "groq url",
            "groq endpoint",
            "groq base url",
            "groq model",
            "is groq active",
            "is groq used",
            "confirm groq",
            "check groq",
            "show groq config",
            "what url does groq use",
            "what model does groq use",
            "estado de groq",
            "url de groq",
            "endpoint de groq",
            "modelo de groq",
            "groq activo",
            "groq en uso",
        )
    )


def _looks_like_web_status_request(text: str) -> bool:
    lowered = (text or "").lower()
    provider_mentioned = bool(re.search(r"\b(searxng|serper|tavily|searchapi)\b", lowered))
    setup_mentioned = bool(re.search(r"\b(env|setup|snippet|config|configuration|configuracion|configuración)\b", lowered))
    activation_mentioned = bool(
        re.search(r"\b(use|activate|enable|connect|configure|configura|activar|activa|habilita|setup|try|usa)\b", lowered)
    )
    provider_catalog_mentioned = bool(
        re.search(r"\b(provider|providers|preset|presets|proveedor|proveedores)\b", lowered)
        and re.search(r"\b(web|internet|search|retriever|supported|support|soporta|soporta|which|what)\b", lowered)
    )
    if provider_mentioned and setup_mentioned:
        return True
    if provider_mentioned and activation_mentioned:
        return True
    if provider_catalog_mentioned:
        return True
    return any(
        hint in lowered
        for hint in (
            "web status",
            "internet status",
            "retriever status",
            "search status",
            "web config",
            "internet config",
            "search config",
            "web retriever",
            "web search",
            "is the web active",
            "is internet active",
            "is web retrieval active",
            "estado web",
            "estado de internet",
            "estado del retriever",
            "estado de web",
            "config de web",
            "config de internet",
            "config del retriever",
            "retriever web",
            "busqueda web",
            "búsqueda web",
            "internet activo",
            "web activa",
            "web activo",
        )
    )


def _looks_like_time_status_request(text: str) -> bool:
    lowered = _normalize_whitespace(text).lower()
    if not lowered:
        return False
    return any(
        hint in lowered
        for hint in (
            "what day is it",
            "what date is it",
            "what time is it",
            "what day is today",
            "what date is today",
            "today's date",
            "current date",
            "current time",
            "fecha correcta",
            "fecha actual",
            "hora actual",
            "qué día es",
            "que día es",
            "qué dia es",
            "que dia es",
            "qué fecha es",
            "que fecha es",
            "qué hora es",
            "que hora es",
            "qué día es hoy",
            "que día es hoy",
            "qué dia es hoy",
            "que dia es hoy",
        )
    )


def _looks_like_source_scope_request(text: str) -> bool:
    lowered = _normalize_whitespace(text).lower()
    if not lowered:
        return False
    return any(
        hint in lowered
        for hint in (
            "where does your information come from",
            "where does your info come from",
            "where do your data come from",
            "what sources do you use",
            "what source do you use",
            "what can you search",
            "what can u search",
            "what can you look up",
            "what information can you search",
            "what information can you use",
            "qué puedes buscar",
            "que puedes buscar",
            "qué información puedes buscar",
            "que información puedes buscar",
            "qué informacion puedes buscar",
            "que informacion puedes buscar",
            "de dónde proviene tu información",
            "de donde proviene tu informacion",
            "de dónde viene tu información",
            "de donde viene tu informacion",
            "de dónde vienen tus datos",
            "de donde vienen tus datos",
            "qué fuentes usas",
            "que fuentes usas",
            "qué fuentes utilizas",
            "que fuentes utilizas",
        )
    )


def _looks_like_semantic_lookup_request(text: str) -> bool:
    lowered = (text or "").lower()
    if re.search(
        r"\b(what is|what's|qué es|que es)\s+(?:(?:the|a|an|el|la|un|una)\s+)?(volatility|volatilidad|risk|riesgo|return|retorno|rendimiento|alpha|beta|correlation|correlación|correlacion|leverage|apalancamiento|arbitrage|hedge|spread|liquidity|drawdown|momentum|mean reversion|slippage|hold|long|short|ticker|benchmark|portfolio|position|exposure|factor|signal|trend|variance|standard deviation|sharpe ratio|sortino ratio|covariance|pnl|profit and loss|excess return|benchmark return|tracking error|information ratio|dataset|artifact|manifest|target|feature engineering|grounding|memory|artifact store|evidence ledger|drift|conversational layer|validator|shadow run|promotion gate|challenger|champion|policy engine|retraining scheduler|feature registry|adaptive selector|shadow runner|promotion policy|row|fila|schema|esquema|raw column|clean column|model variable|column|columns|columna|columnas|variable|variables|feature|features|adj_close|adjusted close)\b(?!\s+(?:of|for|de|para|in|en)\b)",
        lowered,
    ):
        return True
    exclusions = (
        "volume",
        "volumen",
        "volatility",
        "volatilidad",
        "open",
        "high",
        "low",
        "close",
        "precio",
        "price",
        "metric",
        "metrics",
        "métrica",
        "métricas",
        "metrica",
        "metricas",
        "row",
        "rows",
        "fila",
        "filas",
        "prediction",
        "predicción",
        "prediccion",
        "decision",
        "decisión",
        "run_",
        "summary",
        "status",
        "result",
        "report",
        "latest",
        "current",
        "today",
        "now",
        "schema",
        "clean",
        "cleaned",
        "cleaning",
        "data",
    )
    if any(token in lowered for token in exclusions):
        return False
    if "binance" in lowered:
        return False
    if re.search(
        r"\b(what is the difference between|what's the difference between|difference between|diferencia entre|que diferencia hay entre|qué diferencia hay entre|compare|compara)\b",
        lowered,
    ):
        return True
    if re.search(
        r"\b(what does .+ mean|qué significa .+|que significa .+|define|definition|definición|definicion|meaning|significado)\b",
        lowered,
    ):
        return True
    if re.fullmatch(
        r"(?:(?:the|a|an|el|la|un|una)\s+)?(?:volatility|volatilidad|risk|riesgo|return|retorno|rendimiento|alpha|beta|correlation|correlación|correlacion|leverage|apalancamiento|arbitrage|hedge|spread|liquidity|drawdown|momentum|mean reversion|slippage|hold|long|short|ticker|yfinance|benchmark|portfolio|position|exposure|factor|signal|trend|variance|standard deviation|sharpe ratio|sortino ratio|covariance|pnl|profit and loss|excess return|benchmark return|tracking error|information ratio|dataset|artifact|manifest|target|feature engineering|grounding|memory|artifact store|evidence ledger|drift|conversational layer|validator|shadow run|promotion gate|challenger|champion|policy engine|retraining scheduler|feature registry|adaptive selector|shadow runner|promotion policy|forex|crypto|equity|fund|index|stock|etf|currency pair|digital asset pair|market index|column|columns|columna|columnas|variable|variables|feature|features|adj_close|adjusted close)",
        lowered,
    ):
        return True
    if not re.search(r"\b(what is|what's|qué es|que es)\b", lowered):
        return False
    return True


def _looks_like_assistant_scorecard_request(text: str) -> bool:
    lowered = (text or "").lower()
    if re.search(r"\b(scorecard|architecture score|architecture status|maturity)\b", lowered):
        return True
    if re.search(r"\b(porcentaje de arquitectura|scorecard del assistant|scorecard de arquitectura|madurez de arquitectura)\b", lowered):
        return True
    if re.search(r"\b(web|internet|retriever)\b", lowered) and re.search(
        r"\b(change|changes|impact|improve|improves|improvement|if i enable|if i configure|si configuro|si activo|qué cambia|que cambia|impacto)\b",
        lowered,
    ):
        return True
    if re.search(r"\b(hybrid runtime|runtime hybrid|híbrido|hibrido)\b", lowered) and re.search(
        r"\b(missing|left|gap|falt[ae]|cu[aá]nto falta|que falta|qué falta)\b",
        lowered,
    ):
        return True
    if re.search(r"\b(local-first|assistant)\b", lowered) and re.search(
        r"\b(score|scorecard|readiness|ready|listo|estado)\b",
        lowered,
    ):
        return True
    return False


def _looks_like_session_status_request(text: str) -> bool:
    lowered = (text or "").lower()
    if "legacy" in lowered:
        return False
    if re.search(r"\bestado\b", lowered):
        return True
    return any(
        hint in lowered
        for hint in (
            "status",
            "session status",
            "mode status",
            "current mode",
            "active mode",
            "what mode am i in",
            "which mode am i in",
            "am i in groq brain",
            "groq brain active",
            "groq brain enabled",
            "brain status",
            "brain enabled",
            "brain active",
            "confidence",
            "how confident",
            "decision source",
            "source of decision",
            "estado de sesión",
            "estado de sesion",
            "modo activo",
            "modo actual",
            "confianza",
            "fuente de decisión",
            "fuente de decision",
        )
    )


def _looks_like_session_status_full_request(text: str) -> bool:
    lowered = re.sub(r"[?.!,;:]+$", "", _normalize_whitespace(text).lower()).strip()
    if "legacy" in lowered:
        return False
    return any(
        hint in lowered
        for hint in (
            "status full",
            "full status",
            "full session status",
            "session status full",
            "mode status full",
            "status detailed",
            "detailed status",
            "status detallado",
            "status completo",
            "estado completo",
            "estado de sesión completo",
            "estado de sesion completo",
            "estado detallado",
        )
    )


def _looks_like_latest_summary_request(text: str) -> bool:
    lowered = re.sub(r"[?.!,;:]+$", "", _normalize_whitespace(text).lower()).strip()
    explicit_phrases = (
        "latest run summary",
        "summarize latest run",
        "summarize the latest run",
        "summary of latest run",
        "latest summary",
        "show the last run",
        "last run",
        "show last run",
        "show me the last run",
        "show latest run",
        "resumen de la última corrida",
        "resumen de la ultima corrida",
        "última corrida",
        "ultima corrida",
    )
    loose_current_phrases = (
        "what's happening today",
        "what is happening today",
        "what is going on today",
        "how is it going today",
        "how's it going today",
        "what happened today",
        "que pasa hoy",
        "qué pasa hoy",
        "que paso hoy",
        "qué pasó hoy",
        "como va hoy",
        "cómo va hoy",
        "que tal va hoy",
        "qué tal va hoy",
        "how is it going",
        "how's it going",
        "como va",
        "cómo va",
        "que paso",
        "qué pasó",
    )
    if any(phrase in lowered for phrase in explicit_phrases):
        return True
    if any(phrase in lowered for phrase in loose_current_phrases):
        current_like = True
    else:
        current_like = False
    has_latest = any(hint in lowered for hint in ("latest", "last", "última", "ultima"))
    has_current = any(hint in lowered for hint in ("today", "hoy", "current", "actual", "now", "ahora"))
    if not current_like and not (has_latest and has_current):
        return False
    if any(
        hint in lowered
        for hint in (
            "extraction",
            "extract",
            "extracción",
            "extraccion",
            "clean",
            "cleaning",
            "limpieza",
            "model",
            "modeling",
            "modelado",
            "variables",
            "features",
            "metric",
            "metrics",
            "métrica",
            "metrica",
            "row",
            "fila",
            "symbol",
            "ticker",
            "asset",
            "activo",
            "market type",
            "tipo de mercado",
            "decision",
            "predic",
            "status",
            "estado",
            "mode",
            "modo",
            "compare",
            "compar",
        )
    ):
        return False
    return True


def _infer_agent_access_stage(text: str) -> Optional[str]:
    lowered = (text or "").lower()
    stage_patterns = {
        "extraction": (
            "ask extraction",
            "open extraction",
            "access extraction",
            "talk to extraction",
            "go to extraction",
            "use extraction",
            "use the extraction",
            "use extraction agent",
            "extraction agent",
            "agente de extracción",
            "agente extraction",
            "usar extracción",
            "usar la extracción",
            "usar el agente de extracción",
            "pregunta extracción",
            "pregunta extraction",
        ),
        "cleaning": (
            "ask cleaning",
            "open cleaning",
            "access cleaning",
            "talk to cleaning",
            "go to cleaning",
            "use cleaning",
            "use the cleaning",
            "use cleaning agent",
            "cleaning agent",
            "agente de limpieza",
            "agente cleaning",
            "usar limpieza",
            "usar la limpieza",
            "usar el agente de limpieza",
            "pregunta limpieza",
            "pregunta cleaning",
        ),
        "modeling": (
            "ask modeling",
            "open modeling",
            "access modeling",
            "talk to modeling",
            "go to modeling",
            "use modeling",
            "use the modeling",
            "use modeling agent",
            "modeling agent",
            "agente de modelado",
            "agente modeling",
            "usar modelado",
            "usar el modelado",
            "usar el agente de modelado",
            "pregunta modelado",
            "pregunta modeling",
        ),
        "orchestrator": (
            "ask orchestrator",
            "open orchestrator",
            "access orchestrator",
            "orchestrator",
            "orquestador",
            "talk to orchestrator",
            "go to orchestrator",
            "use orchestrator",
            "use the orchestrator",
            "use orchestrator agent",
            "orchestrator agent",
            "agente del orquestador",
            "agente orchestrator",
            "usar orquestador",
            "usar el orquestador",
            "usar el agente del orquestador",
            "pregunta orquestador",
            "pregunta orchestrator",
        ),
    }
    for stage, hints in stage_patterns.items():
        if any(hint in lowered for hint in hints):
            return stage
    return None


def _infer_agent_card_stage(text: str) -> Optional[str]:
    lowered = _normalize_whitespace(text).lower()
    lowered = re.sub(r"[?.!,;:]+$", "", lowered).strip()
    card_phrases = {
        "extraction": {
            "ask extraction",
            "ask extraction please",
            "use extraction",
            "use extraction please",
            "open extraction",
            "open extraction please",
            "access extraction",
            "access extraction please",
            "extraction agent",
            "extraction agent please",
            "pregunta extracción",
            "pregunta extracción por favor",
            "pregunta extraction",
            "usar extracción",
            "usar la extracción",
            "usar el agente de extracción",
        },
        "cleaning": {
            "ask cleaning",
            "ask cleaning please",
            "use cleaning",
            "use cleaning please",
            "open cleaning",
            "open cleaning please",
            "access cleaning",
            "access cleaning please",
            "cleaning agent",
            "cleaning agent please",
            "pregunta limpieza",
            "pregunta limpieza por favor",
            "pregunta cleaning",
            "usar limpieza",
            "usar la limpieza",
            "usar el agente de limpieza",
        },
        "modeling": {
            "ask modeling",
            "ask modeling please",
            "use modeling",
            "use modeling please",
            "open modeling",
            "open modeling please",
            "access modeling",
            "access modeling please",
            "modeling agent",
            "modeling agent please",
            "pregunta modelado",
            "pregunta modelado por favor",
            "pregunta modeling",
            "usar modelado",
            "usar el modelado",
            "usar el agente de modelado",
        },
    }
    for stage, phrases in card_phrases.items():
        if lowered in phrases:
            return stage
    return None


def _shortcut_route(text: str, state: AssistantState) -> Optional[AssistantRoute]:
    normalized = re.sub(r"[?.!,;:\)\]]+$", "", _normalize_whitespace(text).lower()).strip()
    shortcut = _MENU_SHORTCUTS.get(normalized)
    if not shortcut:
        return None
    route = AssistantRoute(raw_message=_normalize_whitespace(text))
    language = state.preferred_language if state.preferred_language in {"en", "es"} else _detect_language(text)
    route.language = language
    route.intent = shortcut["intent"]
    if shortcut.get("stage"):
        route.stage = shortcut["stage"]
    if shortcut.get("language"):
        route.language = shortcut["language"]
    if route.intent in {"show_stage_brief", "show_agent_card", "show_clean_data", "show_session_status", "show_session_status_full", "show_mode_guide", "show_agent_guide"} and state.last_run_id:
        route.run_id = state.last_run_id
    return route


def _extract_run_id(text: str) -> Optional[str]:
    lowered = _normalize_whitespace(text).lower()
    match = re.search(r"\brun(?:[_\s-]*)?(\d+)\b", lowered)
    if not match:
        return None
    digits = match.group(1)
    if len(digits) < 4:
        digits = digits.zfill(4)
    return f"run_{digits}"


def _run_scoped_route(text: str, state: AssistantState) -> Optional[AssistantRoute]:
    run_id = _extract_run_id(text)
    if not run_id:
        return None

    normalized = _normalize_whitespace(text)
    lowered = normalized.lower()
    language = state.preferred_language if state.preferred_language in {"en", "es"} else _detect_language(text)
    route = AssistantRoute(raw_message=normalized, run_id=run_id, language=language)

    if _looks_like_session_status_full_request(normalized):
        route.intent = "show_session_status_full"
        route.stage = "session_status_full"
        return route
    if _looks_like_session_status_request(normalized):
        route.intent = "show_session_status"
        route.stage = "session_status"
        return route
    if _looks_like_extraction_report_request(normalized):
        route.intent = "show_extraction"
        route.stage = "extraction"
        return route
    if _looks_like_model_variables_request(normalized):
        route.intent = "show_model_variables"
        route.stage = "model_variables"
        return route
    if _looks_like_clean_row_analysis_request(normalized) or _looks_like_clean_data_request(normalized):
        route.intent = "show_clean_data"
        route.stage = "clean_data"
        return route

    if any(
        hint in lowered
        for hint in (
            "limpieza",
            "cleaning",
            "clean data",
            "cleaned data",
            "datos limpios",
            "clean row",
            "cleaned row",
        )
    ):
        if any(
            hint in lowered
            for hint in (
                "row",
                "rows",
                "fila",
                "filas",
                "analysis",
                "análisis",
                "analisis",
                "schema",
                "esquema",
                "metric",
                "metrics",
                "métrica",
                "métricas",
                "symbols",
                "símbolos",
                "simbolos",
            )
        ):
            route.intent = "show_clean_data"
            route.stage = "clean_data"
        else:
            route.intent = "show_cleaning"
            route.stage = "cleaning"
        return route

    if any(
        hint in lowered
        for hint in (
            "modelado",
            "modeling",
            "prediction",
            "predicted",
            "predijo",
            "what did the model",
            "qué predijo",
            "que predijo",
            "model ",
        )
    ):
        route.intent = "show_prediction"
        route.stage = "modeling"
        return route

    if any(
        hint in lowered
        for hint in (
            "extracción",
            "extraccion",
            "extraction",
            "extract",
        )
    ):
        route.intent = "show_extraction"
        route.stage = "extraction"
        return route

    if any(
        hint in lowered
        for hint in (
            "orquestador",
            "orchestrator",
            "decision",
            "decidió",
            "decidio",
            "qué decidió",
            "que decidió",
            "qué decidio",
            "que decidio",
        )
    ):
        if any(
            hint in lowered
            for hint in (
                "why",
                "por qué",
                "porque",
                "qué pasó",
                "que paso",
                "what happened",
                "why did",
                "explain",
                "explica",
                "decision",
                "decisión",
                "result",
                "summary",
            )
        ):
            route.intent = "show_decision_explanation"
            route.stage = "decision"
        else:
            route.intent = "show_stage_brief"
            route.stage = "orchestrator"
        return route

    return None


def _looks_like_groq_brain_request(text: str) -> bool:
    lowered = (text or "").lower()
    return any(
        hint in lowered
        for hint in (
            "groq brain",
            "brain mode",
            "experimental groq",
            "experimental brain",
            "global brain",
            "global cerebro",
            "trust groq",
            "trust groq as brain",
            "use groq as brain",
            "let groq decide",
            "remove determinism",
            "non deterministic",
            "non-deterministic",
            "deterministic off",
            "cerebro global",
            "groq como cerebro",
            "modo cerebro",
        )
    )


def _looks_like_mode_guide_request(text: str) -> bool:
    lowered = re.sub(r"[?.!,;:]+$", "", _normalize_whitespace(text).lower()).strip()
    if lowered in {
        "local_only",
        "local only",
        "--groq-brain",
        "groq brain",
        "groq-brain",
        "mode guide",
        "modes",
        "mode",
    }:
        return True
    return any(
        hint in lowered
        for hint in (
            "what modes can i use",
            "what mode can i use",
            "available modes",
            "show modes",
            "mode help",
            "help modes",
            "guia de modos",
            "guía de modos",
            "modos disponibles",
            "qué modos puedo usar",
            "que modos puedo usar",
            "como usar los modos",
            "cómo usar los modos",
        )
    )


def _looks_like_agent_guide_request(text: str) -> bool:
    lowered = re.sub(r"[?.!,;:]+$", "", _normalize_whitespace(text).lower()).strip()
    if lowered in {
        "agents",
        "agent",
        "agent guide",
        "show agents",
        "open agents",
        "guide agents",
        "guia de agentes",
        "guía de agentes",
        "agentes",
    }:
        return True
    return any(
        hint in lowered
        for hint in (
            "what agents can i use",
            "what agent can i use",
            "available agents",
            "show agent guide",
            "agent access",
            "access agents",
            "agent menu",
            "menu de agentes",
            "qué agentes puedo usar",
            "que agentes puedo usar",
            "cómo usar los agentes",
            "como usar los agentes",
            "what can i ask now",
            "what can i ask next",
            "what should i ask next",
            "what can each one do",
            "what does each one do",
            "what does each agent do",
            "qué puede hacer cada uno",
            "que puede hacer cada uno",
            "qué hace cada uno",
            "que hace cada uno",
            "how do the agents talk",
            "how do the agents talk to each other",
            "how do i talk to the agents",
            "how do i move from one agent to another",
        )
    )


def _looks_like_help_request(text: str) -> bool:
    lowered = _normalize_whitespace(text).lower()
    if not lowered:
        return False
    return any(
        hint in lowered
        for hint in (
            "help",
            "what can you do",
            "what can u do",
            "what u can do",
            "what can it do",
            "what do you do",
            "what do u do",
            "how can i ask",
            "how i can ask",
            "how do i ask",
            "how can you help me",
            "how can u help me",
            "how do you help",
            "how do you work",
            "how u work",
            "how to ask",
            "what can i do with the data",
            "what can i do with data",
            "what can i do using the data",
            "what can i do with the dataset",
            "what do i do with that",
            "and what do i do with that",
            "what do i do with this",
            "what should i do next",
            "what do i do next",
            "what can i do next",
            "what now",
            "and now",
            "what do you recommend now",
            "qué hago con eso",
            "que hago con eso",
            "y qué hago con eso",
            "y que hago con eso",
            "qué hago ahora",
            "que hago ahora",
            "ahora qué sigue",
            "ahora que sigue",
            "qué me recomiendas ahora",
            "que me recomiendas ahora",
            "can u search",
            "can you search",
            "ask you",
            "capabilities",
            "available commands",
            "what are your skills",
            "what is your skill",
            "en que me podrias ayudar",
            "en qué me podrías ayudar",
            "en que me puedes ayudar",
            "en qué me puedes ayudar",
            "como me puedes ayudar",
            "cómo me puedes ayudar",
            "que puedo hacer con los datos",
            "qué puedo hacer con los datos",
        )
    )


def _looks_like_decision_explanation_request(text: str) -> bool:
    lowered = _normalize_whitespace(text).lower()
    if _looks_like_semantic_lookup_request(lowered):
        return False
    return any(
        hint in lowered
        for hint in (
            "how does this affect hold",
            "como afecta hold",
            "cómo afecta hold",
            "affect hold",
            "affect long",
            "affect short",
            "what mean each hold long short",
            "what does long mean",
            "what does short mean",
            "what does hold mean",
            "meaning of long",
            "meaning of short",
            "meaning of hold",
            "what is long short hold",
            "what is short long and hold",
            "what does long short hold mean",
            "how does cleaning affect long short hold",
            "how does clean data affect long short hold",
            "how does clean data affect hold long short",
            "why long short hold",
            "how do the clean features affect",
            "how does clean data affect",
            "como afecta la limpieza a hold",
            "cómo afecta la limpieza a hold",
            "como afecta la limpieza a long short hold",
            "cómo afecta la limpieza a long short hold",
            "qué es short long y hold",
            "que es short long y hold",
            "qué es long short hold",
            "que es long short hold",
            "qué significa long short hold",
            "que significa long short hold",
            "explica long short hold",
        )
    )


def _looks_like_symbol_guide_request(text: str) -> bool:
    lowered = re.sub(r"[?.!,;:]+$", "", _normalize_whitespace(text).lower()).strip()
    if lowered in {"symbols", "símbolos", "simbolos", "symbol list", "ticker list", "tickers list"}:
        return True
    return any(
        hint in lowered
        for hint in (
            "what tickers can i use",
            "what symbols can i use",
            "what symbol can i use",
            "what symbols do i use",
            "what symbols can u use",
            "what ticker should i use",
            "which symbols can i use",
            "supported assets",
            "supported markets",
            "supported tickers",
            "que simbolos puedo usar",
            "qué símbolos puedo usar",
            "qué simbolos puedo usar",
            "que símbolos puedo usar",
            "list symbols",
            "list tickers",
            "available symbols",
            "active symbols",
        )
    )


def _parse_date_token(token: str, *, year_hint: Optional[int] = None) -> Optional[date]:
    token = token.strip()
    if not token:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"):
        try:
            return datetime.strptime(token, fmt).date()
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(token).date()
    except ValueError:
        pass

    m = re.match(r"^(?P<month>[A-Za-z]+)\s+(?P<day>\d{1,2})(?:,\s*(?P<year>\d{4}))?$", token)
    if m:
        month = _MONTH_ALIASES.get(m.group("month").lower())
        if month:
            year = int(m.group("year") or year_hint or date.today().year)
            day = int(m.group("day"))
            try:
                return date(year, month, day)
            except ValueError:
                return None
    return None


def _month_end(year: int, month: int) -> date:
    last_day = calendar.monthrange(year, month)[1]
    return date(year, month, last_day)


def _parse_date_range(text: str) -> tuple[Optional[date], Optional[date]]:
    raw = _normalize_whitespace(text)
    explicit = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", raw)
    if len(explicit) >= 2:
        return _parse_date_token(explicit[0]), _parse_date_token(explicit[1])

    # Month-only ranges like "Jan to Mar" or "from Jan to Mar".
    lowered = raw.lower()
    month_hits: list[tuple[int, str]] = []
    for token in re.findall(r"\b[a-zA-Z]+\b", lowered):
        month = _MONTH_ALIASES.get(token)
        if month:
            month_hits.append((month, token))
    if len(month_hits) >= 2:
        year = date.today().year
        start_month = month_hits[0][0]
        end_month = month_hits[1][0]
        return date(year, start_month, 1), _month_end(year, end_month)

    # Single date with contextual month/year words, e.g. "Mar 1 2026".
    token_matches = re.findall(r"\b(?:[A-Za-z]+\s+\d{1,2}(?:,\s*\d{4})?|\d{4}-\d{2}-\d{2})\b", raw)
    if len(token_matches) >= 2:
        return _parse_date_token(token_matches[0]), _parse_date_token(token_matches[1])

    return None, None


def _ticker_candidates(text: str) -> List[str]:
    raw = text or ""
    patterns = [
        r"\b[A-Z]{1,10}-(?:USD|USDT|USDC|BUSD)\b",
        r"\b[A-Z]{1,10}-USD\b",
        r"\b[A-Z]{1,10}USDT\b",
        r"\b[A-Z]{1,10}USDC\b",
        r"\b[A-Z]{1,10}BUSD\b",
        r"\b[A-Z]{1,10}=X\b",
        r"\^[A-Z]{2,8}\b",
        r"\b[A-Z]{2,8}\b",
    ]
    stopwords = {
        "I",
        "A",
        "AND",
        "OR",
        "THE",
        "RUN",
        "SHOW",
        "LAST",
        "LATEST",
        "MODE",
        "FOR",
        "FROM",
        "TO",
        "WITH",
        "WHAT",
        "THAT",
        "THIS",
        "THEN",
        "CAN",
        "YOU",
        "LEGACY",
        "BINANCE",
        "YFINANCE",
        "MODEL",
        "MODELS",
        "SUMMARY",
        "PREDICTION",
        "CURRENT",
        "REVIEWER",
        "COMPARE",
        "EXTRACTION",
        "EXTRACT",
        "CLEAN",
        "CONTINUE",
        "USD",
        "USDT",
        "USDC",
        "BUSD",
        "JAN",
        "JANUARY",
        "FEB",
        "FEBRUARY",
        "MAR",
        "MARCH",
        "APR",
        "APRIL",
        "MAY",
        "JUN",
        "JUNE",
        "JUL",
        "JULY",
        "AUG",
        "AUGUST",
        "SEP",
        "SEPT",
        "SEPTEMBER",
        "OCT",
        "OCTOBER",
        "NOV",
        "NOVEMBER",
        "DEC",
        "DECEMBER",
        "USD",
        "USDT",
        "USDC",
        "BUSD",
    }
    seen: set[str] = set()
    tickers: List[str] = []
    for pattern in patterns:
        for match in re.findall(pattern, raw):
            candidate = str(match).strip()
            if not candidate:
                continue
            if candidate.upper() in stopwords:
                continue
            if candidate not in seen:
                seen.add(candidate)
                tickers.append(candidate)
    return tickers


def _normalize_yfinance_ticker(token: str) -> str:
    raw = (token or "").strip().upper()
    if not raw:
        return raw
    if raw.startswith("^") or raw.endswith("=X") or raw.endswith(".TO") or raw.endswith(".DE"):
        return raw
    if raw.endswith("-USD"):
        return raw
    match = re.match(r"^(?P<asset>[A-Z0-9]{2,10})-?(?:USD|USDT|USDC|BUSD)$", raw)
    if match:
        asset = match.group("asset")
        if asset:
            return f"{asset}-USD"
    if raw in _CRYPTO_FAMILY_ALIASES:
        return f"{raw}-USD"
    return raw


def _infer_compare_fields(route: AssistantRoute) -> AssistantRoute:
    if not route.compare_binance:
        return route
    if route.tickers and not route.comparison_yfinance_ticker:
        route.comparison_yfinance_ticker = route.tickers[0]
    if route.comparison_asset is None and route.tickers:
        base = re.sub(r"[^A-Z0-9]", "", route.tickers[0].upper())
        for suffix in ("USDT", "USDC", "BUSD", "USD"):
            if base.endswith(suffix) and len(base) > len(suffix):
                base = base[: -len(suffix)]
                break
        if base.endswith("USD") and route.tickers[0].endswith("-USD"):
            base = route.tickers[0].split("-", 1)[0]
        route.comparison_asset = base or None
    if route.comparison_asset and route.comparison_asset.upper() in {"USD", "USDT", "USDC", "BUSD"}:
        route.comparison_asset = None
    if route.comparison_asset and not route.comparison_binance_symbol:
        route.comparison_binance_symbol = f"{route.comparison_asset.upper()}USDT"
    return route


def _infer_intent(message: str) -> str:
    lowered = message.lower()
    if _infer_agent_card_stage(message):
        return "show_agent_card"
    if _infer_agent_access_stage(message):
        return "show_stage_brief"
    if _looks_like_mode_guide_request(message):
        return "show_mode_guide"
    if _looks_like_agent_guide_request(message):
        return "show_agent_guide"
    if _looks_like_groq_status_request(message):
        return "show_groq_status"
    if _looks_like_semantic_lookup_request(message):
        return "show_semantic_lookup"
    if _looks_like_assistant_scorecard_request(message):
        return "show_assistant_scorecard"
    if _looks_like_web_status_request(message):
        return "show_web_status"
    if _looks_like_session_status_full_request(message):
        return "show_session_status_full"
    if _looks_like_session_status_request(message):
        return "show_session_status"
    if _looks_like_extraction_report_request(message):
        return "show_extraction"
    if _looks_like_clean_row_analysis_request(message) or _looks_like_clean_data_request(message):
        return "show_clean_data"
    if _looks_like_market_metrics_request(message):
        return "show_market_metrics"
    if _looks_like_latest_summary_request(message):
        return "show_latest_summary"
    if "extraction" in lowered and any(hint in lowered for hint in ("show", "result", "latest")):
        return "show_extraction"
    if _looks_like_asset_used_request(message):
        return "show_asset_used"
    if _looks_like_market_type_request(message):
        return "show_market_type"
    if _looks_like_decision_explanation_request(message):
        return "show_decision_explanation"
    if _looks_like_clean_row_analysis_request(message):
        return "show_clean_data"
    if any(
        hint in lowered
        for hint in (
            "clean market data",
            "cleaned market data",
            "cleaned data",
            "clean data",
            "clean_market_data",
            "datos limpios",
            "datos de limpieza",
            "datos limpios del mercado",
            "inspect clean data",
            "inspect the clean data",
            "show clean data",
            "show cleaned data",
            "show the cleaned data",
            "show clean market data",
            "show the clean market data",
            "latest cleaned row",
            "last cleaned row",
            "cleaned row",
            "rows from clean",
            "open high",
            "open, high",
            "close low",
            "ticker open high",
            "symbols in the cleaned data",
            "symbols in clean data",
            "what symbols are in",
            "which symbols are in",
            "clean data metrics",
            "cleaned data metrics",
            "cleaned csv metrics",
            "metrics in the cleaned data",
            "what metrics are in the cleaned data",
            "what metrics are available in the cleaned data",
            "qué métricas hay en los datos limpios",
            "que metricas hay en los datos limpios",
            "clean data schema",
            "schema of clean data",
            "clean data structure",
            "clean row analysis",
            "analyze the clean row",
            "analyze clean row",
            "analysis of clean row",
            "what does this row mean",
            "specific clean row",
            "specific row",
        )
    ):
        return "show_clean_data"
    if _looks_like_market_metrics_request(message):
        return "show_market_metrics"
    if "clean" in lowered and any(hint in lowered for hint in ("show", "result", "current", "latest")):
        return "show_cleaning"
    if _looks_like_model_variables_request(message):
        return "show_model_variables"
    if "prediction" in lowered or "predicted" in lowered or "what did the model" in lowered:
        return "show_prediction"
    if "legacy" in lowered:
        return "show_legacy_status"
    if _looks_like_groq_status_request(message):
        return "show_groq_status"
    if _infer_language_switch(lowered):
        return "set_language"
    if _looks_like_symbol_guide_request(message):
        return "show_symbol_guide"
    if any(hint in lowered for hint in ("last", "latest", "previous", "last run", "previous run", "latest run")):
        if any(hint in lowered for hint in ("decision", "decission", "orchestrator", "orchestator", "summary", "result")) and any(
            hint in lowered for hint in ("extract", "extraction", "clean", "cleaning", "model", "modeling", "prediction", "predicted")
        ):
            return "show_stage_brief"
        if any(hint in lowered for hint in ("clean", "cleaning")) and any(hint in lowered for hint in ("extract", "extraction")):
            if any(hint in lowered for hint in ("and", "process", "about", "what can", "what u can", "tell me", "toll me", "walk me through", "describe")):
                return "show_stage_brief"
            return "show_cleaning"
        if any(hint in lowered for hint in ("clean", "cleaning")):
            return "show_cleaning"
        if _looks_like_model_variables_request(message):
            return "show_model_variables"
        if any(hint in lowered for hint in ("model", "modeling", "prediction", "predicted")):
            return "show_prediction"
        if any(hint in lowered for hint in ("extract", "extraction")):
            return "show_extraction"
        if any(hint in lowered for hint in ("decision", "orchestrator", "summary", "result")):
            return "show_stage_brief"
    if any(hint in lowered for hint in ("motor", "groq say", "what did groq", "ai say", "groq said", "brief", "explain extraction", "explain cleaning", "explain modeling", "explain orchestrator", "how did it extract", "how did it clean", "how did it model", "how did the orchestrator decide")):
        return "show_stage_brief"
    if any(
        hint in lowered
        for hint in (
            "what say groq",
            "what does groq say",
            "what groq says",
            "que dice groq",
            "qué dice groq",
            "groq here",
            "groq aca",
            "groq aquí",
            "groq aqui",
        )
    ):
        return "show_stage_brief"
    if _looks_like_model_variables_request(message):
        return "show_model_variables"
    if any(hint in lowered for hint in ("what happened", "why did", "why is", "explain", "tell me", "tell u", "toll me", "walk me through", "describe", "what can you tell me", "what can u tell me")):
        return "show_stage_brief"
    if "compare" in lowered and "binance" in lowered:
        return "compare_sources"
    if "continue" in lowered and ("unfinished" in lowered or "task" in lowered or "pending" in lowered):
        return "continue_task"
    if any(
        hint in lowered
        for hint in (
            "help",
            "what can you do",
            "what can u do",
            "what u can do",
            "what can it do",
            "what do you do",
            "what do u do",
            "how can i ask",
            "how i can ask",
            "how do i ask",
            "how can you help me",
            "how can u help me",
            "how do you help",
            "how do you work",
            "how u work",
            "how to ask",
            "can u search",
            "can you search",
            "ask you",
            "capabilities",
            "available commands",
            "what are your skills",
            "what is your skill",
            "what symbols can i use",
            "what symbols can u use",
            "what ticker should i use",
            "which symbols can i use",
            "what symbol was used",
            "what ticker was used",
        )
    ):
        return "help"
    if any(
        hint in lowered
        for hint in (
            "run",
            "trigger",
            "search",
            "fetch",
            "look up",
            "find",
            "pull",
            "test",
            "tests",
            "demo",
            "select it you",
            "you choose",
            "make a test",
            "make tests",
            "not matter when",
        )
    ) and not _extract_run_id(message):
        return "run_pipeline"
    if any(hint in lowered for hint in ("price", "quote", "value")) and any(
        hint in lowered
        for hint in (
            "forex",
            "fx",
            "currency",
            "stock",
            "stocks",
            "equity",
            "action",
            "actions",
            "crypto",
            "bitcoin",
            "btc",
            "eth",
            "etf",
            "fund",
        )
    ):
        return "run_pipeline"
    return "unknown"


def _infer_stage_brief_target(message: str) -> str:
    lowered = message.lower()
    explicit_stage = _infer_agent_access_stage(message)
    if explicit_stage:
        if explicit_stage in {"extraction", "cleaning", "modeling"} and (
            any(
                hint in lowered
                for hint in (
                    "orchestrator",
                    "orchestator",
                    "final decision",
                    "decision layer",
                    "judge",
                    "review",
                    "summary",
                    "result",
                    "decision takes",
                    "decision take",
                    "takes for the orchestrator",
                )
            )
            or ("process" in lowered and any(hint in lowered for hint in ("decision", "decission", "summary", "result", "hold", "long", "short")))
        ):
            return "orchestrator"
        return explicit_stage
    stage_hits = {
        "extraction": any(hint in lowered for hint in ("extract", "extraction")),
        "cleaning": any(hint in lowered for hint in ("clean", "cleaning")),
        "modeling": any(hint in lowered for hint in ("model", "modeling", "prediction", "predicted")),
        "orchestrator": any(
            hint in lowered
            for hint in (
                "orchestrator",
                "orchestator",
                "final decision",
                "decision layer",
                "judge",
                "review",
                "decision",
                "decission",
                "decision takes",
                "decision take",
                "takes for the orchestrator",
                "process",
                "hold",
                "long",
                "short",
            )
        ),
    }
    if "motor" in lowered and "groq" not in lowered:
        return "motor"
    if sum(stage_hits.values()) > 1:
        return "orchestrator"
    if stage_hits["extraction"]:
        return "extraction"
    if stage_hits["cleaning"]:
        return "cleaning"
    if stage_hits["modeling"]:
        return "modeling"
    if stage_hits["orchestrator"]:
        return "orchestrator"
    return "orchestrator"


def _extract_explicit_mode(message: str) -> tuple[str, bool, bool]:
    lowered = message.lower()
    if any(hint in lowered for hint in ("no reviewer", "without reviewer", "--no-reviewer", "review off", "review_mode off", "review mode off", "local only")):
        return "off", False, True
    if any(hint in lowered for hint in ("review mode on", "reviewer on", "forced reviewer", "review on")):
        return "on", True, False
    if any(hint in lowered for hint in ("review mode auto", "automatic reviewer", "review auto", "auto reviewer", "only when needed")):
        return "auto", True, False
    return "auto", True, False


def _merge_from_pending(base: AssistantRoute, pending: Optional[Dict[str, Any]]) -> AssistantRoute:
    if not isinstance(pending, dict) or not pending:
        return base
    merged = AssistantRoute.from_dict(pending)
    for field_name in base.__dataclass_fields__:
        value = getattr(base, field_name)
        if field_name == "intent" and value == "unknown":
            continue
        if value in (None, "", [], False, 0.0):
            continue
        setattr(merged, field_name, value)
    merged.tickers = list(dict.fromkeys(merged.tickers))
    if merged.intent in {"run_pipeline", "compare_sources"} and merged.tickers and merged.start and merged.end:
        merged.needs_follow_up = False
        merged.follow_up_question = ""
    elif merged.intent == "unknown" and merged.tickers and merged.start and merged.end:
        merged.intent = "run_pipeline"
        merged.needs_follow_up = False
        merged.follow_up_question = ""
    return merged


def _heuristic_route(message: str, state: AssistantState) -> AssistantRoute:
    normalized = _normalize_whitespace(message)
    lowered = normalized.lower()
    route = AssistantRoute(raw_message=normalized)
    shortcut_route = _shortcut_route(normalized, state)
    if shortcut_route:
        return shortcut_route
    if _looks_like_symbol_guide_request(normalized):
        return AssistantRoute(
            intent="show_symbol_guide",
            stage="symbols",
            language=_detect_language(normalized),
            raw_message=normalized,
        )
    if _looks_like_decision_explanation_request(normalized):
        return AssistantRoute(
            intent="show_decision_explanation",
            stage="decision",
            language=_detect_language(normalized),
            raw_message=normalized,
        )
    route.language = _detect_language(normalized)
    requested_language = _infer_requested_language(normalized)
    if requested_language:
        route.language = requested_language
    route.run_id = _extract_run_id(normalized)
    if _looks_like_mode_guide_request(normalized):
        route.intent = "show_mode_guide"
        route.stage = "mode_guide"
        return route
    route.intent = _infer_intent(normalized)
    route.tickers = [_normalize_yfinance_ticker(item) for item in _ticker_candidates(normalized)]
    route.start, route.end = _parse_date_range(normalized)
    route.interval = "1d"
    route.experimental_groq_brain = _looks_like_groq_brain_request(normalized)
    card_stage = _infer_agent_card_stage(normalized)
    if card_stage:
        route.intent = "show_agent_card"
        route.stage = card_stage
        route.tickers = []
        route.start = None
        route.end = None
        if state.last_run_id:
            route.run_id = state.last_run_id
        return route

    if route.intent == "set_language":
        switch_language = _infer_language_switch(normalized)
        if switch_language:
            route.language = switch_language

    default_ticker = _default_ticker_for_message(normalized)

    if route.intent == "unknown":
        if _looks_like_latest_summary_request(normalized) or "summary" in lowered:
            route.intent = "show_latest_summary"
        elif any(hint in lowered for hint in ("show extraction", "extraction result", "latest extraction", "last extraction")):
            route.intent = "show_extraction"
        elif _looks_like_agent_guide_request(normalized):
            route.intent = "show_agent_guide"
        elif _looks_like_session_status_full_request(normalized):
            route.intent = "show_session_status_full"
        elif _looks_like_time_status_request(normalized):
            route.intent = "show_time_status"
        elif _looks_like_source_scope_request(normalized):
            route.intent = "show_source_scope"
        elif "clean" in lowered:
            route.intent = "show_cleaning"
        elif _looks_like_model_variables_request(normalized):
            route.intent = "show_model_variables"
        elif "prediction" in lowered or "model" in lowered:
            route.intent = "show_prediction"
        elif "legacy" in lowered:
            route.intent = "show_legacy_status"
        elif _looks_like_web_status_request(normalized):
            route.intent = "show_web_status"
        elif _looks_like_session_status_request(normalized):
            route.intent = "show_session_status"
        elif "binance" in lowered and "compare" in lowered:
            route.intent = "compare_sources"
        elif any(hint in lowered for hint in ("continue", "unfinished", "pending")):
            route.intent = "continue_task"
        elif any(hint in lowered for hint in ("run", "trigger")) and not route.run_id:
            route.intent = "run_pipeline"
        elif _looks_like_help_request(normalized):
            route.intent = "help"

    if route.intent == "unknown" and _looks_like_explanation_request(normalized):
        if any(hint in lowered for hint in ("extraction", "extract", "extracción", "extraccion")):
            route.intent = "show_extraction"
        elif any(hint in lowered for hint in ("clean", "cleaning", "limpieza")):
            route.intent = "show_cleaning"
        elif _looks_like_model_variables_request(normalized):
            route.intent = "show_model_variables"
        elif any(hint in lowered for hint in ("model", "modeling", "modelado", "prediction", "predicted", "predijo")):
            route.intent = "show_prediction"
        elif any(hint in lowered for hint in ("hold", "long", "short", "confidence", "number", "score")):
            route.intent = "show_prediction"
        elif any(hint in lowered for hint in ("legacy",)):
            route.intent = "show_legacy_status"
        elif any(hint in lowered for hint in ("decision", "orchestrator", "summary", "result", "motor")):
            route.intent = "show_stage_brief"
        elif _infer_language_switch(normalized):
            route.intent = "set_language"

    if route.intent == "unknown" and route.tickers:
        route.intent = "run_pipeline"

    if route.intent == "unknown" and route.run_id:
        route.intent = "show_session_status"
        route.stage = "session_status"

    if route.intent == "unknown" and default_ticker:
        route.tickers = [default_ticker]
        route.intent = "run_pipeline"

    if route.intent in {"run_pipeline", "compare_sources"} and not route.tickers:
        if route.intent == "compare_sources":
            route.tickers = [default_ticker or "BTC-USD"]
        elif default_ticker:
            route.tickers = [default_ticker]
        elif state.current_asset:
            route.tickers = [state.current_asset]

    if route.intent in {"run_pipeline", "compare_sources"} and route.tickers:
        route.tickers = [_normalize_yfinance_ticker(item) for item in route.tickers]

    if route.intent in {"run_pipeline", "compare_sources"}:
        review_mode, use_reviewer, explicit = _extract_explicit_mode(normalized)
        route.review_mode = review_mode
        route.use_reviewer = use_reviewer
        if route.intent == "compare_sources":
            route.compare_binance = True
            route.review_mode = "off" if not explicit else review_mode
            route.use_reviewer = False if route.review_mode == "off" else use_reviewer
        else:
            # Keep the default reviewer on unless the user explicitly asked for local-only.
            if review_mode == "auto":
                route.use_reviewer = True
            elif review_mode == "on":
                route.use_reviewer = True
            elif review_mode == "off":
                route.use_reviewer = False
        if route.experimental_groq_brain:
            route.review_mode = "off"
            route.use_reviewer = False

    if "compare-binance" in lowered or ("binance" in lowered and route.intent == "run_pipeline"):
        route.compare_binance = True
        route.review_mode = "off" if route.review_mode == "auto" else route.review_mode
        route.use_reviewer = False if route.review_mode == "off" else route.use_reviewer

    if route.intent == "show_legacy_status" and state.last_run_id:
        route.run_id = state.last_run_id
    elif route.intent in _SHOW_INTENTS and not route.run_id and state.last_run_id:
        route.run_id = state.last_run_id

    if route.intent == "show_stage_brief":
        route.stage = _infer_stage_brief_target(normalized)

    if route.intent == "show_symbol_guide":
        route.stage = "symbols"
    if route.intent == "show_asset_used":
        route.stage = "asset"
    if route.intent == "show_clean_data":
        route.stage = "clean_data"
    if route.intent == "show_market_metrics":
        route.stage = "metrics"
    if route.intent == "show_decision_explanation":
        route.stage = "decision"
    if route.intent == "show_groq_status":
        route.stage = "groq_status"
    if route.intent == "show_web_status":
        route.stage = "web_status"
    if route.intent == "show_time_status":
        route.stage = "time_status"
    if route.intent == "show_source_scope":
        route.stage = "source_scope"
    if route.intent == "show_semantic_lookup":
        route.stage = "semantic_lookup"
    if route.intent == "show_session_status_full":
        route.stage = "session_status_full"
    if route.intent == "show_session_status":
        route.stage = "session_status"
    if route.intent == "show_agent_guide":
        route.stage = "agent_guide"
    if route.intent == "set_language":
        switch_language = _infer_language_switch(normalized)
        if switch_language:
            route.language = switch_language
        route.stage = "language"

    if route.intent == "run_pipeline":
        if route.tickers and (route.start is None or route.end is None):
            if _should_default_dates(normalized) or route.tickers:
                route.start, route.end = _default_date_range()
                route.needs_follow_up = False
                route.follow_up_question = ""
            else:
                route.needs_follow_up = True
                route.follow_up_question = "Give me the ticker(s) and date range, for example: AAPL 2024-01-01 2025-01-01."
        elif not route.tickers:
            route.needs_follow_up = True
            route.follow_up_question = "Give me the ticker(s) and date range, for example: AAPL 2024-01-01 2025-01-01."
    elif route.intent == "compare_sources":
        route.compare_binance = True
        if not route.tickers:
            route.tickers = ["BTC-USD"]
        if route.start is None or route.end is None:
            route.needs_follow_up = True
            route.follow_up_question = "For the Binance comparison, give me the asset and date range, for example: ETH-USD 2024-01-01 2025-01-01."

    if route.intent in {"show_extraction", "show_cleaning", "show_prediction"}:
        if not route.run_id and state.last_run_id:
            route.run_id = state.last_run_id
    if route.intent == "show_agent_card" and not route.run_id and state.last_run_id:
        route.run_id = state.last_run_id

    route = _infer_compare_fields(route)
    route.stage = {
        "show_extraction": "extraction",
        "show_cleaning": "cleaning",
        "show_clean_data": "clean_data",
        "show_prediction": "modeling",
        "show_model_variables": "model_variables",
        "show_legacy_status": "legacy",
        "show_symbol_guide": "symbols",
        "show_asset_used": "asset",
        "show_decision_explanation": "decision",
        "show_latest_summary": "summary",
        "show_semantic_lookup": "semantic_lookup",
        "show_web_status": "web_status",
        "show_agent_guide": "agent_guide",
        "show_agent_card": "agent_card",
        "show_session_status": "session_status",
        "compare_sources": "comparison",
        "continue_task": "continue",
        "help": "help",
        "set_language": "language",
        "show_mode_guide": "mode_guide",
    }.get(route.intent, route.stage)
    return route


def _groq_first_fallback_route(message: str, state: AssistantState) -> AssistantRoute:
    normalized = _normalize_whitespace(message)
    lowered = normalized.lower()
    route = AssistantRoute(raw_message=normalized)
    shortcut_route = _shortcut_route(normalized, state)
    if shortcut_route:
        return shortcut_route
    route.language = _detect_language(normalized)
    requested_language = _infer_requested_language(normalized)
    if requested_language:
        route.language = requested_language
    route.run_id = _extract_run_id(normalized)

    switch_language = _infer_language_switch(normalized)
    if switch_language:
        route.intent = "set_language"
        route.language = switch_language
        route.stage = "language"
        return route
        if _looks_like_mode_guide_request(normalized):
            route.intent = "show_mode_guide"
            route.stage = "mode_guide"
            return route
        if _looks_like_agent_guide_request(normalized):
            route.intent = "show_agent_guide"
            route.stage = "agent_guide"
            return route
        if _looks_like_time_status_request(normalized):
            route.intent = "show_time_status"
            route.stage = "time_status"
            return route
        if _looks_like_source_scope_request(normalized):
            route.intent = "show_source_scope"
            route.stage = "source_scope"
            return route
        if _looks_like_symbol_guide_request(normalized):
            route.intent = "show_symbol_guide"
            route.stage = "symbols"
            return route
    if _looks_like_decision_explanation_request(normalized):
        route.intent = "show_decision_explanation"
        route.stage = "decision"
        return route

    route.tickers = [_normalize_yfinance_ticker(item) for item in _ticker_candidates(normalized)]
    route.start, route.end = _parse_date_range(normalized)
    route.interval = "1d"
    route.experimental_groq_brain = _looks_like_groq_brain_request(normalized)
    card_stage = _infer_agent_card_stage(normalized)
    access_stage = _infer_agent_access_stage(normalized)

    default_ticker = _default_ticker_for_message(normalized)

    if card_stage:
        route.intent = "show_agent_card"
        route.stage = card_stage
    elif access_stage:
        route.intent = "show_stage_brief"
        route.stage = access_stage
    elif _looks_like_session_status_full_request(normalized):
        route.intent = "show_session_status_full"
        route.stage = "session_status_full"
    elif _looks_like_time_status_request(normalized):
        route.intent = "show_time_status"
        route.stage = "time_status"
    elif _looks_like_source_scope_request(normalized):
        route.intent = "show_source_scope"
        route.stage = "source_scope"
    elif _looks_like_extraction_report_request(normalized):
        route.intent = "show_extraction"
        route.stage = "extraction"
    elif _looks_like_clean_row_analysis_request(normalized) or _looks_like_clean_data_request(normalized):
        route.intent = "show_clean_data"
        route.stage = "clean_data"
    elif _looks_like_market_metrics_request(normalized):
        route.intent = "show_market_metrics"
        route.stage = "metrics"
    elif _looks_like_model_variables_request(normalized):
        route.intent = "show_model_variables"
        route.stage = "model_variables"
    elif "compare" in lowered and "binance" in lowered:
        route.intent = "compare_sources"
        route.compare_binance = True
    elif _looks_like_groq_status_request(normalized):
        route.intent = "show_groq_status"
        route.stage = "groq_status"
    elif _looks_like_assistant_scorecard_request(normalized):
        route.intent = "show_assistant_scorecard"
        route.stage = "assistant_scorecard"
    elif _looks_like_time_status_request(normalized):
        route.intent = "show_time_status"
        route.stage = "time_status"
    elif _looks_like_source_scope_request(normalized):
        route.intent = "show_source_scope"
        route.stage = "source_scope"
    elif _looks_like_web_status_request(normalized):
        route.intent = "show_web_status"
        route.stage = "web_status"
    elif _looks_like_semantic_lookup_request(normalized):
        route.intent = "show_semantic_lookup"
        route.stage = "semantic_lookup"
    elif _looks_like_session_status_request(normalized):
        route.intent = "show_session_status"
        route.stage = "session_status"
    elif any(hint in lowered for hint in ("volume", "volatility", "volumen", "volatilidad")):
        route.intent = "show_market_metrics"
        route.stage = "metrics"
    elif _looks_like_clean_row_analysis_request(normalized):
        route.intent = "show_clean_data"
        route.stage = "clean_data"
    elif any(
        hint in lowered
        for hint in (
        "clean market data",
        "cleaned market data",
        "clean data",
        "clean data metrics",
        "cleaned data metrics",
        "cleaned csv metrics",
        "metrics in the cleaned data",
        "what metrics are in the cleaned data",
        "what metrics are available in the cleaned data",
        "qué métricas hay en los datos limpios",
        "que metricas hay en los datos limpios",
        "datos limpios",
        "datos limpios del mercado",
        "open high",
        "clean row analysis",
        "analyze the clean row",
        "analyze clean row",
        "analysis of clean row",
        "what does this row mean",
            "symbols in the cleaned data",
            "symbols in clean data",
            "what symbols are in",
            "which symbols are in",
            "clean data schema",
            "schema of clean data",
            "clean data structure",
            "specific clean row",
            "specific row",
        )
    ):
        route.intent = "show_clean_data"
        route.stage = "clean_data"
    elif _looks_like_asset_used_request(normalized):
        route.intent = "show_asset_used"
        route.stage = "asset"
    elif _looks_like_market_type_request(normalized):
        route.intent = "show_market_type"
        route.stage = "market_type"
    elif _looks_like_latest_summary_request(normalized):
        route.intent = "show_latest_summary"
        route.stage = "summary"
    elif any(
        hint in lowered
        for hint in (
            "show extraction",
            "extraction result",
            "latest extraction",
            "last extraction",
            "raw columns",
            "missing columns",
        )
    ):
        route.intent = "show_extraction"
        route.stage = "extraction"
    elif any(hint in lowered for hint in ("cleaning", "clean")):
        route.intent = "show_cleaning"
        route.stage = "cleaning"
    elif _looks_like_model_variables_request(normalized):
        route.intent = "show_model_variables"
        route.stage = "model_variables"
    elif any(hint in lowered for hint in ("prediction", "predicted", "model", "modeling")):
        route.intent = "show_prediction"
        route.stage = "modeling"
    elif "legacy" in lowered:
        route.intent = "show_legacy_status"
        route.stage = "legacy"
    elif _looks_like_explanation_request(normalized):
        if any(hint in lowered for hint in ("hold", "long", "short", "confidence", "decision")):
            route.intent = "show_decision_explanation"
            route.stage = "decision"
        else:
            if any(hint in lowered for hint in ("extraction", "extract", "extracción", "extraccion")):
                route.intent = "show_extraction"
                route.stage = "extraction"
            elif any(hint in lowered for hint in ("clean", "cleaning", "limpieza")):
                route.intent = "show_cleaning"
                route.stage = "cleaning"
            elif _looks_like_model_variables_request(normalized):
                route.intent = "show_model_variables"
                route.stage = "model_variables"
            elif any(hint in lowered for hint in ("model", "modeling", "modelado", "prediction", "predicted", "predijo")):
                route.intent = "show_prediction"
                route.stage = "modeling"
            else:
                route.intent = "show_stage_brief"
                route.stage = _infer_stage_brief_target(normalized)
    elif _looks_like_help_request(normalized):
        route.intent = "help"
        route.stage = "help"
    elif any(
        hint in lowered
        for hint in (
            "run",
            "extract",
            "trigger",
            "search",
            "fetch",
            "look up",
            "find",
            "pull",
            "test",
            "tests",
            "demo",
            "select it you",
            "you choose",
            "make a test",
            "make tests",
            "not matter when",
        )
    ) and not route.run_id:
        route.intent = "run_pipeline"
    elif any(hint in lowered for hint in ("price", "quote", "value")) and any(
        hint in lowered
        for hint in (
            "forex",
            "fx",
            "currency",
            "stock",
            "stocks",
            "equity",
            "action",
            "actions",
            "crypto",
            "bitcoin",
            "btc",
            "eth",
            "etf",
            "fund",
        )
    ):
        route.intent = "run_pipeline"
    else:
        route.intent = "unknown"

    if route.intent == "unknown" and route.tickers:
        route.intent = "run_pipeline"

    if route.intent == "unknown" and route.run_id:
        route.intent = "show_session_status"
        route.stage = "session_status"

    if route.intent == "unknown" and default_ticker:
        route.tickers = [default_ticker]
        route.intent = "run_pipeline"

    if route.intent in {"run_pipeline", "compare_sources"} and not route.tickers:
        if route.intent == "compare_sources":
            route.tickers = [default_ticker or "BTC-USD"]
        elif default_ticker:
            route.tickers = [default_ticker]
        elif state.current_asset:
            route.tickers = [state.current_asset]

    if route.intent in {"run_pipeline", "compare_sources"}:
        if route.tickers and (route.start is None or route.end is None):
            if _should_default_dates(normalized) or route.tickers:
                route.start, route.end = _default_date_range()
            else:
                route.needs_follow_up = True
                route.follow_up_question = "Give me the ticker(s) and date range, for example: AAPL 2024-01-01 2025-01-01."
        elif not route.tickers:
            route.needs_follow_up = True
            route.follow_up_question = "Give me the ticker(s) and date range, for example: AAPL 2024-01-01 2025-01-01."

    if route.intent == "compare_sources":
        route.compare_binance = True
        route.review_mode = "off"
        route.use_reviewer = False
        if not route.tickers:
            route.tickers = ["BTC-USD"]
        if route.start is None or route.end is None:
            route.needs_follow_up = True
            route.follow_up_question = "For the Binance comparison, give me the asset and date range, for example: ETH-USD 2024-01-01 2025-01-01."
    if route.experimental_groq_brain and route.intent in {"run_pipeline", "compare_sources"}:
        route.review_mode = "off"
        route.use_reviewer = False

    if route.intent == "show_legacy_status" and state.last_run_id:
        route.run_id = state.last_run_id
    elif route.intent in _SHOW_INTENTS and not route.run_id and state.last_run_id:
        route.run_id = state.last_run_id

    if route.intent == "show_agent_guide":
        route.stage = "agent_guide"

    route = _infer_compare_fields(route)
    route.stage = {
        "show_extraction": "extraction",
        "show_cleaning": "cleaning",
        "show_clean_data": "clean_data",
        "show_market_metrics": "metrics",
        "show_prediction": "modeling",
        "show_legacy_status": "legacy",
        "show_symbol_guide": "symbols",
        "show_asset_used": "asset",
        "show_decision_explanation": "decision",
        "show_latest_summary": "summary",
        "show_semantic_lookup": "semantic_lookup",
        "show_source_scope": "source_scope",
        "show_time_status": "time_status",
        "show_groq_status": "groq_status",
        "show_web_status": "web_status",
        "show_session_status_full": "session_status_full",
        "show_session_status": "session_status",
        "show_mode_guide": "mode_guide",
        "compare_sources": "comparison",
        "continue_task": "continue",
        "help": "help",
        "set_language": "language",
    }.get(route.intent, route.stage)
    return route


class GroqRouter:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 45,
    ) -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.base_url = base_url or os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1/chat/completions")
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _extract_json(self, content: str) -> Optional[dict]:
        if not content:
            return None
        content = content.strip()
        if content.startswith("{") and content.endswith("}"):
            try:
                return json.loads(content)
            except Exception:
                pass
        match = re.search(r"\{.*\}", content, flags=re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None
        return None

    def route(self, message: str, state: AssistantState, fallback: AssistantRoute) -> AssistantRoute:
        if not self.enabled:
            return fallback

        route_examples = (
            "Examples: "
            "'run BTC-USD from 2024-01-01 to 2025-01-01' -> run_pipeline; "
            "'extract SPY from Jan to Mar' -> run_pipeline; "
            "'AAPL' -> run_pipeline; "
            "'BTC-USD' -> run_pipeline; "
            "'i need btc data' -> run_pipeline; "
            "'clean the last extraction' -> show_cleaning; "
            "'tell me about the last extraction' -> show_extraction; "
            "'tell me about the last model' -> show_prediction; "
            "'show me what happened' -> show_stage_brief (stage=orchestrator); "
            "'why did it hold?' -> show_stage_brief (stage=orchestrator); "
            "'show the last run' -> show_latest_summary; "
            "'show the extraction result' -> show_extraction; "
            "'show the cleaning result' -> show_cleaning; "
            "'what did the model predict?' -> show_prediction; "
            "'what is the legacy status?' -> show_legacy_status; "
            "'agents' -> show_agent_guide; "
            "'1' -> show_agent_guide; "
            "'2' -> show_agent_card (stage=extraction); "
            "'3' -> show_agent_card (stage=cleaning); "
            "'4' -> show_agent_card (stage=modeling); "
            "'5' -> show_stage_brief (stage=orchestrator); "
            "'6' -> show_session_status; "
            "'7' -> show_session_status_full; "
            "'8' -> show_clean_data; "
            "'9' -> show_mode_guide; "
            "'A' -> set_language (language=en); "
            "'B' -> set_language (language=es); "
            "'H' -> help; "
            "'Extracción' -> show_agent_card (stage=extraction); "
            "'Limpieza' -> show_agent_card (stage=cleaning); "
            "'Modelado' -> show_agent_card (stage=modeling); "
            "'Orquestador' -> show_stage_brief (stage=orchestrator); "
            "'what agents can I use?' -> show_agent_guide; "
            "'guía de agentes' -> show_agent_guide; "
            "'ask extraction' -> show_agent_card (stage=extraction); "
            "'use extraction' -> show_agent_card (stage=extraction); "
            "'open extraction' -> show_agent_card (stage=extraction); "
            "'ask cleaning' -> show_agent_card (stage=cleaning); "
            "'use cleaning' -> show_agent_card (stage=cleaning); "
            "'open cleaning' -> show_agent_card (stage=cleaning); "
            "'ask modeling' -> show_agent_card (stage=modeling); "
            "'use modeling' -> show_agent_card (stage=modeling); "
            "'open modeling' -> show_agent_card (stage=modeling); "
            "'what symbol was used?' -> show_asset_used; "
            "'qué símbolo se usó?' -> show_asset_used; "
            "'show the clean market data' -> show_clean_data; "
            "'what symbols are in the cleaned data?' -> show_clean_data; "
            "'what metrics are in the cleaned data?' -> show_clean_data; "
            "'show the clean row for MSFT on 2026-03-21' -> show_clean_data; "
            "'analyze the clean row for MSFT on 2026-03-21' -> show_clean_data; "
            "'what is the clean data schema?' -> show_clean_data; "
            "'what is the volume of NVDA?' -> show_market_metrics; "
            "'what is the volatility of AAPL?' -> show_market_metrics; "
            "'how does clean data affect hold long short?' -> show_decision_explanation; "
            "'what groq url is used?' -> show_groq_status; "
            "'is groq active?' -> show_groq_status; "
            "'what does forex mean?' -> show_semantic_lookup; "
            "'qué es forex?' -> show_semantic_lookup; "
            "'what day is it?' -> show_time_status; "
            "'qué día es hoy?' -> show_time_status; "
            "'where does your information come from?' -> show_source_scope; "
            "'qué puedes buscar?' -> show_source_scope; "
            "'web status' -> show_web_status; "
            "'is the web retriever active?' -> show_web_status; "
            "'estado del retriever' -> show_web_status; "
            "'what env do I need for tavily?' -> show_web_status; "
            "'assistant scorecard' -> show_assistant_scorecard; "
            "'what is missing for hybrid runtime?' -> show_assistant_scorecard; "
            "'status full' -> show_session_status_full; "
            "'full status' -> show_session_status_full; "
            "'mode status full' -> show_session_status_full; "
            "'estado completo' -> show_session_status_full; "
            "'status' -> show_session_status; "
            "'mode status' -> show_session_status; "
            "'what mode am I in?' -> show_session_status; "
            "'am I in Groq brain?' -> show_session_status; "
            "'what is the confidence?' -> show_session_status; "
            "'what motor was used?' -> show_stage_brief (stage=motor); "
            "'what did Groq say in extraction?' -> show_stage_brief (stage=extraction); "
            "'what did Groq say in cleaning?' -> show_stage_brief (stage=cleaning); "
            "'what did Groq say in modeling?' -> show_stage_brief (stage=modeling); "
            "'what did Groq say in the orchestrator?' -> show_stage_brief (stage=orchestrator); "
            "'ask extraction what happened?' -> show_stage_brief (stage=extraction); "
            "'ask cleaning what happened?' -> show_stage_brief (stage=cleaning); "
            "'ask modeling what happened?' -> show_stage_brief (stage=modeling); "
            "'ask orchestrator what happened?' -> show_stage_brief (stage=orchestrator); "
            "'pregunta extracción qué pasó?' -> show_stage_brief (stage=extraction); "
            "'pregunta limpieza qué columnas se escribieron?' -> show_stage_brief (stage=cleaning); "
            "'pregunta modelado por qué eligió long?' -> show_stage_brief (stage=modeling); "
            "'pregunta al orquestador por qué se eligió esa decisión?' -> show_stage_brief (stage=orchestrator); "
            "'local_only' -> show_mode_guide; "
            "'--groq-brain' -> show_mode_guide; "
            "'what modes can I use?' -> show_mode_guide; "
            "'what symbols can I use?' -> show_symbol_guide; "
            "'qué símbolos puedo usar?' -> show_symbol_guide; "
            "'switch to Spanish' -> set_language (language=es); "
            "'switch to English' -> set_language (language=en); "
            "'answer in Spanish and show the latest run' -> show_latest_summary with language=es; "
            "'traducir al español la última extracción' -> show_extraction with language=es; "
            "'compare yfinance with Binance for ETH-USD' -> compare_sources; "
            "'select it you, make a test' -> run_pipeline with defaults; "
            "'i want the price of a forex active' -> run_pipeline (ticker=EURUSD=X); "
            "'run AAPL with groq brain' -> run_pipeline with experimental_groq_brain=True; "
            "'what can you tell me about the last extraction and clean process?' -> show_stage_brief (stage=orchestrator); "
            "'share me about the last extraction in your words' -> show_extraction; "
            "'share me the last extraction and clean process in your words' -> show_stage_brief (stage=orchestrator); "
            "'how do I ask you?' -> help."
        )

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You route requests for one assistant with three skills: extract, clean, model. "
                        "The user may speak naturally, mix English and Spanish, or omit exact mode names. "
                        "The user may also type a bare number or letter shortcut such as 1, 2, 3, 4, 5, 6, 7, 8, 9, A, B, or H. "
                        "Treat those shortcuts as the corresponding menu view or language switch. "
                        "If the user types only a ticker or a list of tickers such as AAPL, MSFT, NVDA, SPY, QQQ, ^GSPC, BTC-USD, ETH-USD, or EURUSD=X, "
                        "treat it as a run_pipeline request with default dates unless another intent is explicit. "
                        "If the user types only a bare stage name such as Extraction, Extracción, Cleaning, Limpieza, Modeling, Modelado, Orchestrator, or Orquestador, "
                        "open the corresponding agent card or stage brief instead of inventing a new run. "
                        "If the user types only a bare mode label such as local_only, compare-binance, --groq-brain, or compare-binance + --groq-brain, show the mode guide. "
                        "Infer the intent from the message and current state. "
                        "Prefer the simplest natural interpretation and do not force command syntax on the user. "
                        "If the user asks to switch languages, set language preference to Spanish or English and keep it sticky in session state. "
                        "If the user asks for a response in Spanish, English, or to translate something, set the language field accordingly "
                        "for that answer. "
                        "If the user explicitly asks for an experimental Groq brain or global-brain mode, mark the run route so the pipeline can "
                        "let Groq make the final decision instead of the deterministic local path. "
                        "If the user asks about the active mode, current mode, Groq brain usage, the latest confidence, or the source of the last decision, use intent show_session_status. "
                        "If the user asks for a full or detailed session status, use intent show_session_status_full. "
                        "If the user says 'share me', 'in your words', 'simple words', or similar explanation-style language, "
                        "treat it as an explanation request rather than a new run. "
                        "If the user asks for a generic test or says to choose the asset, prefer a sensible default "
                        "instead of forcing a follow-up when the request is clearly permissive. "
                        "If the user refers to the last, latest, or previous extraction/cleaning/modeling result, "
                        "prefer show_extraction, show_cleaning, or show_prediction instead of starting a new run. "
                        "Return only JSON with keys: "
                        "intent, tickers, start, end, interval, review_mode, use_reviewer, compare_binance, "
                        "comparison_asset, comparison_yfinance_ticker, comparison_binance_symbol, language, "
                        "run_id, stage, needs_follow_up, follow_up_question, confidence, explanation, experimental_groq_brain. "
                        "Allowed intents: run_pipeline, show_latest_summary, show_semantic_lookup, show_source_scope, show_time_status, show_extraction, show_cleaning, "
                        "show_prediction, show_model_variables, show_legacy_status, show_symbol_guide, show_asset_used, show_clean_data, show_market_metrics, show_decision_explanation, show_stage_brief, show_groq_status, show_web_status, show_assistant_scorecard, show_session_status, show_session_status_full, show_mode_guide, show_agent_guide, show_agent_card, compare_sources, continue_task, set_language, help, unknown. "
                        "If the user asks about volume or volatility for an asset, symbol, ticker, or clean market data, use intent show_market_metrics. "
                        "If the user asks what a market term means, asks for a definition like 'what does forex mean?' or 'qué es forex?', use intent show_semantic_lookup. "
                        "If the user asks for the date, time, or what day it is, use intent show_time_status. "
                        "If the user asks where the assistant gets information from, what it can search, or what sources it uses, use intent show_source_scope. "
                        "If the user asks whether Groq is active, which Groq URL/endpoint/model is used, or wants a confirmation of Groq configuration, use intent show_groq_status. "
                        "If the user asks for the assistant scorecard, architecture readiness, maturity percentage, or what is missing for hybrid runtime, use intent show_assistant_scorecard. "
                        "If the user asks whether the web retriever, internet search, or external search config is active or configured, use intent show_web_status. "
                        "If the user asks for the env, setup snippet, or configuration needed for searxng, serper, tavily, or searchapi, also use intent show_web_status. "
                        "If the user asks for help, guidance, or what the assistant can do, use intent help. "
                        "If the user asks about the active mode, current mode, Groq brain usage, the latest confidence, or decision source, use intent show_session_status. "
                        "If the user asks for a detailed or full session status, use intent show_session_status_full. "
                        "If the user asks for the agents menu, agent hub, or an overview of the three agents, use intent show_agent_guide. "
                        "If the user asks to open a specific agent card with ask extraction, use extraction, open extraction, ask cleaning, use cleaning, open cleaning, ask modeling, use modeling, or open modeling, use intent show_agent_card and set stage to extraction, cleaning, or modeling. "
                        "If the user asks 'what motor was used?' or 'what did Groq say in extraction/cleaning/modeling/orchestrator?', "
                        "use intent show_stage_brief and set stage to extraction, cleaning, modeling, orchestrator, or motor. "
                        "If the user asks for supported symbols, symbols per ticker, or what ticker to use, use intent show_symbol_guide. "
                        "If the user asks what symbol/ticker/asset was used in the last run, use intent show_asset_used. "
                        "If the user asks what metrics are available in the cleaned data or asks for a cleaned-data metrics list, use intent show_clean_data. "
                        "If the user asks which variables, features, transformations, model inputs, or explanatory variables were used in modeling, use intent show_model_variables. "
                        "If the user says ask extraction what happened, ask cleaning what happened, ask modeling what happened, pregunta extracción qué pasó, pregunta limpieza qué columnas se escribieron, "
                        "pregunta modelado por qué eligió long, or pregunta al orquestador por qué se eligió esa decisión, use intent show_stage_brief and set stage to extraction, cleaning, modeling, or orchestrator. "
                        "If the user says local_only, --groq-brain, groq brain, or asks what modes can be used, use intent show_mode_guide. "
                        "If the user asks for agents or the agent hub, use intent show_agent_guide. "
                        "If the user asks about clean market data, cleaned rows, specific columns/rows of the cleaned dataset, or metrics available in the cleaned data, use intent show_clean_data. "
                        "If the user asks what symbols are present in the cleaned data, asks for one exact cleaned row, asks for a cleaned-row analysis, asks for the clean data schema/structure, or asks for a cleaned-data metrics list, use intent show_clean_data. "
                        "If the user asks how clean data affects hold/long/short or asks for a didactic decision explanation, use intent show_decision_explanation. "
                        "If the user asks to switch to Spanish or English, use intent set_language and set language to es or en. "
                        "Do not force the user to name the stage explicitly. "
                        "If the question mentions both extraction and cleaning, or extraction and final decision, prefer a combined read-only recap. "
                        "Do not invent facts. Use the conversation state and user text. "
                        + route_examples
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "message": message,
                            "state": compact_for_groq(state.to_dict(), max_depth=2, max_items=8, max_string=300),
                            "fallback": compact_for_groq(fallback.to_dict(), max_depth=2, max_items=8, max_string=300),
                        },
                        separators=(",", ":"),
                        default=str,
                    ),
                },
            ],
            "temperature": 0.1,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
            payload_json: Dict[str, Any] = response.json()
            content = payload_json["choices"][0]["message"]["content"]
            parsed = self._extract_json(content)
            if not parsed:
                logger.debug("Groq router returned non-JSON payload; falling back to heuristics.")
                return fallback
            route = AssistantRoute.from_dict(parsed)
            card_stage = _infer_agent_card_stage(message)
            access_stage = _infer_agent_access_stage(message)
            status_request = _looks_like_session_status_request(message)
            status_full_request = _looks_like_session_status_full_request(message)
            if card_stage:
                route.intent = "show_agent_card"
                route.stage = card_stage
            elif access_stage:
                route.intent = "show_stage_brief"
                route.stage = access_stage
            elif status_full_request:
                route.intent = "show_session_status_full"
                route.stage = "session_status_full"
            elif status_request:
                route.intent = "show_session_status"
                route.stage = "session_status"
            elif _looks_like_time_status_request(message):
                route.intent = "show_time_status"
                route.stage = "time_status"
            elif _looks_like_source_scope_request(message):
                route.intent = "show_source_scope"
                route.stage = "source_scope"
            elif _looks_like_asset_used_request(message):
                route.intent = "show_asset_used"
                route.stage = "asset"
            elif _looks_like_market_type_request(message):
                route.intent = "show_market_type"
                route.stage = "market_type"
            elif _looks_like_market_metrics_request(message):
                route.intent = "show_market_metrics"
                route.stage = "metrics"
            elif _looks_like_clean_row_analysis_request(message) or _looks_like_clean_data_request(message):
                route.intent = "show_clean_data"
                route.stage = "clean_data"
            elif _looks_like_model_variables_request(message):
                route.intent = "show_model_variables"
                route.stage = "model_variables"
            elif _looks_like_decision_explanation_request(message):
                route.intent = "show_decision_explanation"
                route.stage = "decision"
            elif _looks_like_semantic_lookup_request(message):
                route.intent = "show_semantic_lookup"
                route.stage = "semantic_lookup"
            elif _looks_like_latest_summary_request(message):
                route.intent = "show_latest_summary"
                route.stage = "summary"
            if route.intent == "unknown":
                route.intent = fallback.intent
            if not route.tickers:
                route.tickers = list(fallback.tickers)
            if route.start is None:
                route.start = fallback.start
            if route.end is None:
                route.end = fallback.end
            if not route.review_mode:
                route.review_mode = fallback.review_mode
            if route.language not in {"en", "es"}:
                route.language = fallback.language if fallback.language in {"en", "es"} else route.language
            elif route.language == "en" and fallback.language == "es" and _infer_requested_language(message) == "es":
                route.language = "es"
            route.experimental_groq_brain = bool(
                getattr(route, "experimental_groq_brain", False)
                or getattr(fallback, "experimental_groq_brain", False)
                or _looks_like_groq_brain_request(message)
            )
            route.compare_binance = bool(route.compare_binance or fallback.compare_binance)
            route.use_reviewer = bool(route.use_reviewer if route.review_mode != "off" else False)
            if route.experimental_groq_brain and route.intent in _RUN_INTENTS | {"compare_sources"}:
                route.review_mode = "off"
                route.use_reviewer = False
            if not route.run_id:
                route.run_id = fallback.run_id
            if route.intent not in {"run_pipeline", "compare_sources"}:
                route.needs_follow_up = False
                route.follow_up_question = ""
            route = _infer_compare_fields(route)
            route.raw_message = fallback.raw_message or message
            return route
        except Exception as exc:
            logger.debug("Groq router unavailable: %s", exc)
            return fallback

    def synthesize(self, question: str, context: Dict[str, Any], fallback_text: str) -> str:
        if not self.enabled:
            return fallback_text

        language = str(context.get("language") or "en").lower()
        es = language.startswith("es")

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a concise assistant for a market-data pipeline with one assistant and three skills: "
                        "extract, clean, model. Use only the provided context. Answer directly, without fabricating facts. "
                        "Prefer the latest run artifacts and keep the reply grounded in the routed context. "
                        "Write in simple, human language with short sentences. Avoid jargon when a plain word works. "
                        "Do not turn the reply into a checklist unless the user asked for one. "
                        "Prefer one or two short paragraphs and a plain explanation. "
                        "Keep symbol names, ticker names, mode names, and command tokens unchanged so the text is easy to translate. "
                        f"Answer in {'Spanish' if es else 'English'}."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "question": question,
                            "context": compact_for_groq(context, max_depth=2, max_items=8, max_string=300),
                            "fallback_text": fallback_text[:5000],
                        },
                        separators=(",", ":"),
                        default=str,
                    ),
                },
            ],
            "temperature": 0.2,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
            payload_json: Dict[str, Any] = response.json()
            content = payload_json["choices"][0]["message"]["content"]
            content = str(content or "").strip()
            return content or fallback_text
        except Exception as exc:
            logger.debug("Groq synthesis unavailable: %s", exc)
            return fallback_text


class AssistantRouter:
    def __init__(self, groq_router: Optional[GroqRouter] = None) -> None:
        self.groq_router = groq_router or GroqRouter()

    def route(self, message: str, state: AssistantState) -> AssistantRoute:
        normalized = _normalize_whitespace(message)
        is_greeting, routed_message = _split_greeting_prefix(normalized)
        if is_greeting and not routed_message:
            route = AssistantRoute(
                intent="greet",
                stage="greeting",
                language=_detect_language(normalized),
                raw_message=normalized,
            )
            preferred_language = str(state.preferred_language or "").lower()
            if preferred_language in {"en", "es"}:
                route.language = preferred_language
            return route
        working_message = routed_message if routed_message else normalized
        preferred_language = str(state.preferred_language or "").lower()
        session_language = preferred_language if preferred_language in {"en", "es"} else ""
        shortcut_route = _shortcut_route(working_message, state)
        if shortcut_route:
            if session_language and shortcut_route.intent != "set_language":
                shortcut_route.language = session_language
            shortcut_route.raw_message = normalized
            return shortcut_route
        if _looks_like_symbol_guide_request(working_message):
            route = AssistantRoute(
                intent="show_symbol_guide",
                stage="symbols",
                language=_detect_language(working_message),
                raw_message=normalized,
            )
            if session_language and route.intent != "set_language":
                route.language = session_language
            return route
        if _looks_like_decision_explanation_request(working_message):
            route = AssistantRoute(
                intent="show_decision_explanation",
                stage="decision",
                language=_detect_language(working_message),
                raw_message=normalized,
            )
            if session_language and route.intent != "set_language":
                route.language = session_language
            return route
        scoped_route = _run_scoped_route(working_message, state)
        if scoped_route:
            if session_language and scoped_route.intent != "set_language":
                scoped_route.language = session_language
            scoped_route.raw_message = normalized
            return scoped_route
        base = _heuristic_route(working_message, state)
        if state.pending_route and base.intent in {"unknown", "continue_task"}:
            base = _merge_from_pending(base, state.pending_route)
        base.tickers = list(dict.fromkeys(base.tickers))
        base.raw_message = normalized
        if session_language and base.intent != "set_language":
            base.language = session_language
        if self.groq_router.enabled:
            routed = self.groq_router.route(working_message, state, base)
            routed.raw_message = normalized
            if session_language and routed.intent != "set_language":
                routed.language = session_language
            return routed
        return base

    def synthesize(self, question: str, context: Dict[str, Any], fallback_text: str) -> str:
        return self.groq_router.synthesize(question, context, fallback_text)
