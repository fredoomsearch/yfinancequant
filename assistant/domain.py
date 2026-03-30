from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

_ETF_SYMBOLS = {
    "SPY",
    "QQQ",
    "DIA",
    "IWM",
    "VTI",
    "VOO",
    "ARKK",
    "EEM",
    "XLF",
    "XLK",
    "TLT",
    "GLD",
    "SLV",
}


@dataclass
class MarketIdentity:
    symbol: str
    market_type: str
    asset_type: str
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SemanticDefinition:
    key: str
    label: str
    brief_en: str
    brief_es: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def render(self, language: str = "en") -> str:
        return self.brief_es if str(language or "").lower().startswith("es") else self.brief_en


@dataclass
class AssistantIdentity:
    name: str
    role_en: str
    role_es: str
    capabilities_en: tuple[str, ...]
    capabilities_es: tuple[str, ...]

    def render(self, language: str = "en") -> str:
        return self.role_es if str(language or "").lower().startswith("es") else self.role_en

    def capability_clause(self, language: str = "en") -> str:
        es = str(language or "").lower().startswith("es")
        capabilities = self.capabilities_es if es else self.capabilities_en
        if not capabilities:
            return ""
        if len(capabilities) == 1:
            joined = capabilities[0]
        elif len(capabilities) == 2:
            joined = f"{capabilities[0]} y {capabilities[1]}" if es else f"{capabilities[0]} and {capabilities[1]}"
        else:
            tail = capabilities[-1]
            head = ", ".join(capabilities[:-1])
            joined = f"{head}, y {tail}" if es else f"{head}, and {tail}"
        if es:
            return f"Puedo ayudarte con {joined}."
        return f"I can help with {joined}."


_ASSISTANT_IDENTITY = AssistantIdentity(
    name="YFINANCE",
    role_en="a quantitative asset assistant",
    role_es="un asistente cuantitativo de activos",
    capabilities_en=(
        "runs",
        "market metrics",
        "model explanations",
        "run comparisons",
        "assistant scorecards",
        "web status",
    ),
    capabilities_es=(
        "corridas",
        "métricas de mercado",
        "explicaciones del modelo",
        "comparaciones de corridas",
        "scorecards del assistant",
        "estado web",
    ),
)


def resolve_assistant_identity() -> AssistantIdentity:
    return _ASSISTANT_IDENTITY


def resolve_market_identity(symbol: str) -> MarketIdentity:
    normalized = str(symbol or "").strip().upper()
    if not normalized:
        return MarketIdentity(symbol="", market_type="unknown", asset_type="unknown", rationale="no symbol provided")
    if normalized.endswith("=X"):
        return MarketIdentity(
            symbol=normalized,
            market_type="forex",
            asset_type="currency pair",
            rationale="symbols ending in =X are treated as forex pairs in yfinance",
        )
    if normalized.startswith("^"):
        return MarketIdentity(
            symbol=normalized,
            market_type="index",
            asset_type="market index",
            rationale="symbols starting with ^ are treated as market indexes",
        )
    if normalized.endswith(("-USD", "-USDT", "-USDC")):
        return MarketIdentity(
            symbol=normalized,
            market_type="crypto",
            asset_type="digital asset pair",
            rationale="symbols ending in -USD/-USDT/-USDC are treated as crypto pairs",
        )
    if normalized in _ETF_SYMBOLS:
        return MarketIdentity(
            symbol=normalized,
            market_type="fund",
            asset_type="ETF",
            rationale="known ETF ticker set",
        )
    return MarketIdentity(
        symbol=normalized,
        market_type="equity",
        asset_type="stock",
        rationale="default listed ticker classification",
    )


_SEMANTIC_GLOSSARY = (
    SemanticDefinition(
        key="forex",
        label="forex",
        brief_en="Forex is the global foreign exchange market where currency pairs trade against each other.",
        brief_es="Forex es el mercado global de divisas donde se negocian pares de monedas entre sí.",
    ),
    SemanticDefinition(
        key="crypto",
        label="crypto market",
        brief_en="A crypto market is a market where digital assets such as coins and tokens trade against quote currencies or other assets.",
        brief_es="Un mercado cripto es un mercado donde activos digitales como monedas y tokens se negocian contra monedas de cotización u otros activos.",
    ),
    SemanticDefinition(
        key="equity",
        label="equity market",
        brief_en="An equity market trades company shares, so the holder owns a fraction of the listed company.",
        brief_es="Un mercado de renta variable negocia acciones de empresas, por lo que el tenedor posee una fracción de la compañía listada.",
    ),
    SemanticDefinition(
        key="fund",
        label="ETF or fund",
        brief_en="An ETF is a fund traded on an exchange like a stock, usually tracking an index, sector, or strategy.",
        brief_es="Un ETF es un fondo negociado en bolsa como una acción, normalmente siguiendo un índice, sector o estrategia.",
    ),
    SemanticDefinition(
        key="index",
        label="market index",
        brief_en="A market index tracks the performance of a basket of securities to summarize a market or segment.",
        brief_es="Un índice de mercado sigue el rendimiento de una cesta de instrumentos para resumir un mercado o segmento.",
    ),
    SemanticDefinition(
        key="benchmark",
        label="benchmark",
        brief_en="A benchmark is a reference point used to judge or compare the performance of a portfolio, asset, or strategy.",
        brief_es="Un benchmark es un punto de referencia usado para juzgar o comparar el desempeño de una cartera, un activo o una estrategia.",
    ),
    SemanticDefinition(
        key="currency pair",
        label="currency pair",
        brief_en="A currency pair quotes how much of one currency is needed to buy one unit of another currency.",
        brief_es="Un par de divisas expresa cuánta cantidad de una moneda se necesita para comprar una unidad de otra.",
    ),
    SemanticDefinition(
        key="digital asset pair",
        label="digital asset pair",
        brief_en="A digital asset pair quotes a crypto asset against a quote currency such as USD or a stablecoin.",
        brief_es="Un par de activo digital cotiza un criptoactivo frente a una moneda de referencia como USD o una stablecoin.",
    ),
    SemanticDefinition(
        key="stock",
        label="stock",
        brief_en="A stock represents ownership shares in a listed company.",
        brief_es="Una acción representa participaciones de propiedad en una empresa listada.",
    ),
    SemanticDefinition(
        key="ETF",
        label="ETF",
        brief_en="An ETF is a basket of assets packaged into a security that trades intraday on an exchange.",
        brief_es="Un ETF es una cesta de activos empaquetada en un instrumento que cotiza intradía en bolsa.",
    ),
    SemanticDefinition(
        key="market index",
        label="market index",
        brief_en="A market index is a benchmark built from multiple securities to track a market or theme.",
        brief_es="Un índice de mercado es un benchmark construido con múltiples instrumentos para seguir un mercado o tema.",
    ),
    SemanticDefinition(
        key="hold",
        label="hold",
        brief_en="Hold means the model does not see enough edge to justify a long or short action right now.",
        brief_es="Hold significa que el modelo no ve suficiente ventaja para justificar una acción long o short en este momento.",
    ),
    SemanticDefinition(
        key="long",
        label="long",
        brief_en="Long means taking a bullish position that benefits if the price rises.",
        brief_es="Long significa tomar una posición alcista que se beneficia si el precio sube.",
    ),
    SemanticDefinition(
        key="short",
        label="short",
        brief_en="Short means taking a bearish position that benefits if the price falls.",
        brief_es="Short significa tomar una posición bajista que se beneficia si el precio cae.",
    ),
    SemanticDefinition(
        key="ticker",
        label="ticker",
        brief_en="A ticker is the exchange symbol used to identify a tradable instrument.",
        brief_es="Un ticker es el símbolo bursátil usado para identificar un instrumento negociable.",
    ),
    SemanticDefinition(
        key="yfinance",
        label="yfinance",
        brief_en="yfinance is the Python library this project uses to fetch market data from Yahoo Finance and build the run artifacts.",
        brief_es="yfinance es la biblioteca de Python que este proyecto usa para traer datos de mercado desde Yahoo Finance y construir los artefactos de la corrida.",
    ),
    SemanticDefinition(
        key="dataset",
        label="dataset",
        brief_en="A dataset is a collection of rows and columns prepared for analysis, validation, or modeling.",
        brief_es="Un dataset es una colección de filas y columnas preparada para análisis, validación o modelado.",
    ),
    SemanticDefinition(
        key="artifact",
        label="artifact",
        brief_en="An artifact is a file or structured output produced by a run, such as summary.json, manifest.json, or clean_market_data.csv.",
        brief_es="Un artifact es un archivo o salida estructurada producida por una corrida, como summary.json, manifest.json o clean_market_data.csv.",
    ),
    SemanticDefinition(
        key="manifest",
        label="manifest",
        brief_en="A manifest is the structured record that describes the request, artifacts, and main outputs of a run.",
        brief_es="Un manifest es el registro estructurado que describe la solicitud, los artifacts y las salidas principales de una corrida.",
    ),
    SemanticDefinition(
        key="target",
        label="target",
        brief_en="A target is the label the model tries to predict, such as target_direction in this project.",
        brief_es="Un target es la etiqueta que el modelo intenta predecir, como target_direction en este proyecto.",
    ),
    SemanticDefinition(
        key="feature engineering",
        label="feature engineering",
        brief_en="Feature engineering is the process of turning raw data into derived signals or variables that are more useful to the model.",
        brief_es="Feature engineering es el proceso de convertir datos raw en señales o variables derivadas que sean más útiles para el modelo.",
    ),
    SemanticDefinition(
        key="grounding",
        label="grounding",
        brief_en="Grounding means answering from evidence the system can point to, such as local artifacts, local tools, or explicit external facts.",
        brief_es="Grounding significa responder desde evidencia que el sistema puede señalar, como artifacts locales, herramientas locales o facts externos explícitos.",
    ),
    SemanticDefinition(
        key="memory",
        label="memory",
        brief_en="Memory is the layer that keeps useful context between turns, such as the active run, symbol, stage, or prior semantic subject.",
        brief_es="Memory es la capa que conserva contexto útil entre turnos, como la corrida activa, el símbolo, la etapa o el sujeto semántico previo.",
    ),
    SemanticDefinition(
        key="artifact store",
        label="artifact store",
        brief_en="An artifact store is the place where run outputs are kept so the assistant can reload summaries, manifests, datasets, and other evidence.",
        brief_es="Un artifact store es el lugar donde se guardan las salidas de las corridas para que el assistant pueda recargar resúmenes, manifests, datasets y otra evidencia.",
    ),
    SemanticDefinition(
        key="evidence ledger",
        label="evidence ledger",
        brief_en="An evidence ledger is the trace of what evidence was used, linked, or cited to support a reply or a decision path.",
        brief_es="Un evidence ledger es el rastro de qué evidencia fue usada, enlazada o citada para sostener una respuesta o una ruta de decisión.",
    ),
    SemanticDefinition(
        key="drift",
        label="drift",
        brief_en="Drift is a meaningful change in data, features, or behavior over time that can make a model or policy less reliable.",
        brief_es="Drift es un cambio relevante en los datos, las variables o el comportamiento a lo largo del tiempo que puede volver menos confiable a un modelo o una política.",
    ),
    SemanticDefinition(
        key="conversational layer",
        label="conversational layer",
        brief_en="The conversational layer is the part of the assistant that keeps track of intent, follow-ups, topic shifts, and what the user means across turns.",
        brief_es="La capa conversacional es la parte del assistant que mantiene la intención, los follow-ups, los cambios de tema y lo que el usuario quiso decir entre turnos.",
    ),
    SemanticDefinition(
        key="validator",
        label="validator",
        brief_en="A validator is the component that tests whether a model, policy, or candidate change meets the required checks before it is accepted.",
        brief_es="Un validator es el componente que prueba si un modelo, política o cambio candidato cumple los chequeos requeridos antes de ser aceptado.",
    ),
    SemanticDefinition(
        key="shadow run",
        label="shadow run",
        brief_en="A shadow run is a parallel run that observes how a candidate would behave without letting that candidate affect production decisions.",
        brief_es="Un shadow run es una corrida paralela que observa cómo se comportaría un candidato sin dejar que ese candidato afecte las decisiones de producción.",
    ),
    SemanticDefinition(
        key="promotion gate",
        label="promotion gate",
        brief_en="A promotion gate is the checkpoint that decides whether a candidate is allowed to move closer to or into production after validation.",
        brief_es="Un promotion gate es el checkpoint que decide si un candidato puede acercarse a producción o entrar en producción después de la validación.",
    ),
    SemanticDefinition(
        key="challenger",
        label="challenger",
        brief_en="A challenger is the candidate model or policy trying to beat the current production baseline.",
        brief_es="Un challenger es el modelo o política candidata que intenta superar a la baseline actual de producción.",
    ),
    SemanticDefinition(
        key="champion",
        label="champion",
        brief_en="A champion is the current production model or policy that acts as the baseline to beat.",
        brief_es="Un champion es el modelo o política actual en producción que actúa como la baseline a superar.",
    ),
    SemanticDefinition(
        key="policy engine",
        label="policy engine",
        brief_en="A policy engine is the rules layer that decides what actions, tool calls, or promotions are allowed under explicit conditions.",
        brief_es="Un policy engine es la capa de reglas que decide qué acciones, llamadas a tools o promociones están permitidas bajo condiciones explícitas.",
    ),
    SemanticDefinition(
        key="retraining scheduler",
        label="retraining scheduler",
        brief_en="A retraining scheduler is the component that decides or schedules when a model should be retrained under approved criteria.",
        brief_es="Un retraining scheduler es el componente que decide o agenda cuándo debe reentrenarse un modelo bajo criterios aprobados.",
    ),
    SemanticDefinition(
        key="feature registry",
        label="feature registry",
        brief_en="A feature registry is the versioned catalog of approved features, definitions, and metadata that the system is allowed to use.",
        brief_es="Un feature registry es el catálogo versionado de features aprobadas, definiciones y metadata que el sistema tiene permitido usar.",
    ),
    SemanticDefinition(
        key="adaptive selector",
        label="adaptive selector",
        brief_en="An adaptive selector is the component that chooses among approved models, policies, or modes using current context and governed rules.",
        brief_es="Un adaptive selector es el componente que elige entre modelos, políticas o modos aprobados usando el contexto actual y reglas gobernadas.",
    ),
    SemanticDefinition(
        key="shadow runner",
        label="shadow runner",
        brief_en="A shadow runner is the execution component that launches and records shadow runs so candidates can be observed safely beside production.",
        brief_es="Un shadow runner es el componente de ejecución que lanza y registra shadow runs para observar candidatos de forma segura junto a producción.",
    ),
    SemanticDefinition(
        key="promotion policy",
        label="promotion policy",
        brief_en="A promotion policy is the explicit rule set that defines when a validated candidate may be promoted toward production.",
        brief_es="Una promotion policy es el conjunto explícito de reglas que define cuándo un candidato validado puede ser promovido hacia producción.",
    ),
    SemanticDefinition(
        key="raw column",
        label="raw column",
        brief_en="A raw column is a field extracted directly from the source before cleaning, normalization, or feature engineering.",
        brief_es="Una columna raw es un campo extraído directamente desde la fuente antes de la limpieza, la normalización o la ingeniería de variables.",
    ),
    SemanticDefinition(
        key="clean column",
        label="clean column",
        brief_en="A clean column is a stabilized field kept after typing, deduplication, cleaning rules, or derived-signal generation.",
        brief_es="Una columna limpia es un campo estabilizado después del tipado, la deduplicación, las reglas de limpieza o la generación de señales derivadas.",
    ),
    SemanticDefinition(
        key="column",
        label="column",
        brief_en="A column is one field of a table or dataframe, such as close, volume, or ticker, repeated across many rows.",
        brief_es="Una columna es un campo de una tabla o dataframe, como close, volume o ticker, repetido a lo largo de muchas filas.",
    ),
    SemanticDefinition(
        key="row",
        label="row",
        brief_en="A row is one record or observation in a table, grouping all columns for a given date, ticker, or entity.",
        brief_es="Una fila es un registro u observación en una tabla, agrupando todas las columnas para una fecha, ticker o entidad dada.",
    ),
    SemanticDefinition(
        key="schema",
        label="schema",
        brief_en="A schema is the structural contract of a dataset: which columns exist, what types they have, and what shape is expected.",
        brief_es="Un esquema es el contrato estructural de un dataset: qué columnas existen, qué tipos tienen y qué forma se espera.",
    ),
    SemanticDefinition(
        key="model variable",
        label="model variable",
        brief_en="A model variable is a column or derived signal that actually enters the model after preprocessing and selection.",
        brief_es="Una variable de modelo es una columna o señal derivada que realmente entra al modelo después del preprocesamiento y la selección.",
    ),
    SemanticDefinition(
        key="variable",
        label="variable",
        brief_en="A variable is a field the system uses to describe, transform, or model data. In modeling, variables are the inputs given to the model.",
        brief_es="Una variable es un campo que el sistema usa para describir, transformar o modelar datos. En modelado, las variables son las entradas que recibe el modelo.",
    ),
    SemanticDefinition(
        key="adj_close",
        label="adj_close",
        brief_en="adj_close means adjusted close, the close price adjusted for events such as splits and dividends when the data source provides them.",
        brief_es="adj_close significa adjusted close, el precio de cierre ajustado por eventos como splits y dividendos cuando la fuente de datos los provee.",
    ),
    SemanticDefinition(
        key="arbitrage",
        label="arbitrage",
        brief_en="Arbitrage means exploiting a price difference for the same asset across markets or venues.",
        brief_es="Arbitrage significa aprovechar una diferencia de precio del mismo activo entre mercados o plataformas.",
    ),
    SemanticDefinition(
        key="hedge",
        label="hedge",
        brief_en="A hedge is a position or instrument used to reduce exposure to an unwanted risk.",
        brief_es="Un hedge es una posición o instrumento usado para reducir exposición a un riesgo no deseado.",
    ),
    SemanticDefinition(
        key="spread",
        label="spread",
        brief_en="A spread is the difference between two prices, yields, or rates, often bid versus ask.",
        brief_es="Un spread es la diferencia entre dos precios, rendimientos o tasas, a menudo bid versus ask.",
    ),
    SemanticDefinition(
        key="liquidity",
        label="liquidity",
        brief_en="Liquidity is how easily an asset can be traded without moving its price too much.",
        brief_es="La liquidez es qué tan fácil puede negociarse un activo sin mover demasiado su precio.",
    ),
    SemanticDefinition(
        key="drawdown",
        label="drawdown",
        brief_en="A drawdown is the drop from a peak value to a lower value over a period of time.",
        brief_es="Un drawdown es la caída desde un valor máximo hasta un valor menor durante un periodo.",
    ),
    SemanticDefinition(
        key="momentum",
        label="momentum",
        brief_en="Momentum means the tendency of price movement to persist in the same direction for a while.",
        brief_es="Momentum significa la tendencia de un movimiento de precio a persistir en la misma dirección por un tiempo.",
    ),
    SemanticDefinition(
        key="mean reversion",
        label="mean reversion",
        brief_en="Mean reversion is the idea that prices tend to move back toward an average after extremes.",
        brief_es="Mean reversion es la idea de que los precios tienden a volver hacia un promedio tras extremos.",
    ),
    SemanticDefinition(
        key="slippage",
        label="slippage",
        brief_en="Slippage is the difference between the expected execution price and the actual execution price.",
        brief_es="Slippage es la diferencia entre el precio de ejecución esperado y el precio real ejecutado.",
    ),
    SemanticDefinition(
        key="volatility",
        label="volatility",
        brief_en="Volatility is how much and how quickly a price moves around its average over time.",
        brief_es="La volatilidad es cuánto y qué tan rápido se mueve un precio alrededor de su promedio con el tiempo.",
    ),
    SemanticDefinition(
        key="return",
        label="return",
        brief_en="Return is the gain or loss produced by an investment over a period of time.",
        brief_es="El retorno es la ganancia o pérdida producida por una inversión durante un periodo.",
    ),
    SemanticDefinition(
        key="risk",
        label="risk",
        brief_en="Risk is the chance that an outcome differs from what you expect, usually in a harmful way.",
        brief_es="El riesgo es la posibilidad de que un resultado difiera de lo esperado, normalmente de forma desfavorable.",
    ),
    SemanticDefinition(
        key="alpha",
        label="alpha",
        brief_en="Alpha is the extra return a strategy generates compared with a benchmark after adjusting for risk.",
        brief_es="Alpha es el retorno extra que genera una estrategia frente a un benchmark, ajustado por riesgo.",
    ),
    SemanticDefinition(
        key="beta",
        label="beta",
        brief_en="Beta measures how sensitive an asset or strategy is to market movements relative to a benchmark.",
        brief_es="Beta mide cuán sensible es un activo o estrategia a los movimientos del mercado frente a un benchmark.",
    ),
    SemanticDefinition(
        key="correlation",
        label="correlation",
        brief_en="Correlation measures how strongly two variables move together, and in what direction.",
        brief_es="La correlación mide qué tan fuertemente dos variables se mueven juntas y en qué dirección.",
    ),
    SemanticDefinition(
        key="leverage",
        label="leverage",
        brief_en="Leverage means using borrowed capital or exposure amplification to increase potential gains and losses.",
        brief_es="Leverage significa usar capital prestado o amplificación de exposición para aumentar ganancias y pérdidas potenciales.",
    ),
    SemanticDefinition(
        key="portfolio",
        label="portfolio",
        brief_en="A portfolio is the collection of positions and exposures held together by an investor or strategy.",
        brief_es="Una cartera es la colección de posiciones y exposiciones mantenidas en conjunto por un inversor o una estrategia.",
    ),
    SemanticDefinition(
        key="position",
        label="position",
        brief_en="A position is the quantity and direction held in an asset.",
        brief_es="Una posición es la cantidad y la dirección mantenidas en un activo.",
    ),
    SemanticDefinition(
        key="exposure",
        label="exposure",
        brief_en="Exposure is how much portfolio value or risk is tied to an asset, factor, or market.",
        brief_es="La exposición es cuánta parte del valor o riesgo de la cartera está ligada a un activo, factor o mercado.",
    ),
    SemanticDefinition(
        key="factor",
        label="factor",
        brief_en="A factor is a driver of return or risk that explains part of an asset's behavior.",
        brief_es="Un factor es un impulsor de retorno o riesgo que explica parte del comportamiento de un activo.",
    ),
    SemanticDefinition(
        key="signal",
        label="signal",
        brief_en="A signal is a rule or model output that suggests an action or condition.",
        brief_es="Una señal es una regla o salida de modelo que sugiere una acción o condición.",
    ),
    SemanticDefinition(
        key="trend",
        label="trend",
        brief_en="A trend is the general direction of price movement over time.",
        brief_es="Una tendencia es la dirección general del movimiento del precio a lo largo del tiempo.",
    ),
    SemanticDefinition(
        key="variance",
        label="variance",
        brief_en="Variance measures how spread out returns are around their average.",
        brief_es="La varianza mide qué tan dispersos están los retornos alrededor de su promedio.",
    ),
    SemanticDefinition(
        key="standard deviation",
        label="standard deviation",
        brief_en="Standard deviation measures return dispersion and is the square root of variance.",
        brief_es="La desviación estándar mide la dispersión de los retornos y es la raíz cuadrada de la varianza.",
    ),
    SemanticDefinition(
        key="sharpe ratio",
        label="sharpe ratio",
        brief_en="The Sharpe ratio measures excess return per unit of total risk.",
        brief_es="El ratio de Sharpe mide el retorno excedente por unidad de riesgo total.",
    ),
    SemanticDefinition(
        key="sortino ratio",
        label="sortino ratio",
        brief_en="The Sortino ratio measures excess return per unit of downside risk.",
        brief_es="El ratio de Sortino mide el retorno excedente por unidad de riesgo a la baja.",
    ),
    SemanticDefinition(
        key="covariance",
        label="covariance",
        brief_en="Covariance measures how two return series move together in raw units.",
        brief_es="La covarianza mide cómo dos series de retornos se mueven juntas en unidades crudas.",
    ),
    SemanticDefinition(
        key="pnl",
        label="PnL",
        brief_en="PnL stands for profit and loss, the net money gained or lost over a period or trade.",
        brief_es="PnL significa profit and loss, la ganancia o pérdida neta obtenida durante un periodo o una operación.",
    ),
    SemanticDefinition(
        key="excess return",
        label="excess return",
        brief_en="Excess return is the return above a benchmark or reference rate.",
        brief_es="El exceso de retorno es el retorno por encima de un benchmark o tasa de referencia.",
    ),
    SemanticDefinition(
        key="benchmark return",
        label="benchmark return",
        brief_en="Benchmark return is the return produced by the reference benchmark over the same period.",
        brief_es="El retorno del benchmark es el retorno producido por la referencia durante el mismo periodo.",
    ),
    SemanticDefinition(
        key="tracking error",
        label="tracking error",
        brief_en="Tracking error measures how much a portfolio's returns deviate from its benchmark over time.",
        brief_es="Tracking error mide cuánto se desvían los retornos de una cartera de su benchmark con el tiempo.",
    ),
    SemanticDefinition(
        key="information ratio",
        label="information ratio",
        brief_en="The information ratio measures excess return per unit of tracking error.",
        brief_es="El information ratio mide el exceso de retorno por unidad de tracking error.",
    ),
)

_SEMANTIC_ALIAS_MAP = {
    "forex": ("forex", "fx", "currency pair", "divisas"),
    "crypto": ("crypto", "cryptocurrency", "bitcoin", "token", "cripto"),
    "equity": ("equity market", "equity", "stock market", "share market", "mercado accionario", "mercado de renta variable"),
    "fund": ("etf", "fund", "exchange traded fund", "fondo"),
    "index": ("index", "índice", "indice"),
    "benchmark": ("benchmark", "reference benchmark", "comparison benchmark", "punto de referencia", "referencia"),
    "currency pair": ("currency pair", "par de divisas"),
    "digital asset pair": ("digital asset pair", "par de activo digital"),
    "stock": ("stock", "share", "acción", "accion"),
    "ETF": ("etf",),
    "market index": ("market index", "índice de mercado", "indice de mercado"),
    "hold": ("hold",),
    "long": ("long",),
    "short": ("short",),
    "ticker": ("ticker", "symbol", "símbolo", "simbolo"),
    "yfinance": ("yfinance", "yfinance library", "yfinance package", "biblioteca yfinance", "librería yfinance"),
    "dataset": ("dataset", "datasets", "data set", "data sets"),
    "artifact": ("artifact", "artifacts", "artefact", "artefacts", "artefacto", "artefactos"),
    "manifest": ("manifest", "run manifest"),
    "target": ("target", "label", "objective variable", "variable objetivo"),
    "feature engineering": ("feature engineering", "engineering features", "ingeniería de variables", "ingenieria de variables"),
    "grounding": ("grounding", "grounded answer", "grounded responses"),
    "memory": ("memory", "assistant memory", "session memory", "memoria"),
    "artifact store": ("artifact store", "artifacts store", "store of artifacts", "almacen de artifacts", "almacén de artifacts"),
    "evidence ledger": ("evidence ledger", "ledger of evidence", "registro de evidencia", "bitácora de evidencia", "bitacora de evidencia"),
    "drift": ("drift", "model drift", "data drift", "feature drift"),
    "conversational layer": ("conversational layer", "conversation layer", "capa conversacional"),
    "validator": ("validator", "validation gate", "validador"),
    "shadow run": ("shadow run", "shadow runs", "shadow execution", "shadow test", "corrida sombra", "ejecución sombra", "ejecucion sombra"),
    "promotion gate": ("promotion gate", "promotion gates", "gate to production", "gating", "puerta de promoción", "puerta de promocion", "gate de promoción", "gate de promocion"),
    "challenger": ("challenger", "candidate challenger", "modelo challenger"),
    "champion": ("champion", "production champion", "baseline champion", "modelo champion"),
    "policy engine": ("policy engine", "engine of policies", "motor de políticas", "motor de politicas"),
    "retraining scheduler": ("retraining scheduler", "retrain scheduler", "scheduler for retraining", "programador de reentrenamiento"),
    "feature registry": ("feature registry", "registry of features", "feature catalog", "registro de features", "registro de variables"),
    "adaptive selector": ("adaptive selector", "selector adaptativo", "adaptive routing selector"),
    "shadow runner": ("shadow runner", "runner of shadow runs", "ejecutor sombra", "shadow execution runner"),
    "promotion policy": ("promotion policy", "policy for promotion", "política de promoción", "politica de promocion"),
    "raw column": ("raw column", "raw columns", "columna raw", "columna cruda", "columnas crudas", "source column", "source columns"),
    "clean column": ("clean column", "clean columns", "cleaned column", "cleaned columns", "columna limpia", "columnas limpias"),
    "column": ("column", "columns", "columna", "columnas", "field", "fields", "campo", "campos"),
    "row": ("row", "rows", "fila", "filas", "record", "records", "registro", "registros"),
    "schema": ("schema", "esquema", "structure", "estructura", "dataset structure", "data schema"),
    "model variable": ("model variable", "model variables", "input variable", "input variables", "model input", "model inputs", "variable de modelo", "variables de modelo"),
    "variable": ("variable", "variables", "feature", "features", "variable explicativa", "variables explicativas"),
    "adj_close": ("adj_close", "adj close", "adjusted close", "cierre ajustado"),
    "arbitrage": ("arbitrage", "arbitraje"),
    "hedge": ("hedge", "cobertura"),
    "spread": ("spread", "diferencial"),
    "liquidity": ("liquidity", "liquidez"),
    "drawdown": ("drawdown", "caída", "caida"),
    "momentum": ("momentum",),
    "mean reversion": ("mean reversion", "reversión a la media", "reversion a la media"),
    "slippage": ("slippage", "deslizamiento"),
    "volatility": ("volatility", "volatilidad"),
    "return": ("return", "retorno", "rendimiento"),
    "risk": ("risk", "riesgo"),
    "alpha": ("alpha",),
    "beta": ("beta",),
    "correlation": ("correlation", "correlación", "correlacion"),
    "leverage": ("leverage", "apalancamiento"),
    "portfolio": ("portfolio", "cartera", "portafolio"),
    "position": ("position", "holding", "posicion", "posición"),
    "exposure": ("exposure", "exposicion", "exposición"),
    "factor": ("factor", "risk factor", "driver", "factor de riesgo"),
    "signal": ("signal", "señal", "senal", "trade signal"),
    "trend": ("trend", "tendencia"),
    "variance": ("variance", "varianza"),
    "standard deviation": ("standard deviation", "desviación estándar", "desviacion estandar", "std dev"),
    "sharpe ratio": ("sharpe ratio", "ratio sharpe"),
    "sortino ratio": ("sortino ratio", "ratio sortino"),
    "covariance": ("covariance", "covarianza"),
    "pnl": ("pnl", "profit and loss", "beneficio y pérdida", "beneficio y perdida"),
    "excess return": ("excess return", "retorno excedente", "exceso de retorno"),
    "benchmark return": ("benchmark return", "retorno benchmark", "retorno de benchmark"),
    "tracking error": ("tracking error", "error de seguimiento"),
    "information ratio": ("information ratio", "ratio de información", "ratio de informacion"),
}


def resolve_semantic_definition_for_term(term: str) -> SemanticDefinition | None:
    normalized = str(term or "").strip().lower()
    if not normalized:
        return None
    for entry in _SEMANTIC_GLOSSARY:
        aliases = _SEMANTIC_ALIAS_MAP.get(entry.key, ())
        if normalized == entry.key.lower() or any(alias == normalized for alias in aliases):
            return entry
    return None


def compare_semantic_definitions(left: SemanticDefinition, right: SemanticDefinition, language: str = "en") -> str:
    es = str(language or "").lower().startswith("es")

    market_frame_keys = {
        "forex",
        "crypto",
        "equity",
        "fund",
        "index",
        "currency pair",
        "digital asset pair",
        "stock",
        "ETF",
        "market index",
    }
    trading_mechanics_keys = {
        "hold",
        "long",
        "short",
        "arbitrage",
        "hedge",
        "spread",
        "liquidity",
        "drawdown",
        "momentum",
        "mean reversion",
        "slippage",
    }
    performance_metrics_keys = {
        "volatility",
        "return",
        "risk",
        "alpha",
        "beta",
        "correlation",
        "leverage",
        "variance",
        "standard deviation",
        "sharpe ratio",
        "sortino ratio",
        "covariance",
        "pnl",
        "excess return",
        "benchmark return",
        "tracking error",
        "information ratio",
    }
    portfolio_analytics_keys = {
        "benchmark",
        "portfolio",
        "position",
        "exposure",
        "factor",
        "signal",
        "trend",
    }
    governance_keys = {
        "validator",
        "shadow run",
        "promotion gate",
        "challenger",
        "champion",
        "policy engine",
        "retraining scheduler",
        "feature registry",
        "adaptive selector",
        "shadow runner",
        "promotion policy",
    }
    data_structure_keys = {
        "dataset",
        "artifact",
        "manifest",
        "target",
        "feature engineering",
        "grounding",
        "memory",
        "artifact store",
        "evidence ledger",
        "drift",
        "conversational layer",
        "raw column",
        "clean column",
        "column",
        "row",
        "schema",
        "model variable",
        "variable",
    }

    if left.key == right.key:
        return left.render(language)

    if {left.key, right.key} == {"benchmark", "portfolio"}:
        common = (
            "Ambos describen la relación entre una referencia y una cartera en evaluación."
            if es
            else "Both describe the relationship between a reference benchmark and the portfolio being evaluated."
        )
    elif {left.key, right.key} == {"portfolio", "position"}:
        common = (
            "Ambos describen la estructura de una cartera y una exposición individual dentro de ella."
            if es
            else "Both describe portfolio structure and a single holding inside it."
        )
    elif {left.key, right.key} == {"variance", "standard deviation"}:
        common = (
            "Ambos describen dispersión de retornos; la desviación estándar es la raíz cuadrada de la varianza."
            if es
            else "Both describe return dispersion; standard deviation is the square root of variance."
        )
    elif {left.key, right.key} == {"sharpe ratio", "sortino ratio"}:
        common = (
            "Ambos comparan retorno excedente con riesgo, pero uno usa riesgo total y el otro riesgo a la baja."
            if es
            else "Both compare excess return with risk, but one uses total risk and the other downside risk."
        )
    elif {left.key, right.key} == {"pnl", "return"}:
        common = (
            "Ambos hablan del resultado económico, pero PnL es dinero neto y return es desempeño relativo."
            if es
            else "Both describe economic outcome, but PnL is net money and return is relative performance."
        )
    elif {left.key, right.key} == {"tracking error", "information ratio"}:
        common = (
            "Ambos evalúan la relación con un benchmark; uno mide desviación y el otro retorno excedente por unidad de desviación."
            if es
            else "Both evaluate benchmark-relative behavior; one measures deviation and the other excess return per unit of deviation."
        )
    elif {left.key, right.key} == {"raw column", "model variable"}:
        common = (
            "Ambos describen campos del pipeline, pero uno viene directo de la fuente y el otro es lo que realmente entra al modelo."
            if es
            else "Both describe pipeline fields, but one comes directly from the source and the other is what actually enters the model."
        )
    elif {left.key, right.key} == {"row", "schema"}:
        common = (
            "Ambos describen estructura de datos, pero una fila es un registro y el esquema es la forma esperada del dataset."
            if es
            else "Both describe data structure, but a row is a record and a schema is the expected shape of the dataset."
        )
    elif {left.key, right.key} == {"column", "variable"}:
        common = (
            "Ambos describen piezas de una tabla o pipeline, pero no cumplen exactamente el mismo rol."
            if es
            else "Both describe parts of a table or pipeline, but they do not play exactly the same role."
        )
    elif {left.key, right.key} == {"column", "model variable"}:
        common = (
            "Ambos pueden verse como campos de una tabla, pero la variable de modelo es la parte de esos campos que termina alimentando al modelo."
            if es
            else "Both can be seen as table fields, but a model variable is the subset of those fields that ultimately feeds the model."
        )
    elif {left.key, right.key} == {"challenger", "champion"}:
        common = (
            "Ambos describen roles de evaluación, pero el champion es la baseline actual y el challenger es el candidato que intenta superarla."
            if es
            else "Both describe evaluation roles, but the champion is the current baseline and the challenger is the candidate trying to beat it."
        )
    elif {left.key, right.key} == {"shadow run", "promotion gate"}:
        common = (
            "Ambos protegen producción, pero un shadow run observa sin impactar y un promotion gate decide si un cambio puede avanzar."
            if es
            else "Both protect production, but a shadow run observes without impact and a promotion gate decides whether a change can advance."
        )
    elif {left.key, right.key} == {"validator", "promotion gate"}:
        common = (
            "Ambos controlan calidad antes de producción, pero el validator prueba y el promotion gate decide."
            if es
            else "Both control quality before production, but the validator tests and the promotion gate decides."
        )
    elif {left.key, right.key} == {"shadow run", "shadow runner"}:
        common = (
            "Ambos pertenecen al mismo flujo, pero el shadow run es la corrida observada y el shadow runner es el componente que la ejecuta."
            if es
            else "Both belong to the same flow, but the shadow run is the observed run and the shadow runner is the component that executes it."
        )
    elif {left.key, right.key} == {"feature registry", "adaptive selector"}:
        common = (
            "Ambos ayudan a gobernar adaptatividad, pero uno define qué features están aprobadas y el otro elige entre opciones permitidas."
            if es
            else "Both help govern adaptivity, but one defines which features are approved and the other chooses among allowed options."
        )
    elif left.key in market_frame_keys and right.key in market_frame_keys:
        common = "Ambos describen piezas del mercado y del activo subyacente." if es else "Both describe parts of market structure and the underlying asset."
    elif left.key in trading_mechanics_keys and right.key in trading_mechanics_keys:
        common = "Ambos describen mecánicas de trading o gestión de riesgo." if es else "Both describe trading mechanics or risk management."
    elif left.key in performance_metrics_keys and right.key in performance_metrics_keys:
        common = "Ambos describen métricas o ideas de rendimiento y riesgo." if es else "Both describe performance and risk concepts or metrics."
    elif left.key in portfolio_analytics_keys and right.key in portfolio_analytics_keys:
        common = (
            "Ambos describen construcción de cartera, exposición o evaluación de desempeño."
            if es
            else "Both describe portfolio construction, exposure management, or performance evaluation."
        )
    elif left.key in governance_keys and right.key in governance_keys:
        common = (
            "Ambos describen gobierno cuantitativo, validación o promoción controlada de cambios."
            if es
            else "Both describe quantitative governance, validation, or controlled promotion of changes."
        )
    elif left.key in data_structure_keys and right.key in data_structure_keys:
        common = (
            "Ambos describen estructura o roles de datos dentro del pipeline."
            if es
            else "Both describe data structure or data roles inside the pipeline."
        )
    else:
        common = "Ambos son conceptos financieros relacionados, pero cumplen roles distintos." if es else "Both are related financial concepts, but they play different roles."

    return (
        f"{common} "
        f"{left.label}: {left.render(language)} "
        f"{'vs' if not es else 'contra'} "
        f"{right.label}: {right.render(language)}"
    )


def resolve_semantic_definition(message: str, symbol: str = "") -> SemanticDefinition | None:
    normalized = str(message or "").strip().lower()
    identity = resolve_market_identity(symbol) if symbol else None

    best_entry: SemanticDefinition | None = None
    best_score = (-1, -1)
    for entry in _SEMANTIC_GLOSSARY:
        aliases = _SEMANTIC_ALIAS_MAP.get(entry.key, ())
        for alias in aliases:
            alias_normalized = str(alias or "").strip().lower()
            if not alias_normalized or alias_normalized not in normalized:
                continue
            exact = int(normalized == alias_normalized)
            score = (exact, len(alias_normalized))
            if score > best_score:
                best_entry = entry
                best_score = score

    if best_entry:
        return best_entry

    if identity:
        if any(term in normalized for term in ("mercado", "market", "asset class", "tipo de mercado", "asset type", "tipo de activo")):
            for entry in _SEMANTIC_GLOSSARY:
                if entry.key == identity.market_type:
                    return entry
        if any(term in normalized for term in ("símbolo", "simbolo", "symbol", "ticker", "activo", "asset")):
            for entry in _SEMANTIC_GLOSSARY:
                if entry.key == identity.asset_type:
                    return entry

    return None
