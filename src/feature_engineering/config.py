"""Configuração da Feature Engineering: caminhos, pesos e limiares parametrizáveis.

Centraliza todas as constantes que governam *como* a camada Trusted é enriquecida,
mantendo os módulos de feature (`temporal`, `severity`, ...) livres de números mágicos.
Reaproveita os caminhos e parâmetros gerais já definidos em `src.config`.
"""

from __future__ import annotations

from pathlib import Path

from src import config

# --------------------------------------------------------------------------- #
# Caminhos de saída (camada Analytics e relatório)
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = config.PROJECT_ROOT
TRUSTED_FILE: Path = config.TRUSTED_FILE
LOG_DIR: Path = config.LOG_DIR
PARQUET_COMPRESSION: str = config.PARQUET_COMPRESSION

ANALYTICS_DIR: Path = config.DATA_DIR / "analytics"
ACIDENTES_ANALYTICS_FILE: Path = ANALYTICS_DIR / "acidentes_analytics.parquet"
TRECHOS_ANALYTICS_FILE: Path = ANALYTICS_DIR / "trechos_analytics.parquet"

REPORT_DIR: Path = PROJECT_ROOT / "outputs" / "feature_engineering"
REPORT_FILE: Path = REPORT_DIR / "relatorio_feature_engineering.md"

# --------------------------------------------------------------------------- #
# Índice de gravidade — pesos por tipo de vítima (facilmente configuráveis)
# --------------------------------------------------------------------------- #
# Justificativa (EDA): as contagens de vítimas têm baixa correlação independente entre
# si; um índice combinado pondera a severidade melhor que cada coluna isolada. Mortos
# pesam mais que feridos graves, que pesam mais que feridos leves.
PESO_MORTOS: int = 12
PESO_FERIDOS_GRAVES: int = 6
PESO_FERIDOS_LEVES: int = 2

# Faixas da classe de gravidade derivada do índice (borda superior inclusiva).
# Índice 0–2 → Baixa; 3–8 → Média; 9–20 → Alta; >20 → Crítica.
CLASSE_BAIXA_MAX: int = 2
CLASSE_MEDIA_MAX: int = 8
CLASSE_ALTA_MAX: int = 20
CLASSE_LABELS: dict[str, str] = {
    "baixa": "Baixa",
    "media": "Média",
    "alta": "Alta",
    "critica": "Crítica",
}

# --------------------------------------------------------------------------- #
# Features temporais
# --------------------------------------------------------------------------- #
# Mapeamento hora → turno (faixas fechadas à esquerda e à direita).
# 00–05 Madrugada · 06–11 Manhã · 12–17 Tarde · 18–23 Noite.
TURNO_MADRUGADA_MAX: int = 5
TURNO_MANHA_MAX: int = 11
TURNO_TARDE_MAX: int = 17
TURNO_LABELS: dict[str, str] = {
    "madrugada": "Madrugada",
    "manha": "Manhã",
    "tarde": "Tarde",
    "noite": "Noite",
}

# Dias de fim de semana na convenção do Polars `dt.weekday()` (1=segunda … 7=domingo).
FIM_DE_SEMANA_WEEKDAYS: set[int] = {6, 7}

# Fases do dia de baixa luminosidade (associadas a maior letalidade na EDA).
FASE_DIA_NOTURNA: set[str] = {"Amanhecer", "Anoitecer", "Plena Noite"}

# --------------------------------------------------------------------------- #
# Identificador de trecho
# --------------------------------------------------------------------------- #
# Tamanho da faixa de km (em km) usada para discretizar a posição na rodovia.
# trecho = UF_BR_<km_faixa>. Faixas de 1 km são fisicamente corretas (a mesma BR cruza
# vários estados) e granulares o bastante para o ranking de segmentos perigosos.
KM_BIN_SIZE: int = 1

# --------------------------------------------------------------------------- #
# Agrupamento de categorias raras
# --------------------------------------------------------------------------- #
# Causas de acidente com frequência relativa abaixo do limiar viram RARE_LABEL.
# Aplicado apenas a `causa_acidente`; `municipio` é preservado (granularidade espacial).
RARE_THRESHOLD_PCT: float = 0.005
RARE_LABEL: str = "Outros"

# Coluna multivalorada a expandir e separador de seus valores.
TRACADO_COL: str = "tracado_via"
TRACADO_SEP: str = ";"
CAUSA_COL: str = "causa_acidente"
