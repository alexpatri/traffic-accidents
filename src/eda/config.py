"""Configuração da EDA: caminhos, grupos de colunas e parâmetros de plotagem."""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Caminhos
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
TRUSTED_FILE: Path = PROJECT_ROOT / "data" / "trusted" / "acidentes_trusted.parquet"
FIG_DIR: Path = PROJECT_ROOT / "outputs" / "eda" / "figures"

# --------------------------------------------------------------------------- #
# Grupos de colunas (nomes da camada Trusted)
# --------------------------------------------------------------------------- #
# Categóricas de baixa/média cardinalidade — plotadas integralmente.
LOW_CARD_CATS: list[str] = [
    "classificacao_acidente",
    "fase_dia",
    "tipo_pista",
    "uso_solo",
    "sentido_via",
    "dia_semana",
    "condicao_meteorologica",
    "tipo_acidente",
]

# Categóricas de alta cardinalidade — plotadas via top-N.
HIGH_CARD_CATS: list[str] = ["causa_acidente", "br", "municipio", "tracado_via"]

# Numéricas para estatísticas descritivas e distribuições.
NUMERIC_COLS: list[str] = [
    "km",
    "pessoas",
    "mortos",
    "feridos_leves",
    "feridos_graves",
    "ilesos",
    "ignorados",
    "feridos",
    "veiculos",
]

# Numéricas para correlação (exclui lat/long: coordenadas não são grandezas comparáveis).
CORR_COLS: list[str] = NUMERIC_COLS

# Coluna alvo (gravidade) usada nas análises bivariadas.
TARGET_COL: str = "classificacao_acidente"

# Condições cruzadas com a gravidade na análise de relações.
RELATION_COLS: list[str] = [
    "fase_dia",
    "condicao_meteorologica",
    "tipo_pista",
    "sentido_via",
]

# --------------------------------------------------------------------------- #
# Parâmetros de plotagem
# --------------------------------------------------------------------------- #
TOP_N: int = 15
FIGSIZE: tuple[int, int] = (10, 6)
DPI: int = 110
BAR_COLOR: str = "#3b6ea5"
