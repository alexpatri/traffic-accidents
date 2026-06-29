"""Configuração da EDA Analytics: caminhos, grupos de colunas e parâmetros de plot.

Espelha `src/eda/config.py`, mas aponta para a camada Analytics e declara os grupos
de features criadas na etapa de Feature Engineering.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Caminhos
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
ANALYTICS_FILE: Path = PROJECT_ROOT / "data" / "analytics" / "acidentes_analytics.parquet"
TRECHOS_FILE: Path = PROJECT_ROOT / "data" / "analytics" / "trechos_analytics.parquet"
FIG_DIR: Path = PROJECT_ROOT / "outputs" / "eda_analytics" / "figures"

# --------------------------------------------------------------------------- #
# Features criadas na Feature Engineering (colunas novas vs. Trusted)
# --------------------------------------------------------------------------- #
FEATURES_TEMPORAIS: list[str] = [
    "hora", "mes", "trimestre", "dia_da_semana", "fim_de_semana", "turno",
]
FEATURES_GRAVIDADE: list[str] = [
    "indice_gravidade", "fatal", "periodo_noturno", "classe_gravidade",
]
FEATURES_ESPACIAIS: list[str] = ["km_faixa", "trecho"]
TRACADO_FLAGS: list[str] = [
    "tem_reta", "tem_curva", "tem_declive", "tem_aclive",
    "tem_intersecao_de_vias", "tem_rotatoria", "tem_retorno_regulamentado",
    "tem_em_obras", "tem_viaduto", "tem_ponte", "tem_desvio_temporario", "tem_tunel",
]
FEATURES_CATEGORICAS: list[str] = ["causa_acidente_agrupada"]

# Conjunto completo das 25 features criadas (para a §1 visão geral).
NOVAS_FEATURES: list[str] = (
    FEATURES_TEMPORAIS
    + FEATURES_GRAVIDADE
    + FEATURES_ESPACIAIS
    + TRACADO_FLAGS
    + FEATURES_CATEGORICAS
)

# --------------------------------------------------------------------------- #
# Domínios ordenados (para eixos coerentes)
# --------------------------------------------------------------------------- #
CLASSE_ORDEM: list[str] = ["Baixa", "Média", "Alta", "Crítica"]
TURNO_ORDEM: list[str] = ["Madrugada", "Manhã", "Tarde", "Noite"]

# --------------------------------------------------------------------------- #
# Numéricas para correlação (exclui derivadas redundantes e identificadores)
# --------------------------------------------------------------------------- #
CORR_COLS: list[str] = [
    "indice_gravidade", "mortos", "feridos_graves", "feridos_leves",
    "feridos", "veiculos", "pessoas", "km",
]

# Condições categóricas cruzadas com a gravidade na análise de relações (§7).
RELATION_COLS: list[str] = [
    "turno",
    "tipo_pista",
    "condicao_meteorologica",
    "uso_solo",
    "causa_acidente_agrupada",
]

# Métricas do dataset por trecho (§9).
TRECHO_METRICAS: list[str] = [
    "qtd_acidentes", "indice_gravidade_total",
    "indice_gravidade_medio", "pct_acidentes_fatais",
]

# --------------------------------------------------------------------------- #
# Parâmetros de plotagem (idênticos à 1ª EDA)
# --------------------------------------------------------------------------- #
TOP_N: int = 20          # rankings de trechos pedem top-20 (vs. top-15 da 1ª EDA)
FIGSIZE: tuple[int, int] = (10, 6)
DPI: int = 110
BAR_COLOR: str = "#3b6ea5"
# Paleta da gravidade (mais grave -> menos grave), reuso da §7 da 1ª EDA.
CLASSE_CORES: list[str] = ["#b2182b", "#ef8a62", "#67a9cf", "#cccccc"]
