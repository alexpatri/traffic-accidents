"""Configuração da análise final de priorização."""

from __future__ import annotations

from pathlib import Path

from src import config

PROJECT_ROOT: Path = config.PROJECT_ROOT
ACIDENTES_FILE: Path = PROJECT_ROOT / "data" / "analytics" / "acidentes_analytics.parquet"
TRECHOS_FILE: Path = PROJECT_ROOT / "data" / "analytics" / "trechos_analytics.parquet"
CLUSTERS_FILE: Path = PROJECT_ROOT / "outputs" / "clustering" / "trechos_clusters.parquet"

OUTPUT_DIR: Path = PROJECT_ROOT / "outputs" / "priorizacao"
FIG_DIR: Path = OUTPUT_DIR / "figures"
REPORT_FILE: Path = OUTPUT_DIR / "relatorio_priorizacao.md"
RANKING_FILE: Path = OUTPUT_DIR / "trechos_prioritarios.parquet"

# Parâmetros de ranking
TOP_BR: int = 12
TOP_HOTSPOTS: int = 15        # trechos por carga (índice total)
TOP_LETAIS: int = 15          # trechos por letalidade
MIN_VOL_LETAL: int = 5        # volume mínimo p/ ranking de letalidade (evita 100% de 1 acid.)
CLUSTER_LETAL: int = 2        # cluster de baixa freq. + alta letalidade (etapa de clusterização)

# Limiares para as regras de intervenção (perfil do trecho -> proposta)
LIM_NOTURNO_PCT: float = 45.0   # % de acidentes em período noturno -> iluminação
LIM_CURVA_PCT: float = 40.0     # % de acidentes em curva -> geometria/sinalização
LIM_URBANO_PCT: float = 60.0    # % em área urbana -> travessias/gestão urbana
LIM_SIMPLES_FATAL: float = 8.0  # pista simples + % fatal alto -> duplicação

FIGSIZE: tuple[int, int] = (10, 6)
DPI: int = 110
BAR_COLOR: str = "#b2182b"
