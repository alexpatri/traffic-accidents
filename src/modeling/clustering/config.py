"""Configuração da etapa de Clusterização: caminhos, features, faixa de K e plot.

Centraliza todas as decisões de modelagem para que `preprocessing`, `elbow`,
`trechos`, `visualization` e `report` permaneçam sem números mágicos.
Espelha o padrão de `src/analysis/eda_analytics/config.py`.
"""

from __future__ import annotations

from pathlib import Path

from src import config

# --------------------------------------------------------------------------- #
# Caminhos
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = config.PROJECT_ROOT
TRECHOS_FILE: Path = PROJECT_ROOT / "data" / "analytics" / "trechos_analytics.parquet"

OUTPUT_DIR: Path = PROJECT_ROOT / "outputs" / "clustering"
FIG_DIR: Path = OUTPUT_DIR / "figures"
REPORT_FILE: Path = OUTPUT_DIR / "relatorio_clustering.md"
TRECHOS_CLUSTERS_FILE: Path = OUTPUT_DIR / "trechos_clusters.parquet"

# --------------------------------------------------------------------------- #
# Reprodutibilidade e seleção de K
# --------------------------------------------------------------------------- #
RANDOM_STATE: int = 42
KMEANS_N_INIT: int = 10
K_RANGE: range = range(2, 11)  # K = 2..10

# Silhouette é O(n²) em memória/tempo (matriz de distâncias n×n). Mesmo 33k trechos
# exigiriam ~8 GB; por isso é sempre avaliado numa amostra fixa (mesmos índices em
# todos os K, para a curva ser comparável).
SILHOUETTE_SAMPLE_SIZE: int = 10_000

# --------------------------------------------------------------------------- #
# Clusterização 1 — Trechos
# --------------------------------------------------------------------------- #
# Features de FORMAÇÃO dos clusters (frequência + severidade + risco + letalidade).
TRECHOS_FEATURES: list[str] = [
    "qtd_acidentes",
    "indice_gravidade_total",
    "indice_gravidade_medio",
    "pct_acidentes_fatais",
]
# Colunas com cauda longa que recebem log1p antes do StandardScaler.
TRECHOS_LOG_FEATURES: list[str] = ["qtd_acidentes", "indice_gravidade_total"]
# Identificador (removido da matriz; mantido para rastrear cada linha).
TRECHOS_ID: str = "trecho"
# Run de sensibilidade: remove a feature redundante (qtd↔total, r≈0.89).
TRECHOS_FEATURES_SENSIBILIDADE: list[str] = [
    "qtd_acidentes",
    "indice_gravidade_medio",
    "pct_acidentes_fatais",
]
# Desfecho retido só para caracterizar grupos depois (NÃO entra na formação).
TRECHOS_DESCRITORES: list[str] = [
    "mortos",
    "feridos_graves",
    "feridos_leves",
    "indice_gravidade_maximo",
]
# K candidato final (decisão documentada no relatório; sobrescreve a busca).
TRECHOS_K: int = 4

# Obs.: a clusterização de acidentes foi investigada e NÃO incluída como entrega
# (KMeans não formou perfis multivariados nítidos — apenas redescobre `tipo_pista`).
# Conclusão documentada no relatório e no README.

# --------------------------------------------------------------------------- #
# Parâmetros de plotagem (idênticos à EDA)
# --------------------------------------------------------------------------- #
FIGSIZE: tuple[int, int] = (10, 6)
DPI: int = 110
BAR_COLOR: str = "#3b6ea5"
# Paleta qualitativa para colorir clusters (até 10 grupos).
CLUSTER_CMAP: str = "tab10"
