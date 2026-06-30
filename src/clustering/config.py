"""Configuração da etapa de Clusterização: caminhos, features, faixa de K e plot.

Centraliza todas as decisões de modelagem para que `preprocessing`, `elbow`,
`trechos`, `acidentes`, `visualization` e `report` permaneçam sem números mágicos.
Espelha o padrão de `src/eda_analytics/config.py`.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Caminhos
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
ANALYTICS_FILE: Path = PROJECT_ROOT / "data" / "analytics" / "acidentes_analytics.parquet"
TRECHOS_FILE: Path = PROJECT_ROOT / "data" / "analytics" / "trechos_analytics.parquet"

OUTPUT_DIR: Path = PROJECT_ROOT / "outputs" / "clustering"
FIG_DIR: Path = OUTPUT_DIR / "figures"
REPORT_FILE: Path = OUTPUT_DIR / "relatorio_clustering.md"
TRECHOS_CLUSTERS_FILE: Path = OUTPUT_DIR / "trechos_clusters.parquet"
ACIDENTES_CLUSTERS_FILE: Path = OUTPUT_DIR / "acidentes_clusters.parquet"

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

# --------------------------------------------------------------------------- #
# Clusterização 2 — Acidentes
# --------------------------------------------------------------------------- #
ACIDENTES_ID: str = "id"

# Vazamento de desfecho: PROIBIDAS na matriz de formação (assert garante isso).
ACIDENTES_LEAKAGE: list[str] = [
    "indice_gravidade",
    "fatal",
    "classe_gravidade",
    "periodo_noturno",
    "mortos",
    "feridos_graves",
    "feridos_leves",
    "feridos",
    "ilesos",
    "pessoas",
]

# Numéricas de formação.
ACIDENTES_NUM_LOG: list[str] = ["veiculos"]          # log1p + MinMax
ACIDENTES_CICLICA: str = "hora"                        # sin/cos (respeita 23h≈0h)
ACIDENTES_BOOL: list[str] = ["fim_de_semana"]         # 0/1 cru

# Flags de traçado mantidas (prevalência ≥ ~3%); as raras viram ruído.
ACIDENTES_TRACADO_FLAGS: list[str] = [
    "tem_reta",
    "tem_curva",
    "tem_declive",
    "tem_aclive",
    "tem_intersecao_de_vias",
]

# Categóricas one-hot (com colapso de níveis raros).
ACIDENTES_TIPO_PISTA: str = "tipo_pista"              # 3 níveis
ACIDENTES_USO_SOLO: str = "uso_solo"                  # Sim/Não -> 1 booleano
ACIDENTES_METEO: str = "condicao_meteorologica"       # colapsada p/ MeteoMantidas + Outros
ACIDENTES_METEO_MANTIDAS: list[str] = ["Céu Claro", "Nublado", "Chuva"]
ACIDENTES_CAUSA: str = "causa_acidente_agrupada"      # top-N + Outros
ACIDENTES_CAUSA_TOP_N: int = 8

# Descritores (rótulos legíveis) e desfecho — só para caracterização pós-cluster.
ACIDENTES_DESCRITORES_CAT: list[str] = ["turno", "fase_dia", "tipo_pista", "uso_solo"]
ACIDENTES_DESCRITORES_OUT: list[str] = [
    "indice_gravidade",
    "fatal",
    "classe_gravidade",
    "mortos",
    "feridos_graves",
]
# K candidato final (decisão documentada no relatório).
ACIDENTES_K: int = 5

# Bloco -> peso aplicado (1/√n_cols do bloco) é calculado no preprocessing.
# Aqui só nomeamos os blocos conceituais para o relatório.
BLOCOS_ACIDENTES: list[str] = [
    "temporal", "via", "meteorologia", "causa", "tracado",
]

# --------------------------------------------------------------------------- #
# Ordens de domínio (eixos coerentes)
# --------------------------------------------------------------------------- #
CLASSE_ORDEM: list[str] = ["Baixa", "Média", "Alta", "Crítica"]
TURNO_ORDEM: list[str] = ["Madrugada", "Manhã", "Tarde", "Noite"]

# --------------------------------------------------------------------------- #
# Parâmetros de plotagem (idênticos à EDA)
# --------------------------------------------------------------------------- #
FIGSIZE: tuple[int, int] = (10, 6)
DPI: int = 110
BAR_COLOR: str = "#3b6ea5"
# Paleta qualitativa para colorir clusters (até 10 grupos).
CLUSTER_CMAP: str = "tab10"
