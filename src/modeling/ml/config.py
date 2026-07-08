"""Configuração da etapa de ML Supervisionado (classificação de risco de trechos).

Centraliza caminhos, definição do alvo, mapa UF→região, variantes de features, modelos e
parâmetros de validação. Espelha o padrão de `src/modeling/clustering/config.py`.
"""

from __future__ import annotations

from pathlib import Path

from src import config

# --------------------------------------------------------------------------- #
# Caminhos
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = config.PROJECT_ROOT
ACIDENTES_FILE: Path = PROJECT_ROOT / "data" / "analytics" / "acidentes_analytics.parquet"
TRECHOS_FILE: Path = PROJECT_ROOT / "data" / "analytics" / "trechos_analytics.parquet"

OUTPUT_DIR: Path = PROJECT_ROOT / "outputs" / "ml"
FIG_DIR: Path = OUTPUT_DIR / "figures"
REPORT_FILE: Path = OUTPUT_DIR / "relatorio_ml.md"
MODEL_FILE: Path = OUTPUT_DIR / "modelo_risco_trecho.joblib"
PRED_FILE: Path = OUTPUT_DIR / "trechos_risco_pred.parquet"

# --------------------------------------------------------------------------- #
# Reprodutibilidade e validação
# --------------------------------------------------------------------------- #
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.25
CV_FOLDS: int = 5
# Validação por grupo (rodovia): holdout e CV mantêm as BRs disjuntas, medindo a
# generalização a vias NÃO vistas (objetivo do modelo). Coluna de agrupamento:
GROUP_COL: str = "br"
# Tuning de hiperparâmetros (RandomizedSearchCV). 0 desliga (usa defaults).
TUNE_ITER: int = 15

# --------------------------------------------------------------------------- #
# Alvo — classe de risco do trecho
# --------------------------------------------------------------------------- #
# Quantidade contínua discretizada no alvo (severidade média por acidente do trecho).
TARGET_BASE: str = "indice_gravidade_medio"
# Filtro de ruído: índice médio é instável em trechos com pouquíssimos acidentes.
MIN_ACIDENTES: int = 3

# Denoising do alvo (empirical Bayes): encolhe o índice médio do trecho em direção à
# média global ponderada, proporcionalmente ao nº de acidentes. Testado (ablação): comprime
# o alvo e PIORA o F1 macro (classes coladas), com ganho ínfimo de QWK/AUC -> DESLIGADO.
# Mantido como opção reprodutível.
SHRINKAGE: bool = False
SHRINKAGE_K: float = 5.0
SCORE_COL: str = "risco_score"  # alvo efetivo que é discretizado (= índice cru se off)

CLASSE_LABELS: list[str] = ["Baixa", "Média", "Alta", "Crítica"]
# Classes consideradas "alto risco" na visão-resumo binária.
ALTO_RISCO_CLASSES: list[str] = ["Alta", "Crítica"]
# Piso de frequência: se a menor classe ficar abaixo disso, o alvo é degenerado.
CLASSE_RARA_MIN_PCT: float = 2.0

# Estratégia de corte do alvo:
#   "classe_gravidade" (estratégia A) -> cortes fixos do projeto aplicados ao índice médio
#       (Baixa 0–2 | Média >2–8 | Alta >8–20 | Crítica >20).
#   "quantil" (estratégia B) -> quartis do índice médio -> 4 faixas de prioridade ~25%.
# A estratégia A é degenerada nestes dados (Média ~77%, Crítica ~0,5%, abaixo do piso),
# então o padrão é "quantil" — decisão documentada no relatório.
CUT_STRATEGY: str = "quantil"
CLASSE_CORTES_A: list[float] = [2.0, 8.0, 20.0]  # estratégia A (referência)
QUANTIS: list[float] = [0.25, 0.50, 0.75]        # estratégia B (padrão)

# --------------------------------------------------------------------------- #
# Features estruturais (sem identificadores, sem desfecho)
# --------------------------------------------------------------------------- #
TRECHO_ID: str = "trecho"
TIPO_PISTA_COL: str = "tipo_pista"
USO_SOLO_COL: str = "uso_solo"
VEICULOS_COL: str = "veiculos"
# Flags de traçado agregadas como proporção de acidentes do trecho.
TRACADO_FLAGS: list[str] = [
    "tem_reta", "tem_curva", "tem_declive", "tem_aclive",
    "tem_intersecao_de_vias", "tem_ponte", "tem_tunel", "tem_rotatoria",
]

# Variantes de features a comparar.
VARIANTES: list[str] = ["fisico", "fisico_regiao", "fisico_uf"]
# Variante usada como MODELO DE ENTREGA (persistido). `fisico_uf` é avaliada para
# comparação, mas não é entregue por depender de localização (não transfere a vias novas);
# `fisico_regiao` mantém o sinal regional de forma transferível.
VARIANTE_ENTREGA: str = "fisico_regiao"

# Colunas PROIBIDAS na matriz de formação (desfecho + identificadores).
LEAKAGE_COLS: list[str] = [
    "indice_gravidade", "indice_gravidade_medio", "indice_gravidade_maximo",
    "indice_gravidade_total", "mortos", "feridos_graves", "feridos_leves",
    "feridos", "ilesos", "ignorados", "pessoas", "fatal", "classe_gravidade",
    "qtd_acidentes", "qtd_acidentes_fatais", "pct_acidentes_fatais",
    "br", "km_faixa", "km", "trecho",
]

# --------------------------------------------------------------------------- #
# Mapa UF -> região (5 macrorregiões) — proxy regional transferível
# --------------------------------------------------------------------------- #
UF_REGIAO: dict[str, str] = {
    "AC": "Norte", "AP": "Norte", "AM": "Norte", "PA": "Norte",
    "RO": "Norte", "RR": "Norte", "TO": "Norte",
    "AL": "Nordeste", "BA": "Nordeste", "CE": "Nordeste", "MA": "Nordeste",
    "PB": "Nordeste", "PE": "Nordeste", "PI": "Nordeste", "RN": "Nordeste",
    "SE": "Nordeste",
    "DF": "Centro-Oeste", "GO": "Centro-Oeste", "MT": "Centro-Oeste",
    "MS": "Centro-Oeste",
    "ES": "Sudeste", "MG": "Sudeste", "RJ": "Sudeste", "SP": "Sudeste",
    "PR": "Sul", "RS": "Sul", "SC": "Sul",
}

# --------------------------------------------------------------------------- #
# Modelos (dispatcher em model.py)
# --------------------------------------------------------------------------- #
ALGOS: list[str] = ["logreg", "random_forest", "hist_gboost"]

# Grades para RandomizedSearchCV (chaves com prefixo do passo `clf` do Pipeline).
PARAM_GRIDS: dict[str, dict] = {
    "logreg": {
        "clf__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
    },
    "random_forest": {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [None, 6, 10, 16, 24],
        "clf__min_samples_leaf": [1, 3, 5, 10, 20],
        "clf__max_features": ["sqrt", 0.5, 1.0],
    },
    "hist_gboost": {
        "clf__learning_rate": [0.03, 0.05, 0.1, 0.2],
        "clf__max_leaf_nodes": [15, 31, 63],
        "clf__max_iter": [200, 400],
        "clf__min_samples_leaf": [20, 50, 100],
    },
}

# --------------------------------------------------------------------------- #
# Plotagem (idêntica às demais etapas)
# --------------------------------------------------------------------------- #
FIGSIZE: tuple[int, int] = (10, 6)
DPI: int = 110
BAR_COLOR: str = "#3b6ea5"
CLASSE_CORES: list[str] = ["#cccccc", "#67a9cf", "#ef8a62", "#b2182b"]  # Baixa→Crítica
