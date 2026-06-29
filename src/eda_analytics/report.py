"""§10 + §11 Relatório consolidado (apenas logado — não grava artefatos).

Loga, no console, as estatísticas-chave que sustentam o texto do README:
preparação para Clusterização (§10) e para Machine Learning / Data Leakage (§11).
Não produz figuras nem arquivos; serve de checklist quantitativo para a redação.
"""

from __future__ import annotations

import logging

import polars as pl

from src.eda_analytics import config, data

logger = logging.getLogger(__name__)

# Variáveis derivadas do desfecho do acidente — só podem ser ALVO (risco de leakage).
ALVOS_LEAKAGE = [
    "indice_gravidade", "classe_gravidade", "fatal",
    "mortos", "feridos_graves", "feridos_leves", "feridos",
]
# Candidatas a preditoras (conhecidas antes/independentes do desfecho).
PREDITORAS = (
    config.FEATURES_TEMPORAIS
    + ["uf", "br", "trecho", "km_faixa"]
    + config.TRACADO_FLAGS
    + ["causa_acidente_agrupada", "tipo_pista", "condicao_meteorologica",
       "uso_solo", "veiculos", "pessoas"]
)


def report(df: pl.DataFrame) -> dict:
    """Consolida números para clusterização e modelagem."""
    logger.info("===== §10 Preparação para Clusterização =====")

    # Correlação entre o índice e seus componentes (evidencia redundância).
    corr = df.select(config.CORR_COLS).corr()
    nomes = corr.columns
    pares = []
    mat = corr.to_numpy()
    for i in range(len(nomes)):
        for j in range(i + 1, len(nomes)):
            pares.append((nomes[i], nomes[j], round(float(mat[i][j]), 2)))
    altos = sorted([p for p in pares if abs(p[2]) >= 0.6], key=lambda x: -abs(x[2]))
    logger.info("Pares numéricos com |r|>=0.6 (redundância para clusterização): %s", altos)
    logger.info(
        "Variáveis candidatas à clusterização de trechos: "
        "qtd_acidentes, indice_gravidade_total, indice_gravidade_medio, "
        "pct_acidentes_fatais, mortos, feridos_graves (evitar índice + componentes juntos)."
    )

    logger.info("===== §11 Preparação para Machine Learning =====")
    logger.info("Variáveis ALVO / risco de Data Leakage (NÃO usar como preditoras): %s",
                ALVOS_LEAKAGE)
    logger.info("Candidatas a PREDITORAS: %s", PREDITORAS)

    # Desbalanceamento do alvo (classe e fatal).
    classe = (
        df["classe_gravidade"].value_counts()
        .with_columns((pl.col("count") / df.height * 100).round(2).alias("pct"))
        .sort("count", descending=True)
    )
    fatal_pct = round(df["fatal"].mean() * 100, 2)
    logger.info("Desbalanceamento classe_gravidade:\n%s", classe)
    logger.info("Desbalanceamento alvo binário fatal: %.2f%% positivos", fatal_pct)

    return {"pares_altos": altos, "classe": classe, "fatal_pct": fatal_pct}
