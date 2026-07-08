"""Engenharia de features e alvo da classificação de risco de trechos.

Constrói uma tabela com **uma linha por trecho**: features estruturais agregadas do
nível-acidente (sem identificadores, sem desfecho) + o alvo ordinal `classe_risco`
derivado de `indice_gravidade_medio`. Aplica o filtro de volume mínimo e a guarda
anti-vazamento.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

from src.modeling.ml import config

logger = logging.getLogger(__name__)


def _prop_name(flag: str) -> str:
    """`tem_curva` -> `prop_curva` (proporção de acidentes do trecho com a flag)."""
    return f"prop_{flag.removeprefix('tem_')}"


def compute_cortes(table: pl.DataFrame) -> list[float]:
    """Determina os 3 limites do alvo (sobre o score efetivo) conforme a estratégia."""
    if config.CUT_STRATEGY == "classe_gravidade":
        return list(config.CLASSE_CORTES_A)
    if config.CUT_STRATEGY == "quantil":
        col = table.get_column(config.SCORE_COL)
        return [round(float(col.quantile(q)), 3) for q in config.QUANTIS]
    raise ValueError(f"CUT_STRATEGY desconhecida: {config.CUT_STRATEGY!r}")


def _expr_classe(cortes: list[float]) -> pl.Expr:
    """Discretiza o score efetivo (`SCORE_COL`) em `classe_risco` dados os limites."""
    labels = config.CLASSE_LABELS
    expr = pl.when(pl.col(config.SCORE_COL) <= cortes[0]).then(pl.lit(labels[0]))
    for i in range(1, len(cortes)):
        expr = expr.when(pl.col(config.SCORE_COL) <= cortes[i]).then(pl.lit(labels[i]))
    return expr.otherwise(pl.lit(labels[-1])).alias("classe_risco")


def _agregar_estrutural(df_acidentes: pl.DataFrame) -> pl.DataFrame:
    """Agrega atributos estruturais do trecho a partir dos acidentes (proporções/modas)."""
    flag_exprs = [
        pl.col(f).mean().round(4).alias(_prop_name(f)) for f in config.TRACADO_FLAGS
    ]
    return df_acidentes.group_by(config.TRECHO_ID).agg(
        pl.col("uf").first().alias("uf"),
        pl.col("br").first().alias("br"),  # só para agrupar na validação (não é feature)
        pl.col(config.TIPO_PISTA_COL).mode().first().alias("tipo_pista_pred"),
        (pl.col(config.USO_SOLO_COL) == "Sim").mean().round(4).alias("pct_urbano"),
        pl.col(config.VEICULOS_COL).mean().round(3).alias("veiculos_medio"),
        *flag_exprs,
    )


def build_feature_table(
    df_acidentes: pl.DataFrame, df_trechos: pl.DataFrame
) -> pl.DataFrame:
    """Monta a tabela trecho × (features estruturais + alvo), filtrada por volume.

    Returns:
        DataFrame com `trecho`, `uf`, `regiao`, features estruturais, `qtd_acidentes`,
        `indice_gravidade_medio` (base do alvo) e `classe_risco` (alvo).
    """
    estrut = _agregar_estrutural(df_acidentes)
    base = df_trechos.select(config.TRECHO_ID, config.TARGET_BASE, "qtd_acidentes")

    filtrada = estrut.join(base, on=config.TRECHO_ID, how="inner").filter(
        pl.col("qtd_acidentes") >= config.MIN_ACIDENTES
    )

    # Score efetivo do alvo: encolhimento empirical-Bayes (denoising) ou o índice cru.
    if config.SHRINKAGE:
        gmean = float(
            (filtrada.get_column(config.TARGET_BASE) * filtrada.get_column("qtd_acidentes")).sum()
            / filtrada.get_column("qtd_acidentes").sum()
        )
        k = config.SHRINKAGE_K
        score_expr = (
            (pl.col("qtd_acidentes") * pl.col(config.TARGET_BASE) + k * gmean)
            / (pl.col("qtd_acidentes") + k)
        ).round(4).alias(config.SCORE_COL)
    else:
        score_expr = pl.col(config.TARGET_BASE).alias(config.SCORE_COL)

    filtrada = filtrada.with_columns(score_expr)
    cortes = compute_cortes(filtrada)  # sobre o score efetivo, nos trechos já filtrados
    table = filtrada.with_columns(
        pl.col("uf").replace(config.UF_REGIAO).alias("regiao"),
        _expr_classe(cortes),
    )
    logger.info(
        "Tabela de features: %d trechos (filtro qtd_acidentes >= %d) de %d totais | "
        "estratégia=%s | shrinkage=%s (K=%.1f) | cortes=%s",
        table.height, config.MIN_ACIDENTES, df_trechos.height,
        config.CUT_STRATEGY, config.SHRINKAGE, config.SHRINKAGE_K, cortes,
    )
    return table


def class_distribution(table: pl.DataFrame) -> pl.DataFrame:
    """Distribuição absoluta e relativa das classes do alvo (ordenada Baixa→Crítica)."""
    total = table.height
    dist = table.group_by("classe_risco").agg(pl.len().alias("n"))
    ordem = {lab: i for i, lab in enumerate(config.CLASSE_LABELS)}
    return (
        dist.with_columns(
            (pl.col("n") / total * 100).round(2).alias("pct"),
            pl.col("classe_risco").replace_strict(ordem, return_dtype=pl.Int8).alias("_o"),
        )
        .sort("_o")
        .drop("_o")
    )


def leakage_guard(feature_names: list[str]) -> None:
    """Falha se qualquer feature for de desfecho ou identificador."""
    proibidas = sorted(set(feature_names) & set(config.LEAKAGE_COLS))
    if proibidas:
        raise ValueError(
            f"Vazamento/identificador na matriz de features: {proibidas}. "
            "Use apenas atributos estruturais do trecho."
        )


def select_features(
    table: pl.DataFrame, variante: str
) -> tuple[pl.DataFrame, list[str]]:
    """Seleciona e codifica (one-hot) as features de uma variante.

    Args:
        table: saída de `build_feature_table`.
        variante: "fisico" | "fisico_regiao" | "fisico_uf".

    Returns:
        (X codificada, nomes das features). X tem só colunas numéricas (0/1 e proporções).
    """
    num_cols = ["pct_urbano", "veiculos_medio"] + [
        _prop_name(f) for f in config.TRACADO_FLAGS
    ]
    cat_cols = ["tipo_pista_pred"]
    if variante == "fisico_regiao":
        cat_cols.append("regiao")
    elif variante == "fisico_uf":
        cat_cols.append("uf")
    elif variante != "fisico":
        raise ValueError(f"Variante desconhecida: {variante!r}")

    leakage_guard(num_cols + cat_cols)  # checa antes do one-hot
    X = table.select(num_cols + cat_cols).to_dummies(columns=cat_cols)
    return X, X.columns


def encode_target(table: pl.DataFrame) -> np.ndarray:
    """Codifica `classe_risco` em inteiros ordinais 0..3 (Baixa..Crítica)."""
    ordem = {lab: i for i, lab in enumerate(config.CLASSE_LABELS)}
    return (
        table.get_column("classe_risco")
        .replace_strict(ordem, return_dtype=pl.Int8)
        .to_numpy()
    )
