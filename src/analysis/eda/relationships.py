"""§7 Relações entre variáveis: gravidade × condições.

Cruza `classificacao_acidente` (gravidade) com condições da via/ambiente, mostrando a
distribuição percentual da gravidade dentro de cada categoria. Sem inferência causal.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import polars as pl

from src.analysis.eda import config, data

logger = logging.getLogger(__name__)

# Ordem fixa das classes (da mais grave para a menos grave) para cores consistentes.
_CLASSES = ["Com Vítimas Fatais", "Com Vítimas Feridas", "Sem Vítimas", "Não informado"]
_CORES = ["#b2182b", "#ef8a62", "#67a9cf", "#cccccc"]


def _crosstab_pct(df: pl.DataFrame, cond: str) -> pl.DataFrame:
    """% de cada classe de gravidade dentro de cada categoria da condição."""
    return (
        df.group_by([cond, config.TARGET_COL])
        .len()
        .with_columns(
            (pl.col("len") / pl.col("len").sum().over(cond) * 100).round(2).alias("pct")
        )
        .sort(cond)
    )


def _stacked(ct: pl.DataFrame, cond: str) -> None:
    """Barras empilhadas (100%) da gravidade por categoria da condição."""
    cats = ct[cond].unique(maintain_order=True).to_list()
    pivot = ct.pivot(values="pct", index=cond, on=config.TARGET_COL)

    fig, ax = plt.subplots(figsize=config.FIGSIZE)
    bottom = [0.0] * len(cats)
    for cls, cor in zip(_CLASSES, _CORES):
        if cls not in pivot.columns:
            continue
        vals = pivot[cls].fill_null(0).to_list()
        ax.bar(cats, vals, bottom=bottom, label=cls, color=cor)
        bottom = [b + v for b, v in zip(bottom, vals)]

    ax.set_title(f"Gravidade (%) por {cond}")
    ax.set_ylabel("% dentro da categoria")
    ax.legend(fontsize=8, loc="lower right")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    data.save_fig(fig, f"07_gravidade_por_{cond}")


def relationships(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Gera as tabelas cruzadas e os gráficos de gravidade por condição."""
    out: dict[str, pl.DataFrame] = {}
    for cond in config.RELATION_COLS:
        ct = _crosstab_pct(df, cond)
        out[cond] = ct
        logger.info("Gravidade por %s:\n%s", cond, ct)
        _stacked(ct, cond)
    return out
