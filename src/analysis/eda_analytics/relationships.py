"""§7 Relações entre as novas features e a gravidade.

Cruza a classe de gravidade com condições da via/ambiente/causa, mostrando a
distribuição percentual das classes dentro de cada categoria (barras 100% empilhadas).
Sem inferência causal — apenas associações observadas.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import polars as pl

from src.analysis.eda_analytics import config, data

logger = logging.getLogger(__name__)


def _crosstab_pct(df: pl.DataFrame, cond: str) -> pl.DataFrame:
    """% de cada classe de gravidade dentro de cada categoria da condição."""
    return (
        df.group_by([cond, "classe_gravidade"])
        .len()
        .with_columns(
            (pl.col("len") / pl.col("len").sum().over(cond) * 100).round(2).alias("pct")
        )
        .sort(cond)
    )


def _stacked(ct: pl.DataFrame, cond: str, top_n: int | None) -> None:
    """Barras 100% empilhadas das classes de gravidade por categoria."""
    # Ordena as categorias pela % da classe mais grave (Crítica) — destaca as piores.
    ordem_cat = (
        ct.filter(pl.col("classe_gravidade") == "Crítica")
        .sort("pct", descending=True)[cond]
        .to_list()
    )
    cats = ordem_cat or ct[cond].unique(maintain_order=True).to_list()
    if top_n:
        cats = cats[:top_n]
    pivot = ct.filter(pl.col(cond).is_in(cats)).pivot(
        values="pct", index=cond, on="classe_gravidade"
    )
    # Reordena o pivot conforme `cats`.
    pivot = pivot.with_columns(
        pl.col(cond).replace_strict({c: i for i, c in enumerate(cats)}, default=99).alias("ord")
    ).sort("ord")
    eixo = pivot[cond].to_list()

    fig, ax = plt.subplots(figsize=config.FIGSIZE)
    bottom = [0.0] * len(eixo)
    # Empilha da mais grave para a menos grave, com a paleta correspondente.
    for cls, cor in zip(config.CLASSE_ORDEM[::-1], config.CLASSE_CORES):
        if cls not in pivot.columns:
            continue
        vals = pivot[cls].fill_null(0).to_list()
        ax.bar(eixo, vals, bottom=bottom, label=cls, color=cor)
        bottom = [b + v for b, v in zip(bottom, vals)]

    ax.set_title(f"Classe de gravidade (%) por {cond}")
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
        logger.info("Classe de gravidade por %s:\n%s", cond, ct)
        # Causa tem alta cardinalidade: limita o gráfico às piores categorias.
        top_n = config.TOP_N if cond == "causa_acidente_agrupada" else None
        _stacked(ct, cond, top_n)
    return out
