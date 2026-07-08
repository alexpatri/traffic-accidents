"""Clusterização 1 — perfis de risco de trechos rodoviários.

Pipeline: pré-processa (4 features de frequência/severidade) -> avalia K (cotovelo +
silhouette) -> ajusta KMeans no K decidido -> perfila os grupos (médias, medianas e
desfecho retido) -> nomeia cada perfil -> persiste os rótulos.

Não é um ranking: o objetivo é descobrir *perfis* (ex.: alto-volume/baixa-letalidade
vs baixo-volume/alta-letalidade vs hotspot).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import polars as pl

from src.modeling.clustering import cluster, config, data, elbow, preprocessing, visualization

logger = logging.getLogger(__name__)


def _niveis(medias: list[float], populacao: pl.Series) -> list[str]:
    """Classifica cada média de cluster em Baixo/Médio/Alto.

    Os cortes vêm da distribuição POPULACIONAL (p50/p90 de todos os trechos), não dos
    poucos valores de cluster — assim a letalidade extrema (muito assimétrica) não fica
    no mesmo balde que uma letalidade apenas moderada.
    """
    lo, hi = np.quantile(populacao.to_numpy(), [0.5, 0.9])
    return [
        "Baixo" if v <= lo else "Alto" if v >= hi else "Médio" for v in medias
    ]


def _perfil(df_lab: pl.DataFrame) -> pl.DataFrame:
    """Tabela de perfil por cluster: tamanho, médias/medianas e desfecho retido."""
    total = df_lab.height
    return (
        df_lab.group_by("cluster")
        .agg(
            pl.len().alias("n"),
            pl.col("qtd_acidentes").mean().round(2).alias("qtd_media"),
            pl.col("qtd_acidentes").median().alias("qtd_mediana"),
            pl.col("indice_gravidade_total").mean().round(2).alias("ig_total_media"),
            pl.col("indice_gravidade_medio").mean().round(2).alias("ig_medio_media"),
            pl.col("pct_acidentes_fatais").mean().round(2).alias("pct_fatais_media"),
            pl.col("mortos").sum().alias("mortos_total"),
            pl.col("feridos_graves").sum().alias("feridos_graves_total"),
            pl.col("indice_gravidade_maximo").max().alias("ig_maximo"),
        )
        .with_columns((pl.col("n") / total * 100).round(2).alias("pct"))
        .sort("cluster")
    )


def _nomear(perfil: pl.DataFrame, df: pl.DataFrame) -> list[dict[str, Any]]:
    """Gera nome e descrição textual de cada cluster a partir do perfil."""
    volume = _niveis(perfil.get_column("qtd_media").to_list(), df.get_column("qtd_acidentes"))
    letal = _niveis(perfil.get_column("pct_fatais_media").to_list(), df.get_column("pct_acidentes_fatais"))
    risco = _niveis(perfil.get_column("ig_medio_media").to_list(), df.get_column("indice_gravidade_medio"))

    perfis: list[dict[str, Any]] = []
    for i, row in enumerate(perfil.iter_rows(named=True)):
        nome = f"{volume[i]} volume · {letal[i]} letalidade"
        descricao = (
            f"{row['n']} trechos ({row['pct']:.1f}%); "
            f"média {row['qtd_media']:.1f} acidentes/trecho, "
            f"índice médio {row['ig_medio_media']:.1f} ({risco[i].lower()} risco por evento), "
            f"{row['pct_fatais_media']:.1f}% de acidentes fatais; "
            f"{row['mortos_total']} mortos e {row['feridos_graves_total']} feridos graves no total."
        )
        perfis.append(
            {"cluster": row["cluster"], "nome": nome, "descricao": descricao}
        )
    return perfis


def run() -> dict[str, Any]:
    """Executa a clusterização de trechos e retorna os artefatos para o relatório."""
    logger.info("--- Clusterização 1: perfis de trechos rodoviários ---")
    df = data.load_trechos()

    # 1) Pré-processamento (4 features: frequência + severidade + risco + letalidade)
    cm = preprocessing.build_matrix_trechos(df)

    # 2) Seleção de K (cotovelo + silhouette)
    selection = elbow.evaluate_k(
        cm.X, silhouette_sample_size=config.SILHOUETTE_SAMPLE_SIZE
    )
    k = config.TRECHOS_K
    logger.info(
        "K escolhido (trechos) = %d | melhor silhouette em K=%d",
        k, elbow.best_k_by_silhouette(selection),
    )

    # 3) Ajuste final do KMeans
    labels, _ = cluster.fit_cluster(cm.X, algo="kmeans", k=k)
    df_lab = df.with_columns(pl.Series("cluster", labels))

    # 4) Perfilamento e nomeação
    perfil = _perfil(df_lab)
    perfis = _nomear(perfil, df)
    for p in perfis:
        logger.info("Cluster %d (trechos) — %s", p["cluster"], p["nome"])

    # 5) Run de sensibilidade (3 features, sem indice_gravidade_total redundante)
    cm3 = preprocessing.build_matrix_trechos(
        df, features=config.TRECHOS_FEATURES_SENSIBILIDADE
    )
    sel3 = elbow.evaluate_k(cm3.X, silhouette_sample_size=config.SILHOUETTE_SAMPLE_SIZE)
    sil3 = sel3.silhouettes[sel3.ks.index(k)] if k in sel3.ks else float("nan")
    sil4 = selection.silhouettes[selection.ks.index(k)] if k in selection.ks else float("nan")
    logger.info("Sensibilidade trechos @K=%d: silhouette 4-feat=%.4f vs 3-feat=%.4f",
                k, sil4, sil3)

    # 6) Visualizações
    visualization.plot_elbow(
        selection.ks, selection.inertias, "trechos_elbow",
        "Método do Cotovelo — Trechos",
    )
    visualization.plot_silhouette(
        selection.ks, selection.silhouettes, "trechos_silhouette",
        "Silhouette Score — Trechos",
    )
    pca_var = visualization.plot_pca(
        cm.X, labels, "trechos_pca", "PCA 2D — Clusters de Trechos",
    )
    for col in ("qtd_acidentes", "indice_gravidade_medio", "pct_acidentes_fatais"):
        visualization.plot_box_por_cluster(
            df_lab, col, "cluster", f"trechos_box_{col}",
            f"{col} por cluster (Trechos)",
        )

    # 7) Persistência dos rótulos
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (
        df_lab.select(
            config.TRECHOS_ID, "uf", "br", "km_faixa", "cluster",
            *config.TRECHOS_FEATURES,
        ).write_parquet(config.TRECHOS_CLUSTERS_FILE)
    )
    logger.info("Rótulos de trechos salvos: %s",
                config.TRECHOS_CLUSTERS_FILE.relative_to(config.PROJECT_ROOT))

    return {
        "nome": "Trechos",
        "n": df.height,
        "features": config.TRECHOS_FEATURES,
        "k": k,
        "selection": selection,
        "pca_var": pca_var,
        "perfil": perfil,
        "perfis": perfis,
        "sensibilidade": {"k": k, "silhouette_4feat": sil4, "silhouette_3feat": sil3},
    }
