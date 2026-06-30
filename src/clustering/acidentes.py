"""Clusterização 2 — perfis de ocorrência de acidentes.

Pipeline: pré-processa (apenas variáveis conhecidas no momento do acidente, sem
vazamento de desfecho) -> avalia K -> ajusta KMeans -> perfila os grupos com
descritores legíveis e com o desfecho RETIDO (índice de gravidade, % fatal,
classe) usado só para caracterizar — não para formar — os clusters.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import polars as pl

from src.clustering import cluster, config, data, elbow, preprocessing, visualization

logger = logging.getLogger(__name__)


def _niveis(valores: list[float]) -> list[str]:
    """Classifica cada valor em Baixo/Médio/Alto pelos terços da distribuição."""
    arr = np.asarray(valores, dtype=float)
    lo, hi = np.quantile(arr, [1 / 3, 2 / 3])
    return ["Baixo" if v <= lo else "Alto" if v >= hi else "Médio" for v in arr]


def _perfil(df_lab: pl.DataFrame) -> pl.DataFrame:
    """Perfil por cluster: contexto (modas), hora circular e desfecho retido."""
    total = df_lab.height
    base = (
        df_lab.group_by("cluster")
        .agg(
            pl.len().alias("n"),
            pl.col("veiculos").mean().round(2).alias("veiculos_media"),
            (pl.col("fim_de_semana").mean() * 100).round(1).alias("pct_fds"),
            # componentes da hora circular (média vetorial)
            ((pl.col("hora").cast(pl.Float64) / 24 * 2 * math.pi).sin()).mean().alias("_hsin"),
            ((pl.col("hora").cast(pl.Float64) / 24 * 2 * math.pi).cos()).mean().alias("_hcos"),
            # modas de contexto (descritores legíveis)
            pl.col("turno").mode().first().alias("turno_moda"),
            pl.col("tipo_pista").mode().first().alias("pista_moda"),
            pl.col("causa_acidente_agrupada").mode().first().alias("causa_moda"),
            pl.col("condicao_meteorologica").mode().first().alias("meteo_moda"),
            (pl.col("uso_solo").eq("Sim").mean() * 100).round(1).alias("pct_urbano"),
            # desfecho RETIDO (caracterização pós-cluster)
            pl.col("indice_gravidade").mean().round(2).alias("ig_media"),
            (pl.col("fatal").mean() * 100).round(2).alias("pct_fatal"),
            pl.col("mortos").mean().round(3).alias("mortos_media"),
            pl.col("feridos_graves").mean().round(3).alias("fg_media"),
        )
        .sort("cluster")
    )
    hsin = base.get_column("_hsin").to_numpy()
    hcos = base.get_column("_hcos").to_numpy()
    hora_circ = np.round((np.arctan2(hsin, hcos) / (2 * math.pi) * 24) % 24, 1)
    return (
        base.with_columns(
            pl.Series("hora_circular", hora_circ),
            (pl.col("n") / total * 100).round(2).alias("pct"),
        )
        .drop("_hsin", "_hcos")
    )


def _dist_classe(df_lab: pl.DataFrame) -> pl.DataFrame:
    """Distribuição (proporção) de `classe_gravidade` por cluster (p/ barras 100%)."""
    return (
        df_lab.group_by("cluster")
        .agg(
            [pl.col("classe_gravidade").eq(c).mean().alias(c) for c in config.CLASSE_ORDEM]
        )
        .sort("cluster")
    )


def _nomear(perfil: pl.DataFrame) -> list[dict[str, Any]]:
    """Nome e descrição de cada cluster (contexto + severidade relativa)."""
    severidade = _niveis(perfil.get_column("ig_media").to_list())
    perfis: list[dict[str, Any]] = []
    for i, row in enumerate(perfil.iter_rows(named=True)):
        nome = f"{row['turno_moda']} · pista {row['pista_moda']} · {severidade[i].lower()} gravidade"
        descricao = (
            f"{row['n']} acidentes ({row['pct']:.1f}%); "
            f"hora típica ~{row['hora_circular']:.0f}h, {row['pct_fds']:.0f}% no fim de semana, "
            f"{row['pct_urbano']:.0f}% em área urbana; "
            f"causa predominante '{row['causa_moda']}', tempo '{row['meteo_moda']}', "
            f"média de {row['veiculos_media']:.1f} veículos; "
            f"desfecho: índice médio {row['ig_media']:.1f}, {row['pct_fatal']:.1f}% fatais, "
            f"{row['mortos_media']:.3f} mortos/acidente."
        )
        perfis.append(
            {"cluster": row["cluster"], "nome": nome, "descricao": descricao,
             "severidade": severidade[i]}
        )
    return perfis


def run() -> dict[str, Any]:
    """Executa a clusterização de acidentes e retorna os artefatos para o relatório."""
    logger.info("--- Clusterização 2: perfis de ocorrência de acidentes ---")
    df = data.load_acidentes()

    # 1) Pré-processamento (dados mistos; sem vazamento de desfecho)
    cm = preprocessing.build_matrix_acidentes(df)

    # 2) Seleção de K (silhouette em amostra fixa, base é grande)
    selection = elbow.evaluate_k(
        cm.X, silhouette_sample_size=config.SILHOUETTE_SAMPLE_SIZE
    )
    k = config.ACIDENTES_K
    logger.info(
        "K escolhido (acidentes) = %d | melhor silhouette em K=%d",
        k, elbow.best_k_by_silhouette(selection),
    )

    # 3) Ajuste final
    labels, _ = cluster.fit_cluster(cm.X, algo="kmeans", k=k)
    df_lab = df.with_columns(pl.Series("cluster", labels))

    # 4) Perfilamento (contexto + desfecho retido) e nomeação
    perfil = _perfil(df_lab)
    classe = _dist_classe(df_lab)
    perfis = _nomear(perfil)
    for p in perfis:
        logger.info("Cluster %d (acidentes) — %s", p["cluster"], p["nome"])

    # 5) Visualizações
    visualization.plot_elbow(
        selection.ks, selection.inertias, "acidentes_elbow",
        "Método do Cotovelo — Acidentes",
    )
    visualization.plot_silhouette(
        selection.ks, selection.silhouettes, "acidentes_silhouette",
        "Silhouette Score — Acidentes",
    )
    pca_var = visualization.plot_pca(
        cm.X, labels, "acidentes_pca", "PCA 2D — Clusters de Acidentes",
    )
    visualization.plot_barras_proporcao(
        classe, "cluster", config.CLASSE_ORDEM, "acidentes_classe_gravidade",
        "Distribuição de classe_gravidade por cluster (desfecho retido)",
    )
    visualization.plot_box_por_cluster(
        df_lab, "veiculos", "cluster", "acidentes_box_veiculos",
        "Veículos envolvidos por cluster (Acidentes)",
    )

    # 6) Persistência dos rótulos
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (
        df_lab.select(
            config.ACIDENTES_ID, "uf", "br", "hora", "turno", "tipo_pista",
            "causa_acidente_agrupada", "cluster",
        ).write_parquet(config.ACIDENTES_CLUSTERS_FILE)
    )
    logger.info("Rótulos de acidentes salvos: %s",
                config.ACIDENTES_CLUSTERS_FILE.relative_to(config.PROJECT_ROOT))

    return {
        "nome": "Acidentes",
        "n": df.height,
        "k": k,
        "selection": selection,
        "pca_var": pca_var,
        "perfil": perfil,
        "classe": classe,
        "perfis": perfis,
        "blocks": cm.blocks,
        "n_features": len(cm.feature_names),
    }
