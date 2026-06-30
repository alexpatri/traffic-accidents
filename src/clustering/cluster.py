"""Dispatcher de algoritmos de clusterização.

KMeans é o algoritmo desta etapa; a função `fit_cluster` isola a chamada atrás de
um nome de algoritmo para que Agglomerative/DBSCAN possam ser plugados depois sem
tocar no pré-processamento, na seleção de K ou na interpretação.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.base import ClusterMixin
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans

from src.clustering import config

logger = logging.getLogger(__name__)


def make_model(algo: str = "kmeans", k: int | None = None, **kwargs) -> ClusterMixin:
    """Instancia (sem ajustar) o estimador de clusterização pedido.

    Args:
        algo: "kmeans" | "agglomerative" | "dbscan".
        k: número de clusters (ignorado por DBSCAN).
        **kwargs: hiperparâmetros extras repassados ao estimador.

    Returns:
        Estimador scikit-learn não ajustado.
    """
    algo = algo.lower()
    if algo == "kmeans":
        return KMeans(
            n_clusters=k,
            n_init=config.KMEANS_N_INIT,
            random_state=config.RANDOM_STATE,
            **kwargs,
        )
    if algo == "agglomerative":
        return AgglomerativeClustering(n_clusters=k, **kwargs)
    if algo == "dbscan":
        return DBSCAN(**kwargs)
    raise ValueError(f"Algoritmo de clusterização desconhecido: {algo!r}")


def fit_cluster(
    X: np.ndarray, algo: str = "kmeans", k: int | None = None, **kwargs
) -> tuple[np.ndarray, ClusterMixin]:
    """Ajusta o algoritmo e retorna (rótulos, modelo ajustado).

    Args:
        X: matriz de features escalada.
        algo: nome do algoritmo (ver `make_model`).
        k: número de clusters.
        **kwargs: hiperparâmetros extras.

    Returns:
        Tupla (labels, modelo). `labels[i]` é o cluster da linha `i` de `X`.
    """
    model = make_model(algo, k, **kwargs)
    labels = model.fit_predict(X)
    n_clusters = len(set(labels) - {-1})
    logger.info("%s ajustado: %d clusters em %d pontos", algo, n_clusters, X.shape[0])
    return labels, model
