"""Dispatcher de modelos supervisionados (classificação de risco de trechos).

Isola a criação do estimador atrás de um nome de algoritmo, permitindo comparar/trocar
modelos sem alterar features ou avaliação.
"""

from __future__ import annotations

import logging

from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.modeling.ml import config

logger = logging.getLogger(__name__)


def _estimator(algo: str) -> ClassifierMixin:
    """Estimador base (sem pipeline), com `class_weight="balanced"`."""
    algo = algo.lower()
    if algo == "logreg":
        return LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=config.RANDOM_STATE,
        )
    if algo == "random_forest":
        return RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=config.RANDOM_STATE, n_jobs=-1,
        )
    if algo == "hist_gboost":
        return HistGradientBoostingClassifier(
            class_weight="balanced", random_state=config.RANDOM_STATE,
        )
    raise ValueError(f"Algoritmo de classificação desconhecido: {algo!r}")


def make_model(algo: str = "random_forest") -> Pipeline:
    """Pipeline `StandardScaler -> clf` (não ajustado).

    A padronização ajuda a Regressão Logística (e é inócua para as árvores, cujas
    divisões independem de escala), unificando a nomenclatura `clf__*` para o tuning.
    """
    return Pipeline([("scaler", StandardScaler()), ("clf", _estimator(algo))])
