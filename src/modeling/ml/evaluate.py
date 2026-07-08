"""Treino, validação e métricas da classificação de risco de trechos.

Validação **por grupo (rodovia/BR)**: o holdout e a CV mantêm as BRs disjuntas
(`StratifiedGroupKFold`), medindo a generalização a vias NÃO vistas — o objetivo do
modelo. Hiperparâmetros ajustados por `RandomizedSearchCV`. Métricas tratadas como
ORDINAIS (F1 macro, balanced accuracy, kappa quadrático) + visão-resumo binária.
"""

from __future__ import annotations

import logging
from typing import Any, NamedTuple

import numpy as np
import polars as pl
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedGroupKFold,
    cross_val_score,
)

from src.modeling.ml import config, features, model

logger = logging.getLogger(__name__)

_ALTO_IDX: int = min(config.CLASSE_LABELS.index(c) for c in config.ALTO_RISCO_CLASSES)


class ComboResult(NamedTuple):
    """Métricas de uma combinação (variante × algoritmo), validação por grupo."""

    variante: str
    algo: str
    n_features: int
    cv_f1_macro: float
    test_f1_macro: float
    test_balanced_acc: float
    test_qwk: float
    bin_f1: float
    bin_recall: float
    bin_auc: float


class EvalResult(NamedTuple):
    """Resultado completo da etapa de avaliação."""

    combos: list[ComboResult]
    best: dict[str, Any]
    cortes: list[float]
    dist: pl.DataFrame


def prepare_xy(table: pl.DataFrame, variante: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Monta (X, y, nomes) de uma variante a partir da tabela de features."""
    X_df, names = features.select_features(table, variante)
    X = X_df.to_numpy().astype(np.float64)
    y = features.encode_target(table)
    return X, y, names


def _bin(y: np.ndarray) -> np.ndarray:
    """Mapeia rótulos ordinais para a visão binária `alto_risco` (Alta∪Crítica)."""
    return (y >= _ALTO_IDX).astype(int)


def _feature_importance(fitted: Any, names: list[str]) -> list[tuple[str, float]]:
    """Importância: `feature_importances_` (árvores) ou |coef| médio (logreg)."""
    est = fitted.named_steps["clf"] if hasattr(fitted, "named_steps") else fitted
    if hasattr(est, "feature_importances_"):
        vals = np.asarray(est.feature_importances_, dtype=float)
    elif hasattr(est, "coef_"):
        vals = np.abs(np.asarray(est.coef_, dtype=float)).mean(axis=0)
    else:
        return []
    return sorted(zip(names, vals.tolist()), key=lambda t: t[1], reverse=True)


def _grid_size(grid: dict) -> int:
    size = 1
    for v in grid.values():
        size *= len(v)
    return size


def _fit_tuned(X_tr, y_tr, g_tr, algo, inner):
    """Ajusta com RandomizedSearchCV (ou defaults se TUNE_ITER=0); retorna (modelo, cv_f1)."""
    if config.TUNE_ITER > 0 and algo in config.PARAM_GRIDS:
        grid = config.PARAM_GRIDS[algo]
        n_iter = min(config.TUNE_ITER, _grid_size(grid))
        search = RandomizedSearchCV(
            model.make_model(algo), grid, n_iter=n_iter, scoring="f1_macro",
            cv=inner, random_state=config.RANDOM_STATE, n_jobs=-1,
        )
        search.fit(X_tr, y_tr, groups=g_tr)
        return search.best_estimator_, float(search.best_score_)
    est = model.make_model(algo)
    cv = cross_val_score(est, X_tr, y_tr, groups=g_tr, cv=inner,
                         scoring="f1_macro", n_jobs=-1)
    est.fit(X_tr, y_tr)
    return est, float(cv.mean())


def _evaluate_combo(
    table: pl.DataFrame, variante: str, algo: str
) -> tuple[ComboResult, dict[str, Any]]:
    """Treina/ajusta e avalia uma combinação com validação por grupo (BR)."""
    X, y, names = prepare_xy(table, variante)
    groups = table.get_column(config.GROUP_COL).to_numpy()

    # Holdout por grupo: ~TEST_SIZE, BRs disjuntas, classes estratificadas.
    outer = StratifiedGroupKFold(n_splits=round(1 / config.TEST_SIZE), shuffle=True,
                                 random_state=config.RANDOM_STATE)
    tr, te = next(outer.split(X, y, groups))
    inner = StratifiedGroupKFold(n_splits=config.CV_FOLDS, shuffle=True,
                                 random_state=config.RANDOM_STATE)

    fitted, cv_f1 = _fit_tuned(X[tr], y[tr], groups[tr], algo, inner)
    y_te, y_pred = y[te], fitted.predict(X[te])
    proba = fitted.predict_proba(X[te])

    yb_te, proba_alto = _bin(y_te), proba[:, _ALTO_IDX:].sum(axis=1)
    res = ComboResult(
        variante=variante, algo=algo, n_features=len(names),
        cv_f1_macro=round(cv_f1, 4),
        test_f1_macro=round(float(f1_score(y_te, y_pred, average="macro")), 4),
        test_balanced_acc=round(float(balanced_accuracy_score(y_te, y_pred)), 4),
        test_qwk=round(float(cohen_kappa_score(y_te, y_pred, weights="quadratic")), 4),
        bin_f1=round(float(f1_score(yb_te, _bin(y_pred))), 4),
        bin_recall=round(float(recall_score(yb_te, _bin(y_pred))), 4),
        bin_auc=round(float(roc_auc_score(yb_te, proba_alto)), 4),
    )
    art = {
        "fitted": fitted, "names": names, "y_te": y_te, "y_pred": y_pred,
        "confusion": confusion_matrix(y_te, y_pred, labels=list(range(len(config.CLASSE_LABELS)))),
    }
    logger.info("  %-13s | %-12s | CV f1=%.3f | teste f1=%.3f | QWK=%.3f | bin AUC=%.3f",
                variante, algo, res.cv_f1_macro, res.test_f1_macro, res.test_qwk, res.bin_auc)
    return res, art


def run_experiments(table: pl.DataFrame) -> EvalResult:
    """Roda todas as combinações variante × algoritmo e seleciona a melhor (CV f1 macro)."""
    combos: list[ComboResult] = []
    artefatos_por_combo: dict[tuple[str, str], dict[str, Any]] = {}
    for variante in config.VARIANTES:
        for algo in config.ALGOS:
            res, art = _evaluate_combo(table, variante, algo)
            combos.append(res)
            artefatos_por_combo[(variante, algo)] = art

    # Modelo de entrega: melhor algoritmo dentro da variante transferível recomendada.
    candidatos = [c for c in combos if c.variante == config.VARIANTE_ENTREGA]
    best_combo = max(candidatos, key=lambda c: c.cv_f1_macro)
    best_art = artefatos_por_combo[(best_combo.variante, best_combo.algo)]
    best = {
        "variante": best_combo.variante,
        "algo": best_combo.algo,
        "metrics": best_combo,
        "importance": _feature_importance(best_art["fitted"], best_art["names"]),
        **best_art,
    }
    logger.info("Melhor combinação: %s + %s (CV f1 macro=%.3f)",
                best_combo.variante, best_combo.algo, best_combo.cv_f1_macro)
    return EvalResult(
        combos=combos,
        best=best,
        cortes=features.compute_cortes(table),
        dist=features.class_distribution(table),
    )
