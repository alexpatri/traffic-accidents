"""Orquestração da etapa de ML Supervisionado.

Classificação ordinal do risco do trecho (4 níveis). Constrói as features, compara
variantes × modelos, seleciona a melhor combinação, persiste o modelo final (treinado em
toda a base) e as predições, gera figuras e relatório.

Execução:
    python -m src.modeling.ml.main
"""

from __future__ import annotations

import logging
import warnings

import joblib
import polars as pl
from sklearn.base import clone

from src.modeling.ml import config, data, evaluate, features, report, visualization

logger = logging.getLogger("ml")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler()],
    )
    # Aviso cosmético do lbfgs/scipy (opção 'iprint') — não afeta o ajuste.
    warnings.filterwarnings("ignore", message="Unknown solver options")


def _persistir(table: pl.DataFrame, result: evaluate.EvalResult) -> None:
    """Treina o modelo final (hiperparâmetros ajustados) na base completa e persiste."""
    variante = result.best["variante"]
    X, y, _ = evaluate.prepare_xy(table, variante)
    # Reaproveita o pipeline ajustado (com os hiperparâmetros escolhidos no tuning),
    # re-treinando em TODA a base filtrada.
    final = clone(result.best["fitted"]).fit(X, y)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, config.MODEL_FILE)
    logger.info("Modelo final salvo: %s", config.MODEL_FILE.relative_to(config.PROJECT_ROOT))

    rotulo = {i: lab for i, lab in enumerate(config.CLASSE_LABELS)}
    preds = [rotulo[int(p)] for p in final.predict(X)]
    (
        table.select(config.TRECHO_ID, "uf", "regiao", config.TARGET_BASE, "classe_risco")
        .with_columns(pl.Series("classe_prevista", preds))
        .write_parquet(config.PRED_FILE)
    )
    logger.info("Predições salvas: %s", config.PRED_FILE.relative_to(config.PROJECT_ROOT))


def run() -> None:
    """Executa a etapa de ML supervisionado completa."""
    _setup_logging()
    logger.info("=== Início do ML Supervisionado — risco de trechos (PRF) ===")

    df_acidentes = data.load_acidentes()
    df_trechos = data.load_trechos()
    table = features.build_feature_table(df_acidentes, df_trechos)
    logger.info("Distribuição das classes:\n%s", features.class_distribution(table))

    result = evaluate.run_experiments(table)

    visualization.plot_class_distribution(
        result.dist, "distribuicao_classes", "Distribuição das classes de risco")
    visualization.plot_variant_comparison(
        result.combos, "comparacao_variantes", "CV F1 macro por variante × modelo")
    visualization.plot_confusion(
        result.best["confusion"], "matriz_confusao",
        f"Matriz de confusão — {result.best['variante']} + {result.best['algo']}")
    visualization.plot_feature_importance(
        result.best["importance"], "importancia_features",
        f"Importância das features — {result.best['algo']}")

    _persistir(table, result)
    report.report(result)

    logger.info("=== ML Supervisionado concluído — artefatos em outputs/ml/ ===")


if __name__ == "__main__":
    run()
