"""Transform: limpeza, padronização, tipagem, validação e união dos datasets.

Cada função recebe e devolve um `pl.DataFrame`, usando expressões vetorizadas do
Polars (sem loops sobre linhas). A orquestração fica em `transform()`.
"""

from __future__ import annotations

import logging
import unicodedata

import polars as pl

from src import config

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# 1. Padronização de nomes de colunas
# --------------------------------------------------------------------------- #
def _normalize_name(name: str) -> str:
    """Normaliza um nome de coluna: minúsculas, sem acento, espaços -> '_'."""
    nfkd = unicodedata.normalize("NFKD", name)
    sem_acento = "".join(c for c in nfkd if not unicodedata.combining(c))
    return sem_acento.strip().lower().replace(" ", "_")


def normalize_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Aplica normalização genérica + renomeações semânticas do `RENAME_MAP`."""
    normalized = {col: _normalize_name(col) for col in df.columns}
    df = df.rename(normalized)
    rename = {k: v for k, v in config.RENAME_MAP.items() if k in df.columns}
    return df.rename(rename)


# --------------------------------------------------------------------------- #
# 2/3. Limpeza de texto e tokens nulos
# --------------------------------------------------------------------------- #
def clean_strings(df: pl.DataFrame) -> pl.DataFrame:
    """Remove espaços extras e converte tokens de ausência em nulo.

    Atua sobre todas as colunas de texto: faz `strip`, colapsa espaços internos
    múltiplos e transforma tokens como "NA"/"" (já trimmed) em nulo. Não altera a
    caixa dos valores — a exploração mostrou categorias já consistentes.
    """
    null_set = {t.strip().upper() for t in config.NULL_TOKENS}
    text_cols = [c for c, dt in df.schema.items() if dt == pl.Utf8]

    exprs = []
    for col in text_cols:
        cleaned = (
            pl.col(col)
            .str.strip_chars()
            .str.replace_all(r"\s+", " ")  # colapsa espaços internos
        )
        # tokens de ausência (case-insensitive) -> nulo
        cleaned = (
            pl.when(cleaned.str.to_uppercase().is_in(list(null_set)))
            .then(None)
            .otherwise(cleaned)
            .alias(col)
        )
        exprs.append(cleaned)
    return df.with_columns(exprs)


# --------------------------------------------------------------------------- #
# 4. Conversão de tipos
# --------------------------------------------------------------------------- #
def cast_types(df: pl.DataFrame) -> pl.DataFrame:
    """Converte cada coluna para seu tipo alvo.

    Datas/horas via `strptime`; colunas decimais têm o separador "," trocado por
    "." antes do cast para `Float64`; contagens viram `Int32`; `id` vira `Int64`.
    `br` permanece string por ser um código identificador, não uma quantidade.
    """
    exprs: list[pl.Expr] = [
        # cast via Float64 recupera ids exportados em notação científica
        # (ex.: "6e+05" -> 600000) que falhariam no cast direto para inteiro.
        pl.col(config.ID_COL).cast(pl.Float64, strict=False).cast(pl.Int64),
        pl.col(config.DATE_COL).str.strptime(pl.Date, "%Y-%m-%d", strict=False),
        pl.col(config.TIME_COL).str.strptime(pl.Time, "%H:%M:%S", strict=False),
    ]

    for col in config.DECIMAL_COLS:
        exprs.append(
            pl.col(col).str.replace_all(",", ".").cast(pl.Float64, strict=False)
        )

    for col in config.INT_COUNT_COLS:
        exprs.append(pl.col(col).cast(pl.Int32, strict=False))

    return df.with_columns(exprs)


# --------------------------------------------------------------------------- #
# 5. Preenchimento de categóricas
# --------------------------------------------------------------------------- #
def fill_categoricals(df: pl.DataFrame) -> pl.DataFrame:
    """Preenche nulos residuais em colunas categóricas com "Não informado"."""
    cols = [c for c in config.CATEGORICAL_COLS if c in df.columns]
    return df.with_columns(
        pl.col(cols).fill_null(config.NAO_INFORMADO)
    )


# --------------------------------------------------------------------------- #
# 6. Validação de consistência (apenas registra; não altera dados)
# --------------------------------------------------------------------------- #
def validate(df: pl.DataFrame, label: str) -> dict[str, int]:
    """Conta violações de regras de domínio e registra em log.

    Não remove nem corrige linhas — a exploração mostrou violações ~nulas e o foco
    é rastreabilidade. Retorna um relatório {regra: contagem}.
    """
    count_cols = [c for c in config.INT_COUNT_COLS if c in df.columns]
    parts = [c for c in config.VICTIM_PARTS if c in df.columns]

    soma_parts = pl.sum_horizontal([pl.col(c) for c in parts])
    neg_counts = pl.any_horizontal([pl.col(c) < 0 for c in count_cols])

    report = df.select(
        km_negativo=(pl.col("km") < 0).sum(),
        contagem_negativa=neg_counts.sum(),
        lat_fora_intervalo=(
            ~pl.col("latitude").is_between(config.LAT_MIN, config.LAT_MAX)
        ).sum(),
        lon_fora_intervalo=(
            ~pl.col("longitude").is_between(config.LON_MIN, config.LON_MAX)
        ).sum(),
        data_nula=pl.col(config.DATE_COL).is_null().sum(),
        horario_nulo=pl.col(config.TIME_COL).is_null().sum(),
        pessoas_diverge_soma=(pl.col("pessoas") != soma_parts).sum(),
    ).row(0, named=True)

    report = {k: int(v) for k, v in report.items()}
    logger.info("Validação [%s]: %s", label, report)
    return report


# --------------------------------------------------------------------------- #
# 7. Harmonização de schema
# --------------------------------------------------------------------------- #
def harmonize_schema(df: pl.DataFrame) -> pl.DataFrame:
    """Garante a ordem canônica de colunas (defensivo antes da união)."""
    missing = [c for c in config.COLUMN_ORDER if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes após transformação: {missing}")
    return df.select(config.COLUMN_ORDER)


# --------------------------------------------------------------------------- #
# 8. Deduplicação
# --------------------------------------------------------------------------- #
def deduplicate(df: pl.DataFrame, label: str) -> pl.DataFrame:
    """Remove linhas idênticas e registra duplicidades de `id` e de chave natural."""
    before = df.height
    df = df.unique(keep="first")
    removed = before - df.height

    dup_ids = df.height - df[config.ID_COL].n_unique()
    key_cols = [config.DATE_COL, config.TIME_COL, "uf", "br", "km", "municipio"]
    dup_keys = df.height - df.select(key_cols).n_unique()

    logger.info(
        "Dedup [%s]: linhas idênticas removidas=%d | ids duplicados=%d | "
        "chaves data+hora+local repetidas (mantidas)=%d",
        label, removed, dup_ids, dup_keys,
    )
    return df


# --------------------------------------------------------------------------- #
# Pipeline por ano + união
# --------------------------------------------------------------------------- #
def transform_year(df: pl.DataFrame, label: str) -> tuple[pl.DataFrame, dict[str, int]]:
    """Aplica toda a cadeia de transformação a um único dataset."""
    df = normalize_columns(df)
    df = clean_strings(df)
    df = cast_types(df)
    df = fill_categoricals(df)
    report = validate(df, label)
    df = harmonize_schema(df)
    df = deduplicate(df, label)
    return df, report


def transform(frames: dict[int, pl.DataFrame]) -> tuple[pl.DataFrame, dict]:
    """Transforma cada ano, concatena e devolve o DF final + relatório consolidado.

    Returns:
        (DataFrame unificado, dicionário de relatórios por ano).
    """
    processed: list[pl.DataFrame] = []
    reports: dict = {}

    for year, df in frames.items():
        out, report = transform_year(df, str(year))
        processed.append(out)
        reports[year] = report

    united = pl.concat(processed, how="vertical")
    logger.info(
        "União concluída: %d linhas, %d colunas", united.height, united.width
    )

    # Checagem final de duplicidade de id entre anos (esperado: 0).
    dup_ids_total = united.height - united[config.ID_COL].n_unique()
    if dup_ids_total:
        logger.warning("IDs duplicados após união entre anos: %d", dup_ids_total)

    return united, reports
