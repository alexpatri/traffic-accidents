"""Geração do relatório da Feature Engineering em Markdown."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import polars as pl

from src.feature_engineering import config

logger = logging.getLogger(__name__)


def _secao_features(meta: dict[str, Any]) -> str:
    """Monta a seção 'Features criadas'."""
    flags = "\n".join(
        f"| `{coluna}` | Indica traçado do tipo *{cat}*. |"
        for cat, coluna in meta["tracados"]
    )
    return f"""## Features criadas

### Temporais (derivadas de `data` e `horario`)

| Feature | Descrição |
|---|---|
| `hora` | Hora do dia (0–23). |
| `mes` | Mês (1–12). |
| `trimestre` | Trimestre do ano (1–4). |
| `dia_da_semana` | Dia da semana numérico (1=segunda … 7=domingo). |
| `fim_de_semana` | Booleano: sábado ou domingo. |
| `turno` | Madrugada (00–05), Manhã (06–11), Tarde (12–17), Noite (18–23). |

### Gravidade

| Feature | Descrição |
|---|---|
| `indice_gravidade` | {meta['pesos_indice']['mortos']}·mortos + {meta['pesos_indice']['feridos_graves']}·feridos_graves + {meta['pesos_indice']['feridos_leves']}·feridos_leves. |
| `fatal` | Booleano: houve ao menos um morto. |
| `classe_gravidade` | Faixa ordinal do índice: Baixa (0–{config.CLASSE_BAIXA_MAX}), Média ({config.CLASSE_BAIXA_MAX + 1}–{config.CLASSE_MEDIA_MAX}), Alta ({config.CLASSE_MEDIA_MAX + 1}–{config.CLASSE_ALTA_MAX}), Crítica (>{config.CLASSE_ALTA_MAX}). |
| `periodo_noturno` | Booleano: `fase_dia` de baixa luminosidade (Amanhecer/Anoitecer/Plena Noite). |

### Espaciais

| Feature | Descrição |
|---|---|
| `km_faixa` | Km discretizado em faixas de {meta['km_bin_size']} km. |
| `trecho` | Identificador `UF_BR_km_faixa` do segmento rodoviário. |

### Traçado da via (expansão de `tracado_via`, {len(meta['tracados'])} categorias)

| Feature | Descrição |
|---|---|
{flags}

### Categóricas agrupadas

| Feature | Descrição |
|---|---|
| `causa_acidente_agrupada` | `causa_acidente` com categorias raras (< {meta['rare_threshold_pct'] * 100:.1f}%) reunidas em `{config.RARE_LABEL}`. Original preservada. |
"""


def _secao_justificativa() -> str:
    """Monta a seção 'Justificativa' (ligação com a EDA)."""
    return """## Justificativa (achados da EDA)

- **Temporais** — a EDA mostrou pico de acidentes às 17h–19h, concentração no fim de
  semana (domingo/sábado = 31,9%) e sazonalidade mensal (pico em dezembro). `hora`,
  `turno`, `dia_da_semana`, `fim_de_semana`, `mes` e `trimestre` tornam esses padrões
  diretamente analisáveis.
- **Índice de gravidade** — as contagens de vítimas têm baixa correlação independente
  entre si (EDA §correlações); um índice ponderado agrega mais sinal que cada coluna
  isolada. A classe fatal é minoritária (~7%), o que motiva a flag `fatal` e a
  `classe_gravidade` ordinal.
- **`periodo_noturno`** — a EDA observou maior letalidade em Amanhecer (11,0%) e Plena
  Noite (10,1%) frente a Pleno dia (5,0%).
- **Trecho** — a malha viária concentra acidentes em poucas BRs (BR-101, BR-116); um
  identificador espacial estável é a base para o ranking de trechos perigosos.
- **Expansão de `tracado_via`** — campo multivalorado (`Reta;Declive`) responsável pela
  cardinalidade de 898; as flags booleanas resolvem a multivaloração.
- **Agrupamento de causas raras** — `causa_acidente` tem cauda longa (69 categorias);
  agrupar as raras reduz ruído sem perder as causas relevantes.
"""


def _secao_impacto() -> str:
    """Monta a seção 'Impacto esperado'."""
    return """## Impacto esperado

- **Análises estatísticas** — `turno`, `classe_gravidade` e `fim_de_semana` permitem
  recortes diretos; `indice_gravidade` resume a severidade em uma métrica contínua.
- **Clusterização** — features numéricas (`indice_gravidade`, `hora`) e booleanas
  (`tem_*`, `fim_de_semana`, `periodo_noturno`) descrevem cada acidente em um espaço
  interpretável, sem depender de encoding ainda.
- **Machine Learning** — `fatal`/`classe_gravidade` servem de alvo; as features de
  contexto (temporais, traçado, trecho) são preditores candidatos. O dataset por trecho
  habilita modelos e rankings de risco por segmento.
"""


def _secao_decisoes(meta: dict[str, Any]) -> str:
    """Monta a seção 'Decisões tomadas'."""
    tracados = ", ".join(cat for cat, _ in meta["tracados"])
    causas = "\n".join(f"- {c}" for c in meta["causas_raras"]) or "- (nenhuma)"
    return f"""## Decisões tomadas

- **Pesos do índice de gravidade**: mortos = {meta['pesos_indice']['mortos']},
  feridos_graves = {meta['pesos_indice']['feridos_graves']},
  feridos_leves = {meta['pesos_indice']['feridos_leves']} (constantes em `config.py`).
- **Faixa de km do trecho**: {meta['km_bin_size']} km → `trecho = UF_BR_km_faixa`
  ({meta['n_trechos']} trechos distintos). Inclui UF porque o mesmo número de BR cruza
  vários estados.
- **Limiar de causas raras**: frequência relativa < {meta['rare_threshold_pct'] * 100:.1f}%
  → `{config.RARE_LABEL}`. Aplicado apenas a `causa_acidente`; `municipio` preservado.
- **Categorias de `tracado_via` identificadas** ({len(meta['tracados'])}): {tracados}.

### Causas agrupadas em `{config.RARE_LABEL}` ({len(meta['causas_raras'])})

{causas}
"""


def gerar_relatorio(
    meta: dict[str, Any],
    df_acidentes: pl.DataFrame,
    df_trechos: pl.DataFrame,
    path: Path = config.REPORT_FILE,
) -> Path:
    """Gera e grava o relatório Markdown da Feature Engineering.

    Args:
        meta: metadados produzidos pelo pipeline.
        df_acidentes: dataset analítico por acidente.
        df_trechos: dataset agregado por trecho.
        path: caminho de destino do relatório.

    Returns:
        O caminho gravado.
    """
    cabecalho = f"""# Relatório — Feature Engineering (camada Analytics)

Gerado a partir de `data/trusted/acidentes_trusted.parquet`
({df_acidentes.height} acidentes, {df_acidentes.width} colunas) e agregado em
{df_trechos.height} trechos. Todas as features são interpretáveis e independentes de
algoritmo (sem encoding, normalização ou seleção — reservados à Modelagem).
"""
    conteudo = "\n".join(
        [
            cabecalho,
            _secao_features(meta),
            _secao_justificativa(),
            _secao_impacto(),
            _secao_decisoes(meta),
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(conteudo, encoding="utf-8")
    logger.info("Relatório gravado em %s", path.relative_to(config.PROJECT_ROOT))
    return path
