"""Geração do relatório consolidado da Clusterização em Markdown.

A entrega é a clusterização de trechos. A clusterização de acidentes foi investigada e
não incluída — o relatório documenta esse achado negativo (seção 5).
"""

from __future__ import annotations

import logging
from typing import Any

import polars as pl

from src.modeling.clustering import config
from src.modeling.clustering.elbow import SelectionResult

logger = logging.getLogger(__name__)


def _tabela_selecao(sel: SelectionResult) -> str:
    """Tabela markdown K × inertia × silhouette."""
    linhas = "\n".join(
        f"| {k} | {iner:,.0f} | {sil:.4f} |"
        for k, iner, sil in zip(sel.ks, sel.inertias, sel.silhouettes)
    )
    return "| K | Inertia (SSE) | Silhouette |\n|---|---|---|\n" + linhas


def _tabela_polars(df: pl.DataFrame, colunas: list[str], titulos: list[str]) -> str:
    """Renderiza colunas selecionadas de um DataFrame Polars como tabela markdown."""
    head = "| " + " | ".join(titulos) + " |\n|" + "---|" * len(titulos)
    linhas = []
    for row in df.iter_rows(named=True):
        vals = []
        for c in colunas:
            v = row[c]
            vals.append(f"{v:.2f}" if isinstance(v, float) else str(v))
        linhas.append("| " + " | ".join(vals) + " |")
    return head + "\n" + "\n".join(linhas)


def _secao_trechos(r: dict[str, Any]) -> str:
    perfis = "\n".join(
        f"- **Cluster {p['cluster']} — {p['nome']}**: {p['descricao']}"
        for p in r["perfis"]
    )
    tabela = _tabela_polars(
        r["perfil"],
        ["cluster", "n", "pct", "qtd_media", "ig_medio_media", "pct_fatais_media",
         "mortos_total", "feridos_graves_total"],
        ["Cluster", "n", "%", "Qtd.acid (méd)", "Índ.médio", "% fatais",
         "Mortos (tot)", "Fer.graves (tot)"],
    )
    sens = r["sensibilidade"]
    return f"""## 1. Clusterização dos Trechos

**Dataset:** `data/analytics/trechos_analytics.parquet` ({r['n']:,} trechos).
**Features de formação:** {", ".join(f"`{f}`" for f in r['features'])}
(frequência, severidade total, risco médio por evento e letalidade).
**Número de clusters:** **K = {r['k']}**.

### Perfil de cada grupo

{tabela}

> Médias das features de formação e o **desfecho retido** (`mortos`, `feridos_graves`)
> — este último usado apenas para caracterizar os grupos, não para formá-los.

### Interpretação dos perfis

{perfis}

![Clusters de trechos em PCA 2D](figures/trechos_pca.png)

*Projeção PCA (somente visualização): {r['pca_var']:.1%} da variância em 2D.*

### Sensibilidade (redundância qtd ↔ índice total, r≈0,89)

Repetindo a clusterização sem `indice_gravidade_total` (3 features), o Silhouette em
K={sens['k']} passou de {sens['silhouette_4feat']:.4f} (4 features) para
{sens['silhouette_3feat']:.4f} (3 features) — documentado para transparência da decisão
de manter as 4 features pedidas no enunciado.

![Volume por cluster](figures/trechos_box_qtd_acidentes.png)
![Letalidade por cluster](figures/trechos_box_pct_acidentes_fatais.png)
"""


def _secao_cotovelo(r: dict[str, Any]) -> str:
    return f"""## 2. Método do Cotovelo

A inertia (SSE intra-cluster) cai com K; busca-se o ponto de inflexão a partir do qual o
ganho marginal diminui.

{_tabela_selecao(r['selection'])}

![Cotovelo — Trechos](figures/trechos_elbow.png)

Forte queda até K=3–4 (87.557 → 52.556 → 41.744) e desaceleração depois — inflexão na
região de K=3–4; escolhido **K={r['k']}**.
"""


def _secao_silhouette(r: dict[str, Any]) -> str:
    return f"""## 3. Silhouette Score

Mede coesão × separação (−1 a 1). Calculado sobre amostra fixa de
{r['selection'].sample_size:,} trechos (a métrica é O(n²) e inviável na base completa).

![Silhouette — Trechos](figures/trechos_silhouette.png)

**Comparação com o cotovelo + escolha de K.** O Silhouette favorece K menores (pico em
K=3 ≈ 0,48), enquanto o cotovelo admite K maiores. Optou-se por **K={r['k']}** por
**interpretabilidade**: K=4 revela a estrutura 2×2 *volume × letalidade* — incluindo o
grupo raro porém extremamente letal — a um custo mínimo de Silhouette frente a K=3. A
decisão não foi apenas visual: concilia cotovelo, Silhouette e leitura dos perfis.
"""


def _secao_descobertas(r: dict[str, Any]) -> str:
    pt = r["perfil"].sort("pct_fatais_media", descending=True).row(0, named=True)
    return f"""## 4. Principais Descobertas

**Perfis de trechos encontrados.** A malha se separa em dois eixos — *volume/exposição*
(qtd de acidentes, índice total) e *letalidade por evento* (índice médio, % fatal):
corredores de alto volume e baixa letalidade, trechos raros porém letais, e grupos
intermediário/benigno.

**Grupo claramente mais crítico.** O **Cluster {pt['cluster']}**
({pt['pct_fatais_media']:.1f}% de acidentes fatais em média, {pt['mortos_total']:,} mortos)
reúne trechos de baixíssimo volume mas letalidade extrema — invisíveis num ranking por
frequência. Confirma o insight da EDA de que **volume ≠ gravidade**.

**Como orientar recomendações de investimento.**
- *Trechos raros e letais* → **intervenções pontuais de engenharia/sinalização**, de alto
  retorno em vidas.
- *Corredores de alto volume* → **fiscalização, capacidade e fluidez** (concentram feridos
  por exposição, não por gravidade intrínseca).
- *Trechos benignos* → baixa prioridade.

A quantificação do efeito de cada fator fica para a etapa de ML supervisionado. As leituras
aqui são associativas, **não causais**.
"""


def _secao_acidentes() -> str:
    return """## 5. Clusterização de Acidentes — investigada e não incluída

A clusterização de acidentes foi testada, mas **não produziu perfis multivariados
nítidos** e por isso não compõe a entrega (fica documentada aqui).

- **Sem cotovelo** (com todas as features): a inertia cai de forma suave e linear
  (88.781 → 82.787 → 77.421 → 73.826…), sem inflexão.
- **Silhouette ≈ 0,13–0,17** (abaixo de 0,25) e **PCA 2D explica só ~28%** da variância —
  os acidentes não se separam em grupos compactos.
- **Reduzir features** (temporal + via + causa) elevou o Silhouette para ~0,27, mas os
  grupos resultantes **apenas redescobrem a `tipo_pista`**: em K=3, os clusters têm
  exatamente 70.020 / 61.512 / 14.153 registros — os tamanhos de Simples / Dupla /
  Múltipla. Não são arquétipos novos; é uma única variável categórica dominante.

**Conclusão.** Os acidentes variam num **continuum** estruturado sobretudo pela
`tipo_pista` — e pista **Simples** concentra severidade (índice ~5,2; ~9,9% fatais) vs
Dupla (~3,7; ~4,8%) e Múltipla (~3,5; ~4,0%). Isso **confirma a EDA**, não revela perfis
inéditos. Parte do efeito é metodológica (KMeans euclidiano é fraco em dados
majoritariamente one-hot); um método nativo para categóricos (K-prototypes) tenderia a
recuperar a mesma partição. O conhecimento sobre contexto → severidade dos acidentes é
melhor explorado pela EDA Analytics e pela etapa de ML supervisionado.
"""


def _secao_metodologia() -> str:
    return """## 6. Metodologia e Limitações

**Pré-processamento (trechos).** `log1p` em `qtd_acidentes` e `indice_gravidade_total`
(cauda longa); `indice_gravidade_medio` e `pct_acidentes_fatais` mantidos (skew baixo /
massa em 0); `StandardScaler` nas 4. A severidade agregada é tratada como **descritor de
perfil do trecho** (não desfecho de um evento a prever), por isso entra na formação —
conforme o enunciado.

**Algoritmo.** K-Means (`n_init=10`, `random_state` fixo). A arquitetura isola a troca de
algoritmo em `cluster.fit_cluster(X, algo, k)`, permitindo Agglomerative/DBSCAN sem mexer
no pré-processamento.

**Limitações.** Interpretações são associativas, não causais. `pct_acidentes_fatais` é
ruidoso em trechos de 1 acidente (0% ou 100%). O índice de gravidade depende dos pesos
(12/6/2). Janela temporal curta (2024–2025).
"""


def report(trechos_result: dict[str, Any]) -> str:
    """Gera e grava o relatório consolidado da clusterização (trechos).

    Returns:
        O caminho do relatório gravado (como string).
    """
    cabecalho = """# Relatório — Clusterização (Aprendizado Não Supervisionado)

Etapa de descoberta de **perfis naturais** (não rankings) na malha rodoviária federal. A
entrega é a **clusterização de trechos rodoviários**; a clusterização de acidentes foi
investigada e documentada como achado negativo (seção 5). Base para a etapa de ML
supervisionado e para as recomendações de priorização de investimentos.
"""
    conteudo = "\n".join(
        [
            cabecalho,
            _secao_trechos(trechos_result),
            _secao_cotovelo(trechos_result),
            _secao_silhouette(trechos_result),
            _secao_descobertas(trechos_result),
            _secao_acidentes(),
            _secao_metodologia(),
        ]
    )
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORT_FILE.write_text(conteudo, encoding="utf-8")
    logger.info("Relatório gravado em %s",
                config.REPORT_FILE.relative_to(config.PROJECT_ROOT))
    return str(config.REPORT_FILE)
