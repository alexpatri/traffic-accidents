"""Geração do relatório consolidado da Clusterização em Markdown."""

from __future__ import annotations

import logging
from typing import Any

import polars as pl

from src.clustering import config
from src.clustering.elbow import SelectionResult

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

> Médias/medianas das features de formação e o **desfecho retido** (`mortos`,
> `feridos_graves`, `indice_gravidade_maximo`) — este último usado apenas para
> caracterizar os grupos, não para formá-los.

### Interpretação dos perfis

{perfis}

![Clusters de trechos em PCA 2D](figures/trechos_pca.png)

*Projeção PCA (somente visualização): {r['pca_var']:.1%} da variância em 2D.*

### Sensibilidade (redundância qtd ↔ índice total, r≈0,89)

Repetindo a clusterização sem `indice_gravidade_total` (3 features), o Silhouette em
K={sens['k']} passou de {sens['silhouette_4feat']:.4f} (4 features) para
{sens['silhouette_3feat']:.4f} (3 features) — documentado para transparência da
decisão de manter as 4 features pedidas no enunciado.

![Volume por cluster](figures/trechos_box_qtd_acidentes.png)
![Letalidade por cluster](figures/trechos_box_pct_acidentes_fatais.png)
"""


def _secao_acidentes(r: dict[str, Any]) -> str:
    perfis = "\n".join(
        f"- **Cluster {p['cluster']} — {p['nome']}**: {p['descricao']}"
        for p in r["perfis"]
    )
    tabela = _tabela_polars(
        r["perfil"],
        ["cluster", "n", "pct", "hora_circular", "pct_fds", "turno_moda",
         "pista_moda", "ig_media", "pct_fatal"],
        ["Cluster", "n", "%", "Hora típ.", "% FDS", "Turno",
         "Pista", "Índ.méd", "% fatal"],
    )
    blocos = ", ".join(f"{b} ({len(c)})" for b, c in r["blocks"].items())
    return f"""## 2. Clusterização dos Acidentes

**Dataset:** `data/analytics/acidentes_analytics.parquet` ({r['n']:,} acidentes).
**Matriz:** {r['n_features']} colunas em blocos ponderados — {blocos}.
Somente variáveis conhecidas no momento do acidente (sem vazamento de desfecho).
**Número de clusters:** **K = {r['k']}**.

### Perfil de cada grupo

{tabela}

> `ig_media` e `% fatal` são o **desfecho retido**: descrevem a associação
> contexto → severidade observada em cada grupo, sem terem sido usados na formação.

### Interpretação dos perfis

{perfis}

![Clusters de acidentes em PCA 2D](figures/acidentes_pca.png)

*Projeção PCA (somente visualização): {r['pca_var']:.1%} da variância em 2D — baixa,
como esperado em dados majoritariamente one-hot; clusters sobrepostos na tela podem
estar separados no espaço completo.*

![Classe de gravidade por cluster](figures/acidentes_classe_gravidade.png)
![Veículos por cluster](figures/acidentes_box_veiculos.png)
"""


def _secao_cotovelo(t: dict[str, Any], a: dict[str, Any]) -> str:
    return f"""## 3. Método do Cotovelo

A inertia (SSE intra-cluster) cai monotonicamente com K; busca-se o ponto de inflexão
a partir do qual o ganho marginal diminui.

### Trechos

{_tabela_selecao(t['selection'])}

![Cotovelo — Trechos](figures/trechos_elbow.png)

Inflexão na região de K=3–4; escolhido **K={t['k']}**.

### Acidentes

{_tabela_selecao(a['selection'])}

![Cotovelo — Acidentes](figures/acidentes_elbow.png)

Curva suave (típico de dados mistos one-hot); inflexão difusa na região de K=4–6,
escolhido **K={a['k']}**.
"""


def _secao_silhouette(t: dict[str, Any], a: dict[str, Any]) -> str:
    return f"""## 4. Silhouette Score

Mede coesão × separação (−1 a 1). Calculado sobre amostra fixa de
{t['selection'].sample_size:,} (trechos) e {a['selection'].sample_size:,} (acidentes)
pontos — a métrica é O(n²) e inviável nas bases completas.

![Silhouette — Trechos](figures/trechos_silhouette.png)
![Silhouette — Acidentes](figures/acidentes_silhouette.png)

**Comparação com o cotovelo:** o silhouette tende a favorecer K menores (grupos mais
separados), enquanto o cotovelo admite K maiores. A escolha final concilia ambos com a
**interpretabilidade** — o menor K cujos perfis contam uma história distinta e nomeável.
Trechos: K={t['k']}. Acidentes: K={a['k']}.
"""


def _secao_descobertas(t: dict[str, Any], a: dict[str, Any]) -> str:
    # Cluster de trechos mais crítico = maior letalidade média.
    pt = t["perfil"].sort("pct_fatais_media", descending=True).row(0, named=True)
    pa = a["perfil"].sort("ig_media", descending=True).row(0, named=True)
    return f"""## 5. Principais Descobertas

**Perfis de trechos encontrados.** A malha se separa essencialmente em dois eixos —
*volume/exposição* (qtd de acidentes, índice total) e *letalidade por evento* (índice
médio, % fatal). Surgem perfis como alto-volume/baixa-letalidade (corredores movimentados),
baixo-volume/alta-letalidade (trechos pontuais porém letais) e grupos intermediários.
O grupo mais crítico em letalidade é o **Cluster {pt['cluster']}**
({pt['pct_fatais_media']:.1f}% de acidentes fatais em média).

**Perfis de acidentes encontrados.** Os grupos combinam janela temporal (hora circular,
fim de semana), tipo de pista, uso do solo, causa predominante e meteorologia. O grupo
de maior severidade observada (desfecho retido) é o **Cluster {pa['cluster']}**
(índice médio {pa['ig_media']:.1f}, {pa['pct_fatal']:.1f}% fatais).

**Existem grupos claramente mais críticos?** Sim — em ambos os níveis há grupos
destacados: trechos de alta letalidade e contextos de acidente associados a desfechos
mais graves. Isso é associação contexto→severidade, **não relação causal**.

**Como orientar recomendações de investimento.** Os trechos de alta letalidade (mesmo
com baixo volume) sugerem intervenções de engenharia/sinalização pontuais de alto retorno
em vidas; os de alto volume sugerem fiscalização e capacidade. Os perfis de acidentes
indicam quando/onde concentrar fiscalização e campanhas (turno, fim de semana, causa,
condição da via) — a ser quantificado na etapa seguinte de ML supervisionado.
"""


def _secao_metodologia(a: dict[str, Any]) -> str:
    return """## 6. Metodologia e Limitações

**Pré-processamento.**
- *Trechos:* `log1p` em `qtd_acidentes` e `indice_gravidade_total` (cauda longa);
  `indice_gravidade_medio` e `pct_acidentes_fatais` mantidos (skew baixo / massa em 0);
  `StandardScaler` nas 4. Decisão do usuário: severidade agregada é **descritor de
  perfil do trecho**, não desfecho a prever — por isso entra na formação.
- *Acidentes:* `hora` em sin/cos cíclico; `veiculos` em `log1p`+MinMax; categóricas em
  one-hot com colapso de níveis raros (meteorologia → 4; causa → top-8 + Outros);
  flags `tem_*` mantidas só com prevalência ≥ ~3%. One-hot/booleanos **não** são
  padronizados (z-score explodiria dummies raras); peso de bloco (÷√nº de colunas)
  equilibra a contribuição de cada grupo conceitual.
- *Vazamento:* nenhuma variável de desfecho entra na matriz de acidentes (assert no
  pré-processamento); `indice_gravidade`, `fatal`, `classe_gravidade`, `mortos`, etc.
  são usados apenas para caracterizar os grupos.

**Limitações.** O KMeans assume geometria euclidiana e clusters convexos/isotrópicos,
o que é apenas aproximado em dados mistos com muitos one-hot (distância euclidiana em
colunas 0/1 ≈ Hamming escalado). Alternativas mais adequadas — **K-prototypes** (nativo
para misto), **Gower + Agglomerative/HDBSCAN** — ficam para iterações futuras (custo
O(n²) exige amostragem em 145k linhas). A arquitetura já isola a troca de algoritmo em
`cluster.fit_cluster(X, algo, k)`. Interpretações são associativas, não causais.
"""


def report(trechos_result: dict[str, Any], acidentes_result: dict[str, Any]) -> str:
    """Gera e grava o relatório consolidado da clusterização.

    Returns:
        O caminho do relatório gravado (como string).
    """
    cabecalho = """# Relatório — Clusterização (Aprendizado Não Supervisionado)

Etapa de descoberta de **perfis naturais** (não rankings) na malha rodoviária federal,
em dois níveis independentes: trechos rodoviários e acidentes. Base para a etapa de ML
supervisionado e para as recomendações de priorização de investimentos.
"""
    conteudo = "\n".join(
        [
            cabecalho,
            _secao_trechos(trechos_result),
            _secao_acidentes(acidentes_result),
            _secao_cotovelo(trechos_result, acidentes_result),
            _secao_silhouette(trechos_result, acidentes_result),
            _secao_descobertas(trechos_result, acidentes_result),
            _secao_metodologia(acidentes_result),
        ]
    )
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORT_FILE.write_text(conteudo, encoding="utf-8")
    logger.info("Relatório gravado em %s",
                config.REPORT_FILE.relative_to(config.PROJECT_ROOT))
    return str(config.REPORT_FILE)
