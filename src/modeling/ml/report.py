"""Geração do relatório da etapa de ML Supervisionado em Markdown."""

from __future__ import annotations

import logging
from typing import Any

import polars as pl

from src.modeling.ml import config
from src.modeling.ml.evaluate import EvalResult

logger = logging.getLogger(__name__)


def _tabela_dist(dist: pl.DataFrame) -> str:
    linhas = "\n".join(
        f"| {r['classe_risco']} | {r['n']:,} | {r['pct']:.2f}% |"
        for r in dist.iter_rows(named=True)
    )
    return "| Classe | n | % |\n|---|---|---|\n" + linhas


def _tabela_combos(result: EvalResult) -> str:
    head = ("| Variante | Modelo | nfeat | CV F1 macro | Teste F1 macro | "
            "Bal. acc | QWK | Bin F1 | Bin recall | Bin AUC |\n"
            "|---|---|---|---|---|---|---|---|---|---|")
    best = (result.best["variante"], result.best["algo"])
    linhas = []
    for c in sorted(result.combos, key=lambda c: c.cv_f1_macro, reverse=True):
        marca = " **(entrega)**" if (c.variante, c.algo) == best else ""
        linhas.append(
            f"| {c.variante}{marca} | {c.algo} | {c.n_features} | {c.cv_f1_macro:.3f} | "
            f"{c.test_f1_macro:.3f} | {c.test_balanced_acc:.3f} | {c.test_qwk:.3f} | "
            f"{c.bin_f1:.3f} | {c.bin_recall:.3f} | {c.bin_auc:.3f} |"
        )
    return head + "\n" + "\n".join(linhas)


def _secao_alvo(result: EvalResult) -> str:
    estrategia = (
        "quantil (quartis do índice médio → 4 faixas de prioridade ~25%)"
        if config.CUT_STRATEGY == "quantil"
        else "classe_gravidade (cortes fixos do projeto)"
    )
    return f"""## 1. Definição do alvo

Classificação **ordinal** do risco do trecho em {len(config.CLASSE_LABELS)} níveis
(Baixa < Média < Alta < Crítica), discretizando `indice_gravidade_medio` (severidade
média por acidente). Foram usados apenas trechos com **≥ {config.MIN_ACIDENTES} acidentes**
(o índice médio é instável com 1–2 ocorrências).

**Estratégia de corte:** {estrategia}.
**Limites aplicados** (`indice_gravidade_medio`): {result.cortes}.

> A estratégia A (cortes do `classe_gravidade`: 2/8/20) foi testada primeiro, mas produziu
> um alvo **degenerado** — "Crítica" ≈ 0,5% e "Média" ≈ 77% — abaixo do piso de
> {config.CLASSE_RARA_MIN_PCT:.0f}%. Migramos para **cortes por quantil**, que preservam os
> 4 níveis com classes balanceadas e aprendíveis (decisão documentada no plano).

### Distribuição das classes

{_tabela_dist(result.dist)}
"""


def _secao_features() -> str:
    permitidas = (
        "tipo de pista predominante, proporção de traçado (reta/curva/declive/aclive/"
        "interseção/ponte/túnel/rotatória), % urbano, média de veículos; "
        "+ `região` (V2) ou `UF` (V3)"
    )
    return f"""## 2. Features — estruturais, sem vazamento, sem identificador

O modelo aprende com **características do trecho**, não com sua localização — para
generalizar a outras vias. Três variantes são comparadas: **fisico** (V1),
**fisico_regiao** (V2) e **fisico_uf** (V3).

- **Permitidas:** {permitidas}.
- **Proibidas (guarda anti-vazamento):** todo desfecho (`indice_gravidade*`, `mortos`,
  `feridos_*`, `fatal`, `classe_gravidade`, `qtd_acidentes*`, `pct_acidentes_fatais`) e
  identificadores (`br`, `km_faixa`, `km`, `trecho`). Um `assert` em `features.py` falha se
  qualquer uma entrar na matriz.

As features são agregadas por trecho do nível-acidente (reuso do padrão de
`agregar_por_trecho`).
"""


def _secao_resultados(result: EvalResult) -> str:
    b = result.best
    m = b["metrics"]
    return f"""## 3. Resultados (variante × modelo)

Validação **por grupo (rodovia/BR)**: holdout e CV usam `StratifiedGroupKFold`
({config.CV_FOLDS} folds) com as BRs **disjuntas** — mede a generalização a vias **não
vistas** (o objetivo do modelo), evitando o otimismo de um split aleatório que memoriza
trechos da mesma rodovia. Hiperparâmetros ajustados por `RandomizedSearchCV`
({config.TUNE_ITER} amostragens); modelos com `class_weight="balanced"` num `Pipeline`
com `StandardScaler`.

{_tabela_combos(result)}

**Modelo de entrega** (melhor algoritmo na variante transferível recomendada):
`{b['variante']}` + `{b['algo']}` — CV F1 macro
{m.cv_f1_macro:.3f}, teste F1 macro {m.test_f1_macro:.3f}, QWK {m.test_qwk:.3f},
e visão binária (Alta∪Crítica) AUC {m.bin_auc:.3f} / recall {m.bin_recall:.3f}.

> **Transferibilidade × desempenho.** Há um gradiente claro: `fisico` < `fisico_regiao` <
> `fisico_uf` — adicionar localização ajuda. Mas o ganho da **UF** sobre a **região** é
> pequeno (AUC 0.679 vs 0.669; CV F1 0.363 vs 0.360) e vem em boa parte de *decorar* estados
> específicos (dummies `uf_*` entre as features mais fortes), o que **não generaliza** a vias
> novas. A variante **`fisico_regiao`** captura quase todo o sinal regional com 18 features
> (vs 40) e **preserva a transferência** — sendo a escolha recomendada como modelo de
> entrega, alinhada ao objetivo de aplicar o modelo a outras vias. O modelo só-físico
> (`fisico`) é o mais transferível de todos, ao custo de ~0,02 de AUC.

![Comparação das variantes](figures/comparacao_variantes.png)
![Matriz de confusão (melhor modelo)](figures/matriz_confusao.png)
"""


def _secao_interpretacao(result: EvalResult) -> str:
    imp = result.best.get("importance", [])[:8]
    linhas = "\n".join(f"- `{n}`: {v:.3f}" for n, v in imp) or "- (n/d)"
    usa_uf = result.best["variante"] == "fisico_uf"
    nota_uf = (
        "\nVários dos sinais mais fortes do melhor modelo são *dummies* de **UF** "
        "(ex.: `uf_AM`, `uf_RR`, `uf_SP`) — ou seja, parte do desempenho vem de um "
        "componente **regional/de localização**, menos transferível a vias novas. Ver a "
        "nota de transferibilidade em §3.\n"
        if usa_uf else ""
    )
    return f"""## 4. Interpretação

Features mais influentes (melhor modelo):

{linhas}

![Importância das features](figures/importancia_features.png)
{nota_uf}
A leitura é **associativa, não causal**. Entre as features físicas, pesam o **tipo de
pista** (Múltipla), o **uso do solo** (`pct_urbano`) e a **geometria do traçado**
(rotatória, curva, reta) — coerente com a EDA Analytics, que já associava pista e traçado
à severidade. Não se afirma direção causal: são padrões de coincidência.
"""


def _secao_limitacoes(result: EvalResult) -> str:
    m = result.best["metrics"]
    return f"""## 5. Limitações e plano B

- **Teto de desempenho modesto** (F1 macro ≈ {m.test_f1_macro:.2f} em 4 classes; acaso =
  0,25). Esperado: a estrutura da via explica **parte** do risco — comportamento,
  velocidade, fiscalização e aleatoriedade (ausentes nas features) dominam o restante.
  Isso é o esperado de um modelo **sem vazamento**: métricas honestas, não infladas por
  decorar localizações. A visão binária (AUC ≈ {m.bin_auc:.2f}) é útil para triagem.
- **Cortes do alvo são fronteiras** sobre um contínuo ruidoso; a estratégia por quantil
  torna "Crítica" = "top ~25% mais severos", não um extremo absoluto.
- **Cobertura:** o filtro ≥ {config.MIN_ACIDENTES} acidentes exclui trechos esparsos
  (alvo ruidoso); o modelo se aplica a trechos com algum histórico ou a vias novas cujas
  características físicas sejam conhecidas.
- **Para uma predição mais precisa, faltam fatores:** o teto é limitado pelos dados. Ganhos
  reais viriam de variáveis ausentes que dirigem o risco — sobretudo **velocidade/limite da
  via**, volume de tráfego (VMD), geometria fina (raio de curva, rampa), fiscalização/radares
  e iluminação.
- **Plano B (regressão):** prever `indice_gravidade_medio` contínuo (R²/MAE) e mapear para
  os 4 níveis na apresentação evita o problema de classe rara; fica como próximo passo se
  for preciso priorizar a leitura contínua de risco.
"""


def report(result: EvalResult) -> str:
    """Gera e grava o relatório da etapa de ML. Retorna o caminho gravado."""
    cabecalho = """# Relatório — ML Supervisionado (classificação de risco de trechos) — bônus

**Etapa bônus / exploratória** — complementa a entrega central (clusterização de trechos).
Classifica trechos rodoviários em níveis de risco (Baixa/Média/Alta/Crítica) a partir de
**características estruturais** do segmento, sem vazamento de desfecho e sem
identificadores — para subsidiar a priorização de investimentos e generalizar a outras vias.

> As métricas modestas indicam que a estrutura da via explica apenas **parte** do risco:
> uma **predição mais precisa provavelmente exige fatores adicionais** ausentes da base,
> sobretudo **velocidade/limite da via**, volume de tráfego, geometria fina e fiscalização.
"""
    conteudo = "\n".join([
        cabecalho,
        _secao_alvo(result),
        _secao_features(),
        _secao_resultados(result),
        _secao_interpretacao(result),
        _secao_limitacoes(result),
    ])
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORT_FILE.write_text(conteudo, encoding="utf-8")
    logger.info("Relatório gravado em %s",
                config.REPORT_FILE.relative_to(config.PROJECT_ROOT))
    return str(config.REPORT_FILE)
