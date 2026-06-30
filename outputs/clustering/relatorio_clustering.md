# Relatório — Clusterização (Aprendizado Não Supervisionado)

Etapa de descoberta de **perfis naturais** (não rankings) na malha rodoviária federal. A
entrega é a **clusterização de trechos rodoviários**; a clusterização de acidentes foi
investigada e documentada como achado negativo (seção 5). Base para a etapa de ML
supervisionado e para as recomendações de priorização de investimentos.

## 1. Clusterização dos Trechos

**Dataset:** `data/analytics/trechos_analytics.parquet` (33,024 trechos).
**Features de formação:** `qtd_acidentes`, `indice_gravidade_total`, `indice_gravidade_medio`, `pct_acidentes_fatais`
(frequência, severidade total, risco médio por evento e letalidade).
**Número de clusters:** **K = 4**.

### Perfil de cada grupo

| Cluster | n | % | Qtd.acid (méd) | Índ.médio | % fatais | Mortos (tot) | Fer.graves (tot) |
|---|---|---|---|---|---|---|---|
| 0 | 12179 | 36.88 | 3.42 | 5.90 | 9.23 | 3899 | 14905 |
| 1 | 4696 | 14.22 | 17.62 | 4.21 | 5.65 | 3894 | 20343 |
| 2 | 2941 | 8.91 | 1.41 | 18.49 | 87.77 | 4410 | 2703 |
| 3 | 13208 | 40.00 | 1.30 | 2.59 | 0.00 | 0 | 2411 |

> Médias das features de formação e o **desfecho retido** (`mortos`, `feridos_graves`)
> — este último usado apenas para caracterizar os grupos, não para formá-los.

### Interpretação dos perfis

- **Cluster 0 — Médio volume · Médio letalidade**: 12179 trechos (36.9%); média 3.4 acidentes/trecho, índice médio 5.9 (médio risco por evento), 9.2% de acidentes fatais; 3899 mortos e 14905 feridos graves no total.
- **Cluster 1 — Alto volume · Médio letalidade**: 4696 trechos (14.2%); média 17.6 acidentes/trecho, índice médio 4.2 (médio risco por evento), 5.7% de acidentes fatais; 3894 mortos e 20343 feridos graves no total.
- **Cluster 2 — Baixo volume · Alto letalidade**: 2941 trechos (8.9%); média 1.4 acidentes/trecho, índice médio 18.5 (alto risco por evento), 87.8% de acidentes fatais; 4410 mortos e 2703 feridos graves no total.
- **Cluster 3 — Baixo volume · Baixo letalidade**: 13208 trechos (40.0%); média 1.3 acidentes/trecho, índice médio 2.6 (baixo risco por evento), 0.0% de acidentes fatais; 0 mortos e 2411 feridos graves no total.

![Clusters de trechos em PCA 2D](figures/trechos_pca.png)

*Projeção PCA (somente visualização): 87.3% da variância em 2D.*

### Sensibilidade (redundância qtd ↔ índice total, r≈0,89)

Repetindo a clusterização sem `indice_gravidade_total` (3 features), o Silhouette em
K=4 passou de 0.4073 (4 features) para
0.5247 (3 features) — documentado para transparência da decisão
de manter as 4 features pedidas no enunciado.

![Volume por cluster](figures/trechos_box_qtd_acidentes.png)
![Letalidade por cluster](figures/trechos_box_pct_acidentes_fatais.png)

## 2. Método do Cotovelo

A inertia (SSE intra-cluster) cai com K; busca-se o ponto de inflexão a partir do qual o
ganho marginal diminui.

| K | Inertia (SSE) | Silhouette |
|---|---|---|
| 2 | 87,557 | 0.4110 |
| 3 | 52,556 | 0.4813 |
| 4 | 41,744 | 0.4073 |
| 5 | 35,048 | 0.4097 |
| 6 | 28,951 | 0.4350 |
| 7 | 24,438 | 0.4282 |
| 8 | 20,080 | 0.4369 |
| 9 | 17,587 | 0.4296 |
| 10 | 15,472 | 0.4670 |

![Cotovelo — Trechos](figures/trechos_elbow.png)

Forte queda até K=3–4 (87.557 → 52.556 → 41.744) e desaceleração depois — inflexão na
região de K=3–4; escolhido **K=4**.

## 3. Silhouette Score

Mede coesão × separação (−1 a 1). Calculado sobre amostra fixa de
10,000 trechos (a métrica é O(n²) e inviável na base completa).

![Silhouette — Trechos](figures/trechos_silhouette.png)

**Comparação com o cotovelo + escolha de K.** O Silhouette favorece K menores (pico em
K=3 ≈ 0,48), enquanto o cotovelo admite K maiores. Optou-se por **K=4** por
**interpretabilidade**: K=4 revela a estrutura 2×2 *volume × letalidade* — incluindo o
grupo raro porém extremamente letal — a um custo mínimo de Silhouette frente a K=3. A
decisão não foi apenas visual: concilia cotovelo, Silhouette e leitura dos perfis.

## 4. Principais Descobertas

**Perfis de trechos encontrados.** A malha se separa em dois eixos — *volume/exposição*
(qtd de acidentes, índice total) e *letalidade por evento* (índice médio, % fatal):
corredores de alto volume e baixa letalidade, trechos raros porém letais, e grupos
intermediário/benigno.

**Grupo claramente mais crítico.** O **Cluster 2**
(87.8% de acidentes fatais em média, 4,410 mortos)
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

## 5. Clusterização de Acidentes — investigada e não incluída

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

## 6. Metodologia e Limitações

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
