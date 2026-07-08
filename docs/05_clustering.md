# Clusterização (Aprendizado Não Supervisionado)

Etapa de descoberta de **perfis naturais** (não rankings) com K-Means, em dois níveis
independentes. O número de clusters foi escolhido por **Método do Cotovelo + Silhouette +
interpretabilidade**. Código em `src/modeling/clustering/`
(`python -m src.modeling.clustering.main`); artefatos em `outputs/clustering/` (relatório,
figuras e o Parquet com os rótulos, `trechos_clusters.parquet`).

> Não-vazamento: variáveis de desfecho (`indice_gravidade`, `fatal`, `mortos`…) **não**
> entram na formação dos clusters de acidentes — só caracterizam os grupos depois. Para
> **trechos**, a severidade agregada é um *descritor de perfil do segmento* (não o desfecho
> de um evento a prever) e por isso é usada na formação, conforme o enunciado.

## 1. Perfis de trechos rodoviários (K = 4)

Features de formação: `qtd_acidentes`, `indice_gravidade_total`, `indice_gravidade_medio`,
`pct_acidentes_fatais` (`log1p` nas duas primeiras + `StandardScaler`). O cotovelo cai de
87.557 → 52.556 (K3) → 41.744 (K4); silhouette 0,41–0,48; o **PCA captura 87% da variância
em 2D** — os grupos são bem separados. A malha se organiza em dois eixos: **volume/exposição**
× **letalidade por evento**.

| Cluster | Perfil | Trechos | Acid./trecho (méd) | Índice médio | % fatais | Mortos (tot) |
|---|---|---|---|---|---|---|
| 0 | Médio volume · Média letalidade | 12.179 (36,9%) | 3,4 | 5,9 | 9,2% | 3.899 |
| 1 | **Alto volume** · Média letalidade | 4.696 (14,2%) | 17,6 | 4,2 | 5,7% | 3.894 |
| 2 | Baixo volume · **Alta letalidade** | 2.941 (8,9%) | 1,4 | 18,5 | **87,8%** | 4.410 |
| 3 | Baixo volume · Baixa letalidade | 13.208 (40,0%) | 1,3 | 2,6 | 0,0% | 0 |

**Interpretação prática:**

1. **Cluster 2 — trechos raros, porém letais** (a principal descoberta). Apenas 8,9% dos
   trechos, baixíssima frequência, mas **87,8% dos acidentes são fatais** e somam **4.410
   mortos** — mais do que qualquer outro grupo. São pontos de alta severidade por evento que
   o ranking por volume esconde. ➜ **intervenções pontuais de engenharia/sinalização**, de
   alto retorno em vidas.
2. **Cluster 1 — corredores de alto volume** (17,6 acidentes/trecho), mas letalidade por
   evento baixa. Concentram feridos (20.343 graves) por exposição, não por gravidade
   intrínseca. ➜ **fiscalização, capacidade e fluidez**.
3. **Cluster 0 — perfil intermediário e mais comum**, severidade moderada; monitoramento.
4. **Cluster 3 — trechos benignos** (40% da malha, 0% de fatais); baixa prioridade.

Confirma o insight da EDA de que **volume ≠ gravidade**: os dois eixos exigem políticas de
investimento distintas. Figuras geradas em `outputs/clustering/figures/`: `trechos_pca.png`,
`trechos_elbow.png`, `trechos_silhouette.png`, `trechos_box_*.png`.

## 2. Perfis de acidentes — investigado e não incluído

A clusterização de acidentes foi testada, mas **não produziu perfis multivariados nítidos**
e por isso **não compõe a entrega** (fica documentada aqui como achado).

- **Sem cotovelo** (todas as features): inertia cai de forma suave/linear
  (88.781 → 82.787 → 77.421 → 73.826…), sem inflexão.
- **Silhouette ≈ 0,13–0,17** (abaixo de 0,25) e **PCA 2D explica só ~28%** da variância — os
  acidentes não se separam em grupos compactos.
- **Reduzir para features densas** (temporal + via + causa) elevou o Silhouette a ~0,27, mas
  os grupos **apenas redescobrem a `tipo_pista`**: em K=3 os clusters têm exatamente
  **70.020 / 61.512 / 14.153** registros — os tamanhos de Simples / Dupla / Múltipla. É uma
  única variável categórica dominante, não arquétipos novos.

**Conclusão.** Os acidentes variam num **continuum** estruturado sobretudo pela `tipo_pista`,
e **pista Simples concentra severidade** (índice ~5,2; ~9,9% fatais) vs Dupla (~3,7; ~4,8%) e
Múltipla (~3,5; ~4,0%) — o que **confirma a EDA**, não revela perfis inéditos. Parte do efeito
é metodológica (K-Means euclidiano é fraco em dados majoritariamente one-hot; K-prototypes
recuperaria a mesma partição). O conhecimento contexto → severidade dos acidentes é melhor
explorado pela EDA Analytics e pela etapa de ML supervisionado.
