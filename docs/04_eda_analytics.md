# Segunda Análise Exploratória — Camada Analytics

Esta segunda EDA é realizada **exclusivamente sobre a camada Analytics**
(`data/analytics/acidentes_analytics.parquet` e `trechos_analytics.parquet`), com
**Polars** e **Matplotlib**. Diferentemente da primeira (que buscou compreender os dados
da Trusted), aqui o objetivo é **validar as features criadas na Feature Engineering** —
medir se agregaram informação, se o índice de gravidade discrimina severidade e quais
variáveis se associam a acidentes graves — preparando Clusterização e Modelagem.

O código fica em `src/analysis/eda_analytics/` (módulos `overview`, `severity`, `temporal`,
`spatial`, `categorical`, `relationships`, `correlation`, `trechos`, `report` + `data`,
`config`, `main`). A execução (`python -m src.analysis.eda_analytics.main`) gera **apenas
figuras** em `outputs/eda_analytics/figures/`; a interpretação está documentada abaixo.

> Todas as conclusões abaixo são **descritivas** (associações, não causalidade) e baseadas
> nos números registrados no log da execução.

## 1. Visão geral da camada Analytics

- **145.685 registros × 56 colunas**, ~45,2 MB em memória, **sem nulos** — mesma
  granularidade da Trusted (1 linha por acidente), porém **+25 features** (de 31 → 56 colunas).
- As 25 novas features distribuem-se em: **6 temporais** (`hora`, `mes`, `trimestre`,
  `dia_da_semana`, `fim_de_semana`, `turno`), **4 de gravidade** (`indice_gravidade`,
  `fatal`, `periodo_noturno`, `classe_gravidade`), **2 espaciais** (`km_faixa`, `trecho`),
  **12 flags de traçado** (`tem_*`) e **1 categórica agrupada** (`causa_acidente_agrupada`).
- Diferença essencial vs. Trusted: a Analytics adiciona **atributos derivados interpretáveis**
  (índice/classe de gravidade, recortes temporais, identificador de trecho, flags geométricas)
  e disponibiliza um **segundo dataset agregado por trecho** (33.024 linhas) inexistente na Trusted.

## 2. Avaliação do índice de gravidade

- **Distribuição muito assimétrica à direita**: média **4,42**, mediana **2**, p75 **6**,
  p95 **12**, p99 **26**, máximo **490** (acidente de vítimas múltiplas). A cauda longa é
  esperada — a maioria dos acidentes é de baixa severidade.
- **Classes desbalanceadas, porém coerentes**: Baixa **57,6%**, Média **30,7%**, Alta
  **10,1%**, Crítica **1,7%**. O índice médio cresce monotonicamente por classe
  (1,45 → 5,55 → 13,0 → 34,0), confirmando que a classe ordena severidade.
- **Separação forte da fatalidade**: o **% de acidentes fatais é 0% em Baixa e Média** e
  salta para **57,8% (Alta)** e **78,3% (Crítica)**. Isso ocorre por construção coerente —
  1 morto = 12 pontos, já dentro da faixa Alta — ou seja, **toda fatalidade cai nas duas
  classes superiores**. O índice e a classe separam bem os níveis de severidade.

## 3. Features temporais

- **Madrugada é o período mais grave**: turno `Madrugada` tem índice médio **4,83** e
  **11,7% de fatais**, contra **4,96%** (Manhã) e **5,42%** (Tarde). À noite o índice
  permanece alto (4,75; 9,3% fatais). Por hora, o pico de letalidade está entre **2h–4h**
  (~12–13% fatais), apesar do baixo volume — confirma a relevância de `turno`/`hora`/`periodo_noturno`.
- **Gravidade maior no fim de semana**: índice médio **4,74** vs **4,27** em dias úteis, e
  **8,5% vs 6,6%** de fatais. Por dia, **domingo** (4,87; 8,8% fatais) e **sábado** (4,61;
  8,1%) lideram — `fim_de_semana`/`dia_da_semana` capturam esse contraste.
- **Sazonalidade de gravidade fraca**: o índice médio mensal varia pouco (4,29–4,63), sem
  padrão sazonal marcante — `mes`/`trimestre` agregam menos sinal de **gravidade** que os
  recortes de turno e dia (embora sejam úteis para volume).

## 4. Features espaciais

- **Volume vs. gravidade são geograficamente distintos**: BR-101 e BR-116 concentram o
  **maior número de acidentes fatais** (1.323 e 1.319), mas têm índice médio **baixo**
  (4,04 e 3,95) — muitos acidentes, severidade média menor. Já rodovias do interior como
  **BR-242 (7,33)**, **BR-226 (6,79)**, **BR-251 (6,51)** e **BR-316 (5,95; 15,2% fatais)**
  têm o **maior índice médio**: menos acidentes, porém mais letais.
- **Risco concentrado em poucos trechos**: 33.024 trechos distintos, média de **4,4
  acidentes/trecho** (máx **155**). O identificador `trecho` (`UF_BR_km_faixa`) mostrou-se
  **adequado** — discrimina pontos críticos com granularidade de 1 km (detalhado em §9).

## 5. Traçado da via

- **Declive é a geometria mais grave**: índice médio **5,20** e **9,9% de fatais**, acima de
  `ponte` (4,96; 10,5%), `curva` (4,78; 8,3%) e `aclive` (4,69; 8,8%). A `reta` — 72,8% dos
  registros — fica próxima da média geral (4,41; 7,3%).
- **Menor gravidade em ambiente urbano/controlado**: `rotatoria` (3,30; 2,1% fatais),
  `viaduto` (3,55) e `tunel` (3,64) são as menos letais — associadas a menor velocidade.
- A expansão de `tracado_via` em flags **agregou informação**: há gradiente claro de
  gravidade entre características geométricas, antes ocultas num campo multivalorado.

## 6. Causa do acidente

- **Agrupamento eficaz**: `causa_acidente_agrupada` reduziu a cardinalidade de **69 → 27**
  categorias (raras < 0,5% → `Outros`), preservando interpretação — as causas mais
  relevantes permaneceram individualizadas.
- **Causas claramente mais perigosas**: `Transitar na contramão` (índice **9,77**; **28,9%
  fatais**), `Pedestre andava na pista` (8,10; **42,3% fatais**), `Ultrapassagem Indevida`
  (7,89; 17,1%) e `Entrada inopinada do pedestre` (7,23; 29,2%) destacam-se. Causas ligadas
  a **pedestres** e **contramão/ultrapassagem** dominam a letalidade — forte sinal preditivo.

## 7. Relações entre as novas features

Cruzamentos da **classe de gravidade** (% dentro de cada categoria) com condições da via,
ambiente e causa — barras 100% empilhadas, ordenadas pela fração da classe *Crítica*:

- **Turno × gravidade**: Madrugada e Noite têm maior fração de classes Alta/Crítica que
  Manhã/Tarde (consistente com §3).
- **Tipo de pista × gravidade**: pista **Simples** concentra maior proporção de classes
  graves que Dupla/Múltipla.
- **Causa × gravidade**: contramão, ultrapassagem indevida e pedestre puxam a fração de
  classes graves — coerente com §6.
- **Condição meteorológica** e **uso do solo** mostram variações menores na distribuição
  da gravidade.

## 8. Correlação (numéricas, Pearson)

- **O índice resume bem as contagens de vítimas**: `indice_gravidade` correlaciona-se com
  `mortos` (**0,74**), `feridos_graves` (**0,65**), `feridos` (0,57) e `pessoas` (0,44) —
  como esperado pela sua fórmula (`12·mortos + 6·feridos_graves + 2·feridos_leves`).
- **Redundância por composição**: `feridos_leves`~`feridos` (**0,86**) — medem quase o
  mesmo; usar ambas junto ao índice é redundante.
- **`km` é independente** (|r| ≤ 0,03 com tudo): a gravidade **não** é função linear da
  posição na rodovia — o risco vem da combinação de fatores, não do quilômetro em si.

## 9. Dataset por trecho (`trechos_analytics.parquet`)

- **33.024 trechos**; métricas muito assimétricas: `qtd_acidentes` mediana **2** (máx 155),
  `indice_gravidade_total` mediana **10** (máx 526), `indice_gravidade_medio` mediana **4**
  (máx 330), `pct_acidentes_fatais` mediana **0** (média 12,0).
- **Risco fortemente concentrado**: os **10% piores trechos acumulam 44,7%** do índice de
  gravidade total. Além disso, **39,2%** dos trechos têm um único acidente e **74,0%** não
  registram mortos — o problema é **localizado**, validando a estratégia de ranking.
- **Ranking coerente**: o topo é dominado pela **BR-101 em SC** (faixas km 204–215) e por
  pontos como **PE_101_69/70**, **MG_116_286** e **SP_116_219/222** — corredores conhecidos
  por alto volume e severidade. `indice_gravidade_total` combina frequência × gravidade de
  forma adequada para priorização.

## 10. Relatório consolidado

### Validação das features

| Feature | Cumpriu o objetivo? | Evidência |
|---|---|---|
| `indice_gravidade` | **Sim** | Monotônico por classe; correlação 0,74 com `mortos`; toda fatalidade nas classes superiores. |
| `classe_gravidade` | **Sim** | % fatal 0/0/57,8/78,3 — separa nitidamente os níveis. |
| `turno` / `periodo_noturno` | **Sim** | Madrugada 11,7% fatais vs Manhã 5,0% — discrimina gravidade. |
| `fim_de_semana` / `dia_da_semana` | **Sim** | FDS 8,5% vs útil 6,6% fatais; domingo é o pior dia. |
| `mes` / `trimestre` | **Parcial** | Pouca variação de gravidade (úteis p/ volume, fracas p/ severidade). |
| `trecho` / `km_faixa` | **Sim** | 10% dos trechos = 44,7% do risco; ranking coerente. |
| Flags `tem_*` (traçado) | **Sim** | Gradiente claro (declive 5,2 vs rotatória 3,3). |
| `causa_acidente_agrupada` | **Sim** | 69→27 sem perda; isola causas letais (contramão, pedestre). |
| `fatal` | **Sim (como alvo)** | Alvo binário limpo (7,16% positivos). |

### Principais insights

1. **Volume ≠ gravidade**: BR-101/116 lideram em nº de fatais, mas as rodovias mais
   **letais por acidente** são do interior (BR-242, BR-226, BR-316).
2. **Janela noturna e fim de semana** concentram a maior severidade.
3. **Pedestres e manobras proibidas** (contramão, ultrapassagem) são as causas mais letais.
4. **Risco espacialmente concentrado** — poucos trechos respondem por grande parte da gravidade.

### Ranking preliminar dos trechos críticos (por `indice_gravidade_total`)

| # | Trecho | UF/BR/km | Acidentes | Mortos | Índice total |
|---|---|---|---|---|---|
| 1 | MG_116_286 | MG / BR-116 / km 286 | 9 | 12 | 526 |
| 2 | SC_101_206 | SC / BR-101 / km 206 | 154 | 8 | 524 |
| 3 | SC_101_207 | SC / BR-101 / km 207 | 154 | 10 | 490 |
| 4 | PE_101_69 | PE / BR-101 / km 69 | 109 | 12 | 458 |
| 5 | SC_101_205 | SC / BR-101 / km 205 | 136 | 10 | 430 |

> Observação: o 1º colocado (MG_116_286) é puxado por **um acidente de vítimas múltiplas**
> (índice máximo 490 num só registro), enquanto os trechos de SC/PE refletem **alta
> frequência sustentada** — dois perfis distintos de risco a tratar na priorização.

### Recomendações para Clusterização

- **Clusterizar trechos** (sobre `trechos_analytics`) com: `qtd_acidentes`,
  `indice_gravidade_total`, `indice_gravidade_medio`, `pct_acidentes_fatais`,
  `mortos`, `feridos_graves`.
- **Evitar redundância**: não usar `feridos_leves` **e** `feridos` juntos (r=0,86); preferir
  o `indice_gravidade` em vez de empilhar todos os seus componentes (r 0,65–0,74).
- **Padronizar antes** (escalas muito diferentes) e considerar `log1p` nas contagens (caudas longas).
- Variáveis mais **discriminantes** observadas: turno, traçado (declive), causa agrupada, BR/trecho.

### Recomendações para Machine Learning

- **Apenas como ALVO (risco de Data Leakage)** — derivam do desfecho do acidente:
  `indice_gravidade`, `classe_gravidade`, `fatal`, `mortos`, `feridos_graves`,
  `feridos_leves`, `feridos`. **Nunca** como preditoras.
- **Preditoras candidatas** (conhecidas antes/independentes do desfecho): temporais
  (`hora`, `turno`, `fim_de_semana`, `mes`…), espaciais (`uf`, `br`, `trecho`, `km_faixa`),
  flags `tem_*`, `causa_acidente_agrupada`, `tipo_pista`, `condicao_meteorologica`,
  `uso_solo`, `veiculos`, `pessoas` (esta última com cautela — pode embutir informação do desfecho).
- **Desbalanceamento a tratar**: alvo binário `fatal` com **7,16%** positivos; classe
  `Crítica` com apenas **1,66%**. Exigirá reamostragem/pesos e métricas adequadas
  (F1, AUC-PR), não acurácia.

## 11. Limitações

- A EDA é **descritiva**: as relações gravidade × condição são **associações, não efeitos
  causais**.
- **`pct_acidentes_fatais` é ruidoso em trechos pequenos** (1 acidente ⇒ 0% ou 100%); a
  análise da taxa restringiu-se a trechos com >1 acidente.
- O **índice de gravidade depende dos pesos** escolhidos (12/6/2); distribuições e rankings
  refletem essa parametrização.
- Janela temporal curta (**2024–2025**): sazonalidade interanual não pode ser avaliada.
- Possíveis **outliers de vítimas múltiplas** dominam o topo do ranking por índice total —
  convém analisá-los à parte da frequência sustentada.
