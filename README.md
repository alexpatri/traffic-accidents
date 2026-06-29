# ETL — Acidentes de Trânsito da PRF (Camada Trusted)

Pipeline de **ETL (Extract, Transform, Load)** dos dados abertos de acidentes da
Polícia Rodoviária Federal (datasets `datatran`). Esta etapa produz uma única camada
**Trusted** — limpa, padronizada, consistente e confiável — que servirá de fonte para
EDA, Feature Engineering, Clusterização e Machine Learning.

> Esta etapa é **exclusivamente de ETL**. Não há criação de variáveis, encoding,
> normalização, agregações ou indicadores estatísticos — isso pertence às próximas etapas.

## Download dos datasets

Os CSVs de origem não acompanham o repositório. Baixe-os para `data/raw/`:

```bash
mkdir -p data/raw

curl -L "https://drive.google.com/uc?export=download&id=14lB0vqMFkaZj8HZ44b0njYgxs9nAN8KO" -o datatran2024.zip \
  && unzip datatran2024.zip -d data/raw/ && rm -rf datatran2024.zip

curl -L "https://drive.google.com/uc?export=download&id=1-G3MdmHBt6CprDwcW99xxC4BZ2DU5ryR" -o datatran2025.zip \
  && unzip datatran2025.zip -d data/raw/ && rm -rf datatran2025.zip
```

Ao final, devem existir `data/raw/datatran2024.csv` e `data/raw/datatran2025.csv`.

## Estrutura

```text
project/
├── data/
│   ├── raw/                       # CSVs de origem (datatran2024.csv, datatran2025.csv)
│   └── trusted/                   # acidentes_trusted.parquet (gerado)
├── src/
│   ├── config.py                  # paths, schema, mapa de rename, regras de validação
│   ├── extract.py                 # leitura dos CSVs brutos
│   ├── transform.py               # limpeza, tipagem, validação, união
│   ├── load.py                    # persistência em Parquet
│   └── main.py                    # orquestração Extract → Transform → Load
├── logs/                          # logs de execução (gerado)
├── requirements.txt
└── README.md
```

## Como executar

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.main
```

A saída é gravada em `data/trusted/acidentes_trusted.parquet` (compressão `zstd`) e um
log detalhado em `logs/etl_<timestamp>.log`.

## Tecnologias

- Python 3.12+
- [Polars](https://pola.rs/) para manipulação dos dados (operações vetorizadas)
- CSV como fonte, Parquet como formato de persistência

## Decisões de tratamento (justificadas pela exploração)

A exploração dos datasets de 2024 e 2025 (73.156 e 72.529 registros, 30 colunas cada)
orientou cada decisão abaixo:

| Problema observado | Tratamento | Justificativa |
|---|---|---|
| Encoding **ISO-8859-1** | Leitura com `encoding="latin-1"` | Ler como UTF-8 corrompe acentos. |
| **Separador decimal divergente**: `latitude`/`longitude` usam `.` em 2024 e `,` em 2025; `km` usa `,` nos dois anos | Substituir `,`→`.` antes do cast para `Float64` | Única incompatibilidade real entre os anos; tratamento genérico cobre ambos. |
| Tudo lido como texto | Cast manual de tipos (`Date`, `Time`, `Float64`, `Int`) | Evita inferência divergente entre anos e dá controle sobre o separador decimal. |
| 1 `id` corrompido em notação científica (`6e+05`, 2024) | Cast `id` via `Float64` antes de `Int64` | Recupera o valor inteiro (600000, sem colisão) em vez de descartar a linha. Artefato típico de exportação por planilha. |
| Token literal `"NA"` (e vazios) | Convertidos para nulo; categóricas nulas → `"Não informado"` | Apenas 2 linhas no total; preserva o registro sem derivar valores (fora de escopo). |
| Nomes de colunas | Normalização (minúsculas, sem acento, `_`) + `data_inversa`→`data` e correção de grafia `condicao_metereologica`→`condicao_meteorologica` | Consistência e nomenclatura limpa. |
| Espaços extras em texto | `strip` + colapso de espaços internos | Categorias já estavam consistentes em caixa, então **não** se força Title Case (evita corromper `municipio`/`tracado_via`). |
| Duplicidades | Remoção de linhas idênticas; checagem de `id` | Não havia duplicatas reais; linhas com mesma chave data+hora+local têm IDs distintos (acidentes distintos) → mantidas e apenas registradas em log. |
| Violações de domínio (km negativo, contagens negativas, coordenadas fora do Brasil, datas/horários inválidos) | **Registradas em log**, sem remover | A exploração mostrou ~0 violações; manter rastreabilidade. |
| `pessoas` ≠ soma das vítimas (~3.800 linhas/ano) | Apenas registrado em log | Característica conhecida da fonte; corrigir seria arriscado e fora de escopo. |
| `condicao_meteorologica` com `Granizo` só em 2024 | Mantida | Categoria válida; ausência em 2025 não é erro. |
| `tracado_via` multivalorado (`Reta;Declive`) | Mantido como texto | Separar seria Feature Engineering. |

## Resultado

Um único arquivo Parquet com os dados de 2024 e 2025 unificados, tipados e padronizados,
pronto para consumo pelas etapas posteriores do projeto.

---

# Análise Exploratória de Dados (EDA)

A primeira EDA foi realizada **exclusivamente sobre a camada Trusted**
(`data/trusted/acidentes_trusted.parquet`), com **Polars** e **Matplotlib**. O objetivo
não é responder ao problema de negócio, mas **compreender a distribuição dos dados para
orientar a Feature Engineering por evidências**.

O código fica em `src/eda/` (módulos `overview`, `categorical`, `numerical`, `temporal`,
`spatial`, `relationships`, `correlations` + `main`). A execução gera **apenas figuras**
em `outputs/eda/figures/`; a interpretação está documentada abaixo.

```bash
.venv/bin/python -m src.eda.main
```

> **Categorias de alta cardinalidade** (`municipio`, `br`, `causa_acidente`, `tracado_via`)
> são plotadas em **top-15** por legibilidade; as estatísticas no log consideram todos os valores.

## 1. Principais características do dataset

- **145.685 registros × 31 colunas**, ~35,7 MB em memória, **sem valores nulos** (o ETL já
  tratou; restam apenas 2 registros `"Não informado"` em `classificacao_acidente`).
- **Equilíbrio entre anos**: 2024 = 73.156 e 2025 = 72.529 (≈50,2% / 49,8%) — base
  temporalmente balanceada, sem viés de volume entre os anos.
- **Cardinalidade** (orienta o tratamento futuro de cada variável):
  - Baixa: `uso_solo` (2), `sentido_via` (3), `tipo_pista` (3), `classificacao_acidente` (4),
    `fase_dia` (4), `dia_semana` (7).
  - Média: `condicao_meteorologica` (10), `tipo_acidente` (17), `uf` (27), `causa_acidente` (69), `br` (118).
  - Alta: `delegacia` (154), `uop` (399), `tracado_via` (898), `municipio` (1942).

![Acidentes por ano](outputs/eda/figures/01_acidentes_por_ano.png)
![Cardinalidade das categóricas](outputs/eda/figures/02_cardinalidade_categoricas.png)

## 2. Principais padrões encontrados

### Variáveis categóricas

- **Gravidade fortemente desbalanceada** (`classificacao_acidente`): *Com Vítimas Feridas*
  77,1%, *Sem Vítimas* 15,7%, *Com Vítimas Fatais* 7,2%. Implicação direta para modelagem:
  a classe fatal — a mais relevante para o negócio — é minoritária.
- `tracado_via` é **multivalorado** (`Reta;Declive`): 37.199 registros contêm `;`,
  explicando a cardinalidade de 898. `Reta` domina isoladamente.
- `causa_acidente` (69 categorias) tem **cauda longa**: muitas causas raras → candidata a agrupamento.

![Classificação do acidente](outputs/eda/figures/03_cat_classificacao_acidente.png)
![Tipo de acidente](outputs/eda/figures/03_cat_tipo_acidente.png)
![Causa do acidente (top 15)](outputs/eda/figures/03_cat_causa_acidente.png)
![Condição meteorológica](outputs/eda/figures/03_cat_condicao_meteorologica.png)
![Traçado da via (top 15)](outputs/eda/figures/03_cat_tracado_via.png)

### Variáveis numéricas

- Contagens de vítimas **altamente assimétricas à direita** (cauda longa, muitos zeros):
  `mortos` tem 92,8% de zeros (skew 15,4; máx 37), `feridos_graves` 77,4% de zeros (skew 6,9).
  `pessoas` (skew 11,5; máx 93) e `veiculos` (skew 5,4; máx 82) seguem o mesmo padrão.
- `km` é a única aproximadamente contínua (mediana 191; skew 1,0; faixa 0–1470).
- Os histogramas usam **escala log** no eixo de frequência justamente por essa cauda longa.

![Histograma km](outputs/eda/figures/04_hist_km.png)
![Histograma mortos](outputs/eda/figures/04_hist_mortos.png)
![Boxplots das contagens](outputs/eda/figures/04_boxplot_contagens.png)

### Distribuição temporal

- **Sazonalidade mensal moderada**: pico em **dezembro** (13.375) e vale em **fevereiro**
  (10.567) — amplitude ~23% da média.
- **Horário crítico no fim da tarde**: pico às **18h** (10.892), seguido de 17h e 19h;
  vale na madrugada (2h–3h). Coincide com horário de retorno/rush.
- **Concentração no fim de semana**: domingo e sábado lideram; juntos somam **31,9%** dos
  acidentes (acima dos 28,6% esperados se a distribuição fosse uniforme entre os 7 dias).

![Acidentes por mês](outputs/eda/figures/05_por_mes.png)
![Acidentes por hora](outputs/eda/figures/05_por_hora.png)
![Acidentes por dia da semana](outputs/eda/figures/05_por_dia_semana.png)

### Distribuição espacial

- **UFs** com maior volume: **MG** (18.866), **SC**, **PR**, **RJ**, **RS** — concentração
  no Sul/Sudeste.
- **BRs** mais recorrentes: **BR-101** (25.792) e **BR-116** (22.499) destacam-se muito
  das demais.
- **Municípios** mais frequentes: Brasília, Guarulhos, Duque de Caxias, São José, Curitiba
  (regiões metropolitanas).

![Acidentes por UF](outputs/eda/figures/06_por_uf.png)
![Acidentes por BR (top 15)](outputs/eda/figures/06_por_br.png)
![Acidentes por município (top 15)](outputs/eda/figures/06_por_municipio.png)

### Relações entre variáveis

- **Fase do dia × gravidade**: a proporção de acidentes **fatais** é maior em
  **Amanhecer (11,0%)** e **Plena Noite (10,1%)** do que em **Pleno dia (5,0%)** — quase o
  dobro. Indício de que baixa luminosidade se associa a maior letalidade.
- **Tipo de pista × gravidade**: pista **Simples** tem maior proporção de fatais (9,9%) que
  Dupla (4,8%) e Múltipla (4,0%).
- Condição meteorológica e sentido da via mostram variações menores na distribuição da gravidade.

![Gravidade por fase do dia](outputs/eda/figures/07_gravidade_por_fase_dia.png)
![Gravidade por tipo de pista](outputs/eda/figures/07_gravidade_por_tipo_pista.png)
![Gravidade por condição meteorológica](outputs/eda/figures/07_gravidade_por_condicao_meteorologica.png)

### Correlações (numéricas, Pearson)

- Correlações esperadas por composição: `feridos_leves`~`feridos` (0,86),
  `pessoas`~`ilesos` (0,80), `ignorados`~`veiculos` (0,72), `pessoas`~`feridos` (0,52).
- `km`, `latitude` e `longitude` praticamente **não correlacionam** com as contagens
  (|r| < 0,08) — gravidade não é função linear da posição na rodovia.
- Não há correlação forte e *independente* de composição entre as contagens, o que sugere
  que um **índice de gravidade combinado** agregaria mais informação que as colunas isoladas.

![Matriz de correlação](outputs/eda/figures/08_correlacao_numerica.png)

## 3. Possíveis problemas observados

- **Desbalanceamento de classe** em `classificacao_acidente` (fatais ~7%): exigirá
  estratégia específica na modelagem (reamostragem, pesos, métricas adequadas).
- **Cauda longa e outliers** nas contagens de vítimas/veículos: podem distorcer modelos
  sensíveis a escala.
- **Alta cardinalidade** em `municipio`/`uop`/`tracado_via`: inviável usar diretamente como
  categórica em muitos modelos.
- **`tracado_via` multivalorado**: mistura conceitos (`Reta`, `Curva`, `Declive`…) em um
  único campo.
- **`causa_acidente` com categorias raras**: ruído potencial se não agrupadas.
- Resíduo de `"Não informado"` em `classificacao_acidente` (2 registros) e o quirk herdado
  do ETL (`pessoas` ≠ soma das vítimas em ~3.800 linhas/ano).

## 4. Oportunidades de Feature Engineering (priorizadas)

> Identificadas durante a EDA. **Não implementadas nesta etapa.**

1. **Identificador de trecho `BR + km`** — base para o objetivo central (trechos perigosos);
   permitirá agregações espaciais na fase seguinte. *(alta prioridade)*
2. **Atributos temporais** a partir de `data`/`horario`: `mes`, `hora`, `turno`,
   `fim_de_semana` (booleano) — sustentados pelos padrões de hora-pico (18h), sazonalidade
   (dezembro) e concentração de fim de semana. *(alta)*
3. **Índice/flag de gravidade** combinando `mortos`/`feridos_graves`/`feridos_leves` —
   justificado pelo desbalanceamento e pela baixa correlação independente entre as contagens. *(alta)*
4. **Separação de `tracado_via`** em flags booleanas (`tem_curva`, `tem_declive`, …) —
   resolve a multivaloração e a cardinalidade de 898. *(média)*
5. **Agrupamento de categorias raras** em `causa_acidente` (e possivelmente `municipio`) —
   reduz ruído e cardinalidade. *(média)*
6. **Flags de condição** (`periodo_noturno` de `fase_dia`, `com_chuva` de
   `condicao_meteorologica`) — motivadas pela maior letalidade noturna observada. *(média)*
7. **Transformação de variáveis assimétricas** (ex.: `log1p` em contagens) para modelos
   sensíveis a escala. *(baixa/condicional ao modelo)*

## 5. Limitações

- Janela temporal curta (**apenas 2024–2025**): sazonalidade interanual não pode ser avaliada.
- Inconsistência herdada (`pessoas` ≠ soma das vítimas em parte dos registros) não foi
  corrigida no ETL — exige cautela ao usar `pessoas` como total.
- A EDA é **descritiva**: correlações não implicam causalidade; relações gravidade×condição
  são associações, não efeitos causais.
- Nenhuma feature foi criada; as derivações temporais usadas aqui são temporárias e **não
  foram persistidas**.

---

# Feature Engineering (camada Analytics)

Com base **direta** nas evidências da EDA, esta etapa enriquece a camada Trusted e produz
a camada **Analytics**: atributos interpretáveis e **independentes de algoritmo**. Encoding,
normalização, padronização, seleção de atributos e treino de modelo ficam para a **Modelagem**.

O código fica em `src/feature_engineering/` (módulos `config`, `temporal`, `severity`,
`spatial`, `categorical`, `aggregation`, `pipeline`, `report` + `main`), em **Polars eager**
e vetorizado, sem Pandas. A execução gera dois Parquet e um relatório:

```bash
.venv/bin/python -m src.feature_engineering.main
```

```text
data/trusted/acidentes_trusted.parquet
        │
        ▼  Feature Engineering
        ├── data/analytics/acidentes_analytics.parquet   (1 linha por acidente)
        ├── data/analytics/trechos_analytics.parquet     (1 linha por trecho)
        └── outputs/feature_engineering/relatorio_feature_engineering.md
```

## Features criadas (e a evidência da EDA que as justifica)

| Grupo | Features | Justificativa (EDA) |
|---|---|---|
| **Temporais** (de `data`/`horario`) | `hora`, `mes`, `trimestre`, `dia_da_semana` (1–7), `fim_de_semana` (bool), `turno` (Madrugada/Manhã/Tarde/Noite) | Pico às 17h–19h, concentração em domingo/sábado (31,9%), sazonalidade (dezembro). |
| **Gravidade** | `indice_gravidade`, `fatal` (bool), `classe_gravidade` (Baixa/Média/Alta/Crítica), `periodo_noturno` (bool) | Baixa correlação independente entre as contagens → índice combinado agrega mais sinal; maior letalidade noturna (Amanhecer 11,0% / Plena Noite 10,1% vs Pleno dia 5,0%). |
| **Espaciais** | `km_faixa`, `trecho` (`UF_BR_km_faixa`) | Concentração em poucas BRs (101, 116); base do ranking de trechos perigosos. |
| **Traçado** (expansão de `tracado_via`) | 12 flags `tem_*` (`tem_curva`, `tem_declive`, …) | Campo multivalorado (`Reta;Declive`) com cardinalidade 898. |
| **Categóricas agrupadas** | `causa_acidente_agrupada` (raras → `Outros`) | `causa_acidente` com cauda longa (69 categorias). |

Resultado: **acidentes_analytics** com 145.685 linhas × **56 colunas** (25 novas);
**trechos_analytics** com **33.024 trechos** e métricas agregadas (`qtd_acidentes`,
`mortos`/`feridos_graves`/`feridos_leves`, `indice_gravidade_medio`/`_maximo`/`_total`,
`qtd_acidentes_fatais`, `pct_acidentes_fatais`), ordenadas por gravidade total.

## Decisões (parametrizáveis em `src/feature_engineering/config.py`)

- **Pesos do índice de gravidade**: `12·mortos + 6·feridos_graves + 2·feridos_leves`.
- **Classe de gravidade**: 0–2 Baixa · 3–8 Média · 9–20 Alta · >20 Crítica.
- **Trecho**: `UF_BR_floor(km)` em faixas de **1 km** — inclui UF porque o mesmo número de
  BR cruza vários estados; granular o bastante para o ranking.
- **Causas raras**: agrupadas em `Outros` quando freq. relativa **< 0,5%** (43 de 69
  categorias). Aplicado **só** a `causa_acidente`; `municipio` preservado para granularidade
  espacial.
- **Expansão de `tracado_via`**: categorias descobertas automaticamente dos dados (12);
  pertinência por correspondência exata (evita falso positivo `Aclive`/`Declive`).

> O relatório completo (descrição de cada feature, impacto esperado e a lista das 43 causas
> agrupadas) é gerado em `outputs/feature_engineering/relatorio_feature_engineering.md`.

## Estrutura atualizada

```text
project/
├── data/
│   ├── raw/                       # CSVs de origem
│   ├── trusted/                   # acidentes_trusted.parquet
│   └── analytics/                 # acidentes_analytics.parquet, trechos_analytics.parquet (gerado)
├── src/
│   ├── config.py, extract.py, transform.py, load.py, main.py   # ETL
│   ├── eda/                       # 1ª análise exploratória (camada Trusted)
│   ├── feature_engineering/       # config, temporal, severity, spatial, categorical,
│   │                              # aggregation, pipeline, report, main
│   └── eda_analytics/             # 2ª análise exploratória (camada Analytics)
├── outputs/
│   ├── eda/figures/               # figuras da 1ª EDA
│   ├── eda_analytics/figures/     # figuras da 2ª EDA (gerado)
│   └── feature_engineering/       # relatorio_feature_engineering.md (gerado)
└── ...
```

---

# Segunda Análise Exploratória — Camada Analytics (EDA Analytics)

Esta segunda EDA é realizada **exclusivamente sobre a camada Analytics**
(`data/analytics/acidentes_analytics.parquet` e `trechos_analytics.parquet`), com
**Polars** e **Matplotlib**. Diferentemente da primeira (que buscou compreender os dados
da Trusted), aqui o objetivo é **validar as features criadas na Feature Engineering** —
medir se agregaram informação, se o índice de gravidade discrimina severidade e quais
variáveis se associam a acidentes graves — preparando Clusterização e Modelagem.

O código fica em `src/eda_analytics/` (módulos `overview`, `severity`, `temporal`,
`spatial`, `categorical`, `relationships`, `correlation`, `trechos`, `report` + `data`,
`config`, `main`). A execução gera **apenas figuras** em `outputs/eda_analytics/figures/`;
a interpretação está documentada abaixo.

```bash
.venv/bin/python -m src.eda_analytics.main
```

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

![Colunas Trusted vs Analytics](outputs/eda_analytics/figures/01_colunas_trusted_vs_analytics.png)

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

![Histograma do índice](outputs/eda_analytics/figures/02_hist_indice.png)
![Boxplot do índice](outputs/eda_analytics/figures/02_boxplot_indice.png)
![Distribuição das classes](outputs/eda_analytics/figures/02_classe_gravidade.png)
![% fatal por classe](outputs/eda_analytics/figures/02_pct_fatal_por_classe.png)

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

![Índice por hora](outputs/eda_analytics/figures/03_indice_por_hora.png)
![Gravidade por turno](outputs/eda_analytics/figures/03_gravidade_por_turno.png)
![Índice por dia da semana](outputs/eda_analytics/figures/03_indice_por_dia_semana.png)
![Dia útil vs fim de semana](outputs/eda_analytics/figures/03_util_vs_fds.png)

## 4. Features espaciais

- **Volume vs. gravidade são geograficamente distintos**: BR-101 e BR-116 concentram o
  **maior número de acidentes fatais** (1.323 e 1.319), mas têm índice médio **baixo**
  (4,04 e 3,95) — muitos acidentes, severidade média menor. Já rodovias do interior como
  **BR-242 (7,33)**, **BR-226 (6,79)**, **BR-251 (6,51)** e **BR-316 (5,95; 15,2% fatais)**
  têm o **maior índice médio**: menos acidentes, porém mais letais.
- **Risco concentrado em poucos trechos**: 33.024 trechos distintos, média de **4,4
  acidentes/trecho** (máx **155**). O identificador `trecho` (`UF_BR_km_faixa`) mostrou-se
  **adequado** — discrimina pontos críticos com granularidade de 1 km (detalhado em §9).

![Acidentes por trecho](outputs/eda_analytics/figures/04_acidentes_por_trecho.png)
![Índice médio por BR](outputs/eda_analytics/figures/04_indice_por_br.png)
![Acidentes fatais por BR](outputs/eda_analytics/figures/04_fatais_por_br.png)

## 5. Traçado da via

- **Declive é a geometria mais grave**: índice médio **5,20** e **9,9% de fatais**, acima de
  `ponte` (4,96; 10,5%), `curva` (4,78; 8,3%) e `aclive` (4,69; 8,8%). A `reta` — 72,8% dos
  registros — fica próxima da média geral (4,41; 7,3%).
- **Menor gravidade em ambiente urbano/controlado**: `rotatoria` (3,30; 2,1% fatais),
  `viaduto` (3,55) e `tunel` (3,64) são as menos letais — associadas a menor velocidade.
- A expansão de `tracado_via` em flags **agregou informação**: há gradiente claro de
  gravidade entre características geométricas, antes ocultas num campo multivalorado.

![Índice por traçado](outputs/eda_analytics/figures/05_indice_por_tracado.png)
![% fatal por traçado](outputs/eda_analytics/figures/05_pct_fatal_por_tracado.png)

## 6. Causa do acidente

- **Agrupamento eficaz**: `causa_acidente_agrupada` reduziu a cardinalidade de **69 → 27**
  categorias (raras < 0,5% → `Outros`), preservando interpretação — as causas mais
  relevantes permaneceram individualizadas.
- **Causas claramente mais perigosas**: `Transitar na contramão` (índice **9,77**; **28,9%
  fatais**), `Pedestre andava na pista` (8,10; **42,3% fatais**), `Ultrapassagem Indevida`
  (7,89; 17,1%) e `Entrada inopinada do pedestre` (7,23; 29,2%) destacam-se. Causas ligadas
  a **pedestres** e **contramão/ultrapassagem** dominam a letalidade — forte sinal preditivo.

![Frequência por causa](outputs/eda_analytics/figures/06_freq_causa.png)
![Índice por causa](outputs/eda_analytics/figures/06_indice_causa.png)

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

![Gravidade por turno](outputs/eda_analytics/figures/07_gravidade_por_turno.png)
![Gravidade por tipo de pista](outputs/eda_analytics/figures/07_gravidade_por_tipo_pista.png)
![Gravidade por condição meteorológica](outputs/eda_analytics/figures/07_gravidade_por_condicao_meteorologica.png)
![Gravidade por uso do solo](outputs/eda_analytics/figures/07_gravidade_por_uso_solo.png)
![Gravidade por causa agrupada](outputs/eda_analytics/figures/07_gravidade_por_causa_acidente_agrupada.png)

## 8. Correlação (numéricas, Pearson)

- **O índice resume bem as contagens de vítimas**: `indice_gravidade` correlaciona-se com
  `mortos` (**0,74**), `feridos_graves` (**0,65**), `feridos` (0,57) e `pessoas` (0,44) —
  como esperado pela sua fórmula (`12·mortos + 6·feridos_graves + 2·feridos_leves`).
- **Redundância por composição**: `feridos_leves`~`feridos` (**0,86**) — medem quase o
  mesmo; usar ambas junto ao índice é redundante.
- **`km` é independente** (|r| ≤ 0,03 com tudo): a gravidade **não** é função linear da
  posição na rodovia — o risco vem da combinação de fatores, não do quilômetro em si.

![Matriz de correlação](outputs/eda_analytics/figures/08_correlacao_numerica.png)

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

![Histograma acidentes/trecho](outputs/eda_analytics/figures/09_hist_qtd_acidentes.png)
![Histograma índice total](outputs/eda_analytics/figures/09_hist_indice_total.png)
![Concentração do risco](outputs/eda_analytics/figures/09_concentracao_risco.png)
![Top 20 trechos](outputs/eda_analytics/figures/09_top_trechos.png)

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
