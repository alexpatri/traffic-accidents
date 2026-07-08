# Análise Exploratória de Dados (EDA) — Camada Trusted

A primeira EDA foi realizada **exclusivamente sobre a camada Trusted**
(`data/trusted/acidentes_trusted.parquet`), com **Polars** e **Matplotlib**. O objetivo
não é responder ao problema de negócio, mas **compreender a distribuição dos dados para
orientar a Feature Engineering por evidências**.

O código fica em `src/analysis/eda/` (módulos `overview`, `categorical`, `numerical`,
`temporal`, `spatial`, `relationships`, `correlations` + `main`). A execução
(`python -m src.analysis.eda.main`) gera **apenas figuras** em `outputs/eda/figures/`; a
interpretação está documentada abaixo.

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

## 2. Principais padrões encontrados

### Variáveis categóricas

- **Gravidade fortemente desbalanceada** (`classificacao_acidente`): *Com Vítimas Feridas*
  77,1%, *Sem Vítimas* 15,7%, *Com Vítimas Fatais* 7,2%. Implicação direta para modelagem:
  a classe fatal — a mais relevante para o negócio — é minoritária.
- `tracado_via` é **multivalorado** (`Reta;Declive`): 37.199 registros contêm `;`,
  explicando a cardinalidade de 898. `Reta` domina isoladamente.
- `causa_acidente` (69 categorias) tem **cauda longa**: muitas causas raras → candidata a agrupamento.

### Variáveis numéricas

- Contagens de vítimas **altamente assimétricas à direita** (cauda longa, muitos zeros):
  `mortos` tem 92,8% de zeros (skew 15,4; máx 37), `feridos_graves` 77,4% de zeros (skew 6,9).
  `pessoas` (skew 11,5; máx 93) e `veiculos` (skew 5,4; máx 82) seguem o mesmo padrão.
- `km` é a única aproximadamente contínua (mediana 191; skew 1,0; faixa 0–1470).
- Os histogramas usam **escala log** no eixo de frequência justamente por essa cauda longa.

### Distribuição temporal

- **Sazonalidade mensal moderada**: pico em **dezembro** (13.375) e vale em **fevereiro**
  (10.567) — amplitude ~23% da média.
- **Horário crítico no fim da tarde**: pico às **18h** (10.892), seguido de 17h e 19h;
  vale na madrugada (2h–3h). Coincide com horário de retorno/rush.
- **Concentração no fim de semana**: domingo e sábado lideram; juntos somam **31,9%** dos
  acidentes (acima dos 28,6% esperados se a distribuição fosse uniforme entre os 7 dias).

### Distribuição espacial

- **UFs** com maior volume: **MG** (18.866), **SC**, **PR**, **RJ**, **RS** — concentração
  no Sul/Sudeste.
- **BRs** mais recorrentes: **BR-101** (25.792) e **BR-116** (22.499) destacam-se muito
  das demais.
- **Municípios** mais frequentes: Brasília, Guarulhos, Duque de Caxias, São José, Curitiba
  (regiões metropolitanas).

### Relações entre variáveis

- **Fase do dia × gravidade**: a proporção de acidentes **fatais** é maior em
  **Amanhecer (11,0%)** e **Plena Noite (10,1%)** do que em **Pleno dia (5,0%)** — quase o
  dobro. Indício de que baixa luminosidade se associa a maior letalidade.
- **Tipo de pista × gravidade**: pista **Simples** tem maior proporção de fatais (9,9%) que
  Dupla (4,8%) e Múltipla (4,0%).
- Condição meteorológica e sentido da via mostram variações menores na distribuição da gravidade.

### Correlações (numéricas, Pearson)

- Correlações esperadas por composição: `feridos_leves`~`feridos` (0,86),
  `pessoas`~`ilesos` (0,80), `ignorados`~`veiculos` (0,72), `pessoas`~`feridos` (0,52).
- `km`, `latitude` e `longitude` praticamente **não correlacionam** com as contagens
  (|r| < 0,08) — gravidade não é função linear da posição na rodovia.
- Não há correlação forte e *independente* de composição entre as contagens, o que sugere
  que um **índice de gravidade combinado** agregaria mais informação que as colunas isoladas.

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
