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
