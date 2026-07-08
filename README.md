# Acidentes nas Rodovias Federais — Perfis de Risco e Priorização

Pipeline de dados **reprodutível e de ponta a ponta** sobre os dados abertos de acidentes da
Polícia Rodoviária Federal (datasets `datatran`, 2024–2025). Partindo dos CSVs brutos, o
projeto avança camada a camada — limpeza (**Trusted**), enriquecimento (**Analytics**),
descoberta de **perfis de risco** e **priorização de investimentos** — para responder a uma
pergunta prática: *onde e como investir para reduzir mortes nas rodovias federais?*

A espinha dorsal do projeto é um achado que se repete em todas as etapas: **volume ≠
gravidade**. As rodovias com mais acidentes não são as mais letais por acidente, e o risco
está **espacialmente concentrado** em poucos trechos — o que muda completamente a estratégia
de investimento. O pipeline encadeia sete etapas, cada uma consumindo a saída da anterior:

1. **ETL** — lê os CSVs brutos (ISO-8859-1, separadores decimais divergentes entre anos),
   limpa, tipa e unifica 2024+2025 numa única camada **Trusted** (145.685 registros).
2. **EDA** — análise exploratória da Trusted para orientar, por evidências, quais features criar.
3. **Feature Engineering** — enriquece a Trusted e gera a camada **Analytics**: índice de
   gravidade, recortes temporais (turno, fim de semana), identificador de **trecho**
   (`UF_BR_km`), flags de traçado e um segundo dataset **agregado por trecho** (33.024 trechos).
4. **EDA Analytics** — valida se as features criadas realmente agregam informação e discriminam severidade.
5. **Clusterização** — descobre **perfis naturais de trechos** (K-Means, K=4), isolando um
   grupo raro porém altamente letal — a principal entrega analítica.
6. **ML supervisionado (bônus)** — demonstra a viabilidade de prever o nível de risco de um
   trecho a partir de características físicas da via, sem vazamento e validado por transferência.
7. **Priorização** — sintetiza tudo numa lista acionável de vias e trechos prioritários, com
   a intervenção sugerida por perfil.

A narrativa analítica completa de cada etapa (números, tabelas e decisões) está em
[`docs/`](docs/). Há ainda uma **apresentação de slides** autocontida em
[`apresentacao/index.html`](apresentacao/index.html) (basta abrir no navegador).

## Estrutura do repositório

```text
traffic-accidents/
├── src/
│   ├── config.py                  # configuração compartilhada (paths do projeto)
│   ├── etl/                       # [1] extract, transform, load, main → camada Trusted
│   ├── analysis/
│   │   ├── eda/                   # [2] EDA da camada Trusted
│   │   └── eda_analytics/         # [4] EDA da camada Analytics
│   └── modeling/
│       ├── feature_engineering/   # [3] Trusted → Analytics (acidentes + trechos)
│       ├── clustering/            # [5] perfis de trechos (K-Means)
│       ├── ml/                    # [6] classificação de risco de trechos (bônus)
│       └── priorizacao/           # [7] síntese e lista priorizada
├── data/                          # gerado/baixado (ignorado no git)
│   ├── raw/                       # CSVs de origem (datatran2024.csv, datatran2025.csv)
│   ├── trusted/                   # acidentes_trusted.parquet
│   └── analytics/                 # acidentes_analytics.parquet, trechos_analytics.parquet
├── outputs/                       # figuras, relatórios, modelos, predições (gerado, ignorado)
├── docs/                          # análise detalhada por etapa (01_etl … 07_priorizacao)
├── apresentacao/index.html        # deck de slides autocontido
├── requirements.txt
└── README.md
```

- **Camadas de dados**: `raw` (CSV bruto) → `trusted` (limpo/tipado) → `analytics` (enriquecido).
  Os diretórios `data/`, `outputs/` e `logs/` são gerados em runtime e **não** são versionados.
- **Tecnologias**: Python 3.12+, [Polars](https://pola.rs/) (manipulação vetorizada),
  Matplotlib (figuras) e scikit-learn (clusterização e ML). Parquet (`zstd`) como formato de persistência.

## Execução completa

### 1. Pré-requisitos e ambiente

Requer **Python 3.12+**. Crie o ambiente virtual e instale as dependências:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download dos datasets

Os CSVs de origem não acompanham o repositório. Baixe-os para `data/raw/`:

```bash
mkdir -p data/raw

curl -L "https://drive.google.com/uc?export=download&id=14lB0vqMFkaZj8HZ44b0njYgxs9nAN8KO" -o datatran2024.zip \
  && unzip datatran2024.zip -d data/raw/ && rm -rf datatran2024.zip

curl -L "https://drive.google.com/uc?export=download&id=1-G3MdmHBt6CprDwcW99xxC4BZ2DU5ryR" -o datatran2025.zip \
  && unzip datatran2025.zip -d data/raw/ && rm -rf datatran2025.zip
```

Ao final, devem existir `data/raw/datatran2024.csv` e `data/raw/datatran2025.csv`.

### 3. Rodar o pipeline

Cada etapa é um módulo executável. Execute-as na ordem abaixo (com o `.venv` ativo):

```bash
python -m src.etl.main                            # [1] ETL      → data/trusted/
python -m src.analysis.eda.main                   # [2] EDA (opcional)
python -m src.modeling.feature_engineering.main   # [3] Feature Engineering → data/analytics/
python -m src.analysis.eda_analytics.main         # [4] EDA Analytics (opcional)
python -m src.modeling.clustering.main            # [5] Clusterização → outputs/clustering/
python -m src.modeling.ml.main                    # [6] ML (bônus, opcional)
python -m src.modeling.priorizacao.main           # [7] Priorização → outputs/priorizacao/
```

**Dependências entre etapas.** A cadeia obrigatória é **[1] → [3] → [5] → [7]**. As etapas
**[2] EDA**, **[4] EDA Analytics** e **[6] ML** são ramos **opcionais** (geram figuras/relatórios
de análise, mas não alimentam etapas seguintes) e podem ser puladas. A Priorização **[7]** é a
única que consome a saída da Clusterização (`outputs/clustering/trechos_clusters.parquet`),
portanto exige que **[5]** já tenha rodado. Cada etapa valida a existência dos seus arquivos de
entrada e orienta qual etapa anterior rodar caso faltem.

### Onde cada etapa grava

| Etapa | Comando | Principais saídas |
|---|---|---|
| ETL | `python -m src.etl.main` | `data/trusted/acidentes_trusted.parquet`, `logs/` |
| EDA | `python -m src.analysis.eda.main` | `outputs/eda/figures/` |
| Feature Engineering | `python -m src.modeling.feature_engineering.main` | `data/analytics/*.parquet`, `outputs/feature_engineering/relatorio_*.md` |
| EDA Analytics | `python -m src.analysis.eda_analytics.main` | `outputs/eda_analytics/figures/` |
| Clusterização | `python -m src.modeling.clustering.main` | `outputs/clustering/` (rótulos, relatório, figuras) |
| ML | `python -m src.modeling.ml.main` | `outputs/ml/` (modelo `.joblib`, predições, relatório, figuras) |
| Priorização | `python -m src.modeling.priorizacao.main` | `outputs/priorizacao/` (lista priorizada, relatório, figuras) |

## Resumo dos resultados

- **Volume ≠ gravidade.** As BRs com mais acidentes (**BR-101** e **BR-116**, ~1.500 mortos
  cada) são corredores urbanizados de alta exposição, mas **baixa letalidade por acidente**.
  Já as vias mais **letais por evento** são do interior (BR-242, BR-226, BR-316) — menos
  acidentes, porém mais graves. São dois problemas distintos que pedem políticas distintas.
- **Risco espacialmente concentrado.** Dos 33.024 trechos, os **10% piores acumulam 44,7%**
  da gravidade total, e a clusterização isolou um grupo crítico: **8,9% dos trechos concentram
  ~4.410 mortos (≈36% do total)**, com baixíssima frequência e **87,8% de acidentes fatais**.
  É o alvo de maior retorno em vidas para intervenções pontuais de engenharia.
- **Quando e por que morrem mais.** A severidade é maior na **madrugada** (11,7% de fatais vs
  5% pela manhã) e nos **fins de semana**; **pedestres** e **manobras proibidas** (contramão,
  ultrapassagem) são as causas mais letais; **declive** é a geometria mais grave.
- **ML como prova de viabilidade (bônus).** Prever o nível de risco de um trecho só pelas
  características físicas da via atinge desempenho **modesto e honesto** (F1 macro ≈ 0,36;
  AUC ≈ 0,65 na triagem alto-risco), **sem vazamento** e **validado em rodovias não vistas**.
  A estrutura da via explica *parte* do risco; ganhos maiores exigiriam dados hoje ausentes —
  sobretudo **velocidade/limite da via** e **volume de tráfego (VMD)**.
- **Priorização acionável.** A síntese final entrega uma lista de vias e trechos prioritários
  com a intervenção sugerida por perfil (duplicação + barreira para pista simples; iluminação
  e travessias para risco noturno/pedestre; radar e fiscalização para corredores de volume).

> O detalhamento de cada etapa — com todas as tabelas e decisões — está em [`docs/`](docs/).
> Ao rodar o pipeline, os relatórios gerados automaticamente ficam em `outputs/*/relatorio_*.md`.
