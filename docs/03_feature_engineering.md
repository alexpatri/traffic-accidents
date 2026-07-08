# Feature Engineering — Camada Analytics

Com base **direta** nas evidências da EDA, esta etapa enriquece a camada Trusted e produz
a camada **Analytics**: atributos interpretáveis e **independentes de algoritmo**. Encoding,
normalização, padronização, seleção de atributos e treino de modelo ficam para a **Modelagem**.

O código fica em `src/modeling/feature_engineering/` (módulos `config`, `temporal`, `severity`,
`spatial`, `categorical`, `aggregation`, `pipeline`, `report` + `main`), em **Polars eager**
e vetorizado, sem Pandas. A execução (`python -m src.modeling.feature_engineering.main`) gera
dois Parquet e um relatório:

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

## Decisões (parametrizáveis em `src/modeling/feature_engineering/config.py`)

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
