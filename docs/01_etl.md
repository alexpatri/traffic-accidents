# ETL — Camada Trusted

Pipeline de **ETL (Extract, Transform, Load)** dos dados abertos de acidentes da Polícia
Rodoviária Federal (datasets `datatran`). Esta etapa produz uma única camada **Trusted** —
limpa, padronizada, consistente e confiável — que serve de fonte para EDA, Feature
Engineering, Clusterização e Machine Learning.

> Esta etapa é **exclusivamente de ETL**. Não há criação de variáveis, encoding,
> normalização, agregações ou indicadores estatísticos — isso pertence às próximas etapas.

Código em `src/etl/` (`config.py` na raiz de `src/`, `extract.py`, `transform.py`,
`load.py`, `main.py`). A saída é gravada em `data/trusted/acidentes_trusted.parquet`
(compressão `zstd`) e um log detalhado em `logs/etl_<timestamp>.log`.

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

Um único arquivo Parquet com os dados de 2024 e 2025 unificados, tipados e padronizados
(**145.685 registros × 31 colunas**), pronto para consumo pelas etapas posteriores do projeto.
