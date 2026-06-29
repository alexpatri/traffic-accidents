# Relatório — Feature Engineering (camada Analytics)

Gerado a partir de `data/trusted/acidentes_trusted.parquet`
(145685 acidentes, 56 colunas) e agregado em
33024 trechos. Todas as features são interpretáveis e independentes de
algoritmo (sem encoding, normalização ou seleção — reservados à Modelagem).

## Features criadas

### Temporais (derivadas de `data` e `horario`)

| Feature | Descrição |
|---|---|
| `hora` | Hora do dia (0–23). |
| `mes` | Mês (1–12). |
| `trimestre` | Trimestre do ano (1–4). |
| `dia_da_semana` | Dia da semana numérico (1=segunda … 7=domingo). |
| `fim_de_semana` | Booleano: sábado ou domingo. |
| `turno` | Madrugada (00–05), Manhã (06–11), Tarde (12–17), Noite (18–23). |

### Gravidade

| Feature | Descrição |
|---|---|
| `indice_gravidade` | 12·mortos + 6·feridos_graves + 2·feridos_leves. |
| `fatal` | Booleano: houve ao menos um morto. |
| `classe_gravidade` | Faixa ordinal do índice: Baixa (0–2), Média (3–8), Alta (9–20), Crítica (>20). |
| `periodo_noturno` | Booleano: `fase_dia` de baixa luminosidade (Amanhecer/Anoitecer/Plena Noite). |

### Espaciais

| Feature | Descrição |
|---|---|
| `km_faixa` | Km discretizado em faixas de 1 km. |
| `trecho` | Identificador `UF_BR_km_faixa` do segmento rodoviário. |

### Traçado da via (expansão de `tracado_via`, 12 categorias)

| Feature | Descrição |
|---|---|
| `tem_aclive` | Indica traçado do tipo *Aclive*. |
| `tem_curva` | Indica traçado do tipo *Curva*. |
| `tem_declive` | Indica traçado do tipo *Declive*. |
| `tem_desvio_temporario` | Indica traçado do tipo *Desvio Temporário*. |
| `tem_em_obras` | Indica traçado do tipo *Em Obras*. |
| `tem_intersecao_de_vias` | Indica traçado do tipo *Interseção de Vias*. |
| `tem_ponte` | Indica traçado do tipo *Ponte*. |
| `tem_reta` | Indica traçado do tipo *Reta*. |
| `tem_retorno_regulamentado` | Indica traçado do tipo *Retorno Regulamentado*. |
| `tem_rotatoria` | Indica traçado do tipo *Rotatória*. |
| `tem_tunel` | Indica traçado do tipo *Túnel*. |
| `tem_viaduto` | Indica traçado do tipo *Viaduto*. |

### Categóricas agrupadas

| Feature | Descrição |
|---|---|
| `causa_acidente_agrupada` | `causa_acidente` com categorias raras (< 0.5%) reunidas em `Outros`. Original preservada. |

## Justificativa (achados da EDA)

- **Temporais** — a EDA mostrou pico de acidentes às 17h–19h, concentração no fim de
  semana (domingo/sábado = 31,9%) e sazonalidade mensal (pico em dezembro). `hora`,
  `turno`, `dia_da_semana`, `fim_de_semana`, `mes` e `trimestre` tornam esses padrões
  diretamente analisáveis.
- **Índice de gravidade** — as contagens de vítimas têm baixa correlação independente
  entre si (EDA §correlações); um índice ponderado agrega mais sinal que cada coluna
  isolada. A classe fatal é minoritária (~7%), o que motiva a flag `fatal` e a
  `classe_gravidade` ordinal.
- **`periodo_noturno`** — a EDA observou maior letalidade em Amanhecer (11,0%) e Plena
  Noite (10,1%) frente a Pleno dia (5,0%).
- **Trecho** — a malha viária concentra acidentes em poucas BRs (BR-101, BR-116); um
  identificador espacial estável é a base para o ranking de trechos perigosos.
- **Expansão de `tracado_via`** — campo multivalorado (`Reta;Declive`) responsável pela
  cardinalidade de 898; as flags booleanas resolvem a multivaloração.
- **Agrupamento de causas raras** — `causa_acidente` tem cauda longa (69 categorias);
  agrupar as raras reduz ruído sem perder as causas relevantes.

## Impacto esperado

- **Análises estatísticas** — `turno`, `classe_gravidade` e `fim_de_semana` permitem
  recortes diretos; `indice_gravidade` resume a severidade em uma métrica contínua.
- **Clusterização** — features numéricas (`indice_gravidade`, `hora`) e booleanas
  (`tem_*`, `fim_de_semana`, `periodo_noturno`) descrevem cada acidente em um espaço
  interpretável, sem depender de encoding ainda.
- **Machine Learning** — `fatal`/`classe_gravidade` servem de alvo; as features de
  contexto (temporais, traçado, trecho) são preditores candidatos. O dataset por trecho
  habilita modelos e rankings de risco por segmento.

## Decisões tomadas

- **Pesos do índice de gravidade**: mortos = 12,
  feridos_graves = 6,
  feridos_leves = 2 (constantes em `config.py`).
- **Faixa de km do trecho**: 1 km → `trecho = UF_BR_km_faixa`
  (33024 trechos distintos). Inclui UF porque o mesmo número de BR cruza
  vários estados.
- **Limiar de causas raras**: frequência relativa < 0.5%
  → `Outros`. Aplicado apenas a `causa_acidente`; `municipio` preservado.
- **Categorias de `tracado_via` identificadas** (12): Aclive, Curva, Declive, Desvio Temporário, Em Obras, Interseção de Vias, Ponte, Reta, Retorno Regulamentado, Rotatória, Túnel, Viaduto.

### Causas agrupadas em `Outros` (43)

- Acostamento em desnível
- Acumulo de areia ou detritos sobre o pavimento
- Acumulo de óleo sobre o pavimento
- Afundamento ou ondulação no pavimento
- Ausência de sinalização
- Carga excessiva e/ou mal acondicionada
- Condutor desrespeitou a iluminação vermelha do semáforo
- Condutor usando celular
- Curva acentuada
- Declive acentuado
- Deficiência do Sistema de Iluminação/Sinalização
- Deixar de acionar o farol da motocicleta (ou similar)
- Demais Fenômenos da natureza
- Demais falhas na via
- Desvio temporário
- Estacionar ou parar em local proibido
- Faixas de trânsito com largura insuficiente
- Falta de acostamento
- Falta de elemento de contenção que evite a saída do leito carroçável
- Faróis desregulados
- Frear bruscamente
- Fumaça
- Iluminação deficiente
- Ingestão de substâncias psicoativas pelo condutor
- Modificação proibida
- Neblina
- Participar de racha
- Pedestre - Ingestão de álcool/ substâncias psicoativas
- Pista esburacada
- Problema com o freio
- Problema na suspensão
- Redutor de velocidade em desacordo
- Restrição de visibilidade em curvas horizontais
- Restrição de visibilidade em curvas verticais
- Retorno proibido
- Semáforo com defeito
- Sinalização encoberta
- Sinalização mal posicionada
- Sistema de drenagem ineficiente
- Suicídio (presumido)
- Transitar na calçada
- Transtornos Mentais (exceto suicidio)
- Área urbana sem a presença de local apropriado para a travessia de pedestres
