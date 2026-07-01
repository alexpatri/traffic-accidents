# Roteiro de Apresentação — Acidentes nas Rodovias Federais

**Duração alvo:** 15 minutos · **Apresentadores:** 3 · **Slides:** 14
**Navegação:** setas `←` / `→` (ou barra de espaço). O número do slide aparece no canto inferior direito.

> **Fio condutor (amarrar do início ao fim):** a pergunta do slide 3 — *"Onde e como investir para
> salvar mais vidas?"* — é aberta no começo e respondida no fecho. Repita a ideia
> **"volume ≠ gravidade"** em três momentos: contexto (slide 2), EDA (slide 9) e clusterização (slide 12).

> **Ordem narrativa:** contexto e pergunta → hipótese → **como construímos** (arquitetura → pipeline
> → dado Trusted → dado Analytics) → **o que os dados mostram** (EDA + concentração) → perfis de
> risco → conclusão. As evidências só aparecem depois de mostrarmos a base e o pipeline.

---

## Bloco 1 — Apresentador 1 · Contexto, pergunta & hipótese (slides 1–4, ~4:30)

### Slide 1 — Capa · 20s
- **Objetivo:** abrir com o tema e apresentar os três integrantes.
- **Fala:** "Nosso trabalho parte de uma pergunta simples e difícil: com recurso limitado, onde
  investir nas rodovias federais para salvar o máximo de vidas? Vamos percorrer o caminho dos dados
  brutos da PRF até uma recomendação concreta."

### Slide 2 — A dimensão do problema · 70s
- **Objetivo:** dimensionar a tragédia e mostrar que não melhora.
- **Âncoras:** ~145 mil acidentes e 12,2 mil mortes em 2024+2025; ~17 mortes por dia. Duas
  comparações: anos completos 2024×2025 (leve queda) e mesma janela jan–mai nos três anos —
  2026 = 29,8 mil acid / 2.429 mortos, o **pior** do período.
- **Fala:** "Só em 2024 e 2025 foram mais de 145 mil acidentes e 12 mil mortes — cerca de 17 por dia.
  Nos anos fechados há uma leve queda. Mas incorporamos a base de 2026 e, para comparar de forma
  justa, olhamos a **mesma janela — janeiro a maio — nos três anos**: 2026 já é o **pior** deles.
  O problema não está recuando."

### Slide 3 — A pergunta · 30s
- **Objetivo:** cravar a pergunta que o resto responde.
- **Fala:** "É daqui que parte todo o trabalho: **onde e como investir para salvar mais vidas?**
  Recurso é finito — precisamos separar onde há muito acidente de onde há muita morte."

### Slide 4 — Hipótese · 60s
- **Objetivo:** declarar os fatores que achamos governar o risco.
- **Âncoras:** os 5 fatores (pista, traçado, período, comportamento, concentração espacial).
- **Fala:** "Antes de olhar os dados, formulamos a hipótese: o risco depende do tipo de pista, do
  traçado, do período, do comportamento — e, sobretudo, se concentra no espaço. Se poucos trechos
  concentram o risco, dá para priorizar. É isso que vamos testar."
- **Transição:** "Para testar isso com rigor, precisamos de uma base confiável e um pipeline — e é o
  que o [próximo apresentador] vai mostrar."

---

## Bloco 2 — Apresentador 2 · Como construímos & evidências (slides 5–9, ~5:30)

### Slide 5 — Arquitetura lógica · 55s
- **Objetivo:** dar visão de engenharia — reprodutível e em camadas.
- **Âncoras:** Fontes CSV → Polars (ETL) → Trusted (parquet) → Feature Engineering → Analytics →
  Clusterização/Priorização. Stack: Python, Polars, scikit-learn.
- **Fala:** "O trabalho é um pipeline em camadas. As três bases da PRF passam por um ETL em **Polars**,
  viram uma camada **Trusted** em parquet, ganham novas variáveis na feature engineering e chegam à
  camada **Analytics**, que alimenta as análises. Tudo reprodutível e versionado."

### Slide 6 — Da matéria-bruta ao dado analítico · 55s
- **Objetivo:** detalhar os 4 passos e a variável-chave.
- **Âncoras:** índice de gravidade = 12·mortos + 6·graves + 2·leves; trecho = UF_BR_km.
- **Fala:** "O pipeline tem quatro passos: tratamento, exploração, engenharia de variáveis e nova
  exploração. A variável central é o **índice de gravidade**, que pesa cada vítima — 12 para morte,
  6 para grave, 2 para leve. E segmentamos cada rodovia em **trechos**. Vamos ver as duas camadas
  de dado que ele produz."

### Slide 7 — Camada Trusted (amostra) · 30s
- **Objetivo:** mostrar o dado real limpo, no nível do acidente.
- **Âncoras:** 145.685 boletins (2024–2025), 31 colunas; 1 linha = 1 acidente.
- **Fala:** "Esta é a camada Trusted, saída do ETL: cada linha é um acidente já tratado — data, local,
  tipo de via, vítimas. É a matéria-prima confiável de todo o resto."

### Slide 8 — Camada Analytics · trechos (amostra) · 30s
- **Objetivo:** mostrar o dado agregado por trecho, pronto para analisar.
- **Âncoras:** 33.024 trechos (UF_BR_km) com frequência e gravidade por segmento.
- **Fala:** "O resultado é a camada Analytics: os acidentes viram 33 mil trechos, cada um com sua
  frequência e seus índices de gravidade. É sobre esta tabela que rodam a clusterização e a
  priorização."

### Slide 9 — O risco não é aleatório (EDA) · 65s
- **Objetivo:** apresentar as evidências da EDA — agora que a base já foi apresentada.
- **Âncoras:** pico de volume às 18h vs. índice de gravidade maior de madrugada; pista simples ~10%
  fatais vs. ~5% na dupla.
- **Fala:** "Com a base pronta, a exploração confirma parte da hipótese. O volume se concentra no fim
  de tarde — mas a **gravidade média** sobe de madrugada, e a **pista simples** mata o dobro da dupla.
  Aparece de novo a frase-chave: **volume e gravidade não são a mesma coisa.**"

---

## Bloco 3 — Apresentador 3 · Perfis de risco & conclusão (slides 10–14, ~5:00)

### Slide 10 — O risco é concentrado · 50s
- **Objetivo:** entregar a evidência que valida a hipótese da concentração espacial.
- **Âncoras:** 10% dos trechos = 44,7% da gravidade; 36% dos trechos = 80%.
- **Fala:** "E a concentração espacial se confirma: **10% dos trechos concentram quase 45% de toda a
  gravidade.** A priorização deixa de ser opinião e vira aritmética."

### Slide 11 — Quantos perfis existem? · 45s
- **Objetivo:** justificar K=4 sem jargão excessivo.
- **Âncoras:** cotovelo + silhueta; escolha K=4.
- **Fala:** "Usamos KMeans para agrupar trechos parecidos. Cotovelo e silhueta apontam para poucos
  grupos; escolhemos **quatro**, que equilibram estatística e leitura de negócio."

### Slide 12 — Quatro perfis e um grupo crítico · 70s
- **Objetivo:** revelar o achado central da clusterização.
- **Âncoras:** grupo crítico = 8,9% dos trechos, 87,8% fatais, 4.410 mortos ≈ 36% do total; PCA
  87,3% da variância.
- **Fala:** "Os trechos se organizam em dois eixos. Há corredores de alto volume, a malha comum, e um
  **grupo crítico**: raro, mas letal — 8,9% dos trechos que respondem por **mais de um terço das
  mortes**. É o alvo de maior retorno em vidas."

### Slide 13 — Onde investir e o quê fazer · 70s
- **Objetivo:** traduzir tudo em recomendação por perfil de via.
- **Âncoras:** BR-101/116 (carga: ~1,5 mil mortos cada); BR-222/316/153 (letalidade + pista simples).
- **Fala:** "A resposta tem dois lados. Nos **corredores de carga** — BR-101 e BR-116 — o caminho é
  gestão de tráfego, fiscalização e capacidade. Nas **rodovias de pista simples e alta letalidade** —
  BR-316, BR-153 — o maior retorno em vidas vem de **duplicação, barreira central e iluminação**."

### Slide 14 — A resposta · 60s
- **Objetivo:** responder à pergunta do slide 3 e encerrar com a ressalva.
- **Âncoras:** 36% das mortes em 8,9% dos trechos (o anel à direita).
- **Fala:** "Voltando à pergunta que abriu a apresentação: investir onde o risco **se concentra**,
  não onde ele apenas **se acumula**. Três frentes: corredores de volume, pista simples letal e os
  trechos críticos. A ressalva honesta: sem dados de tráfego e de velocidade da via, a priorização
  fina ainda é limitada — é o nosso próximo passo. Obrigado."

---

## Ensaios e dicas de tempo

| Bloco | Slides | Tempo alvo |
|---|---|---|
| 1 — Contexto, pergunta & hipótese | 1–4 | ~4:30 |
| 2 — Construção & evidências | 5–9 | ~5:30 |
| 3 — Perfis de risco & conclusão | 10–14 | ~5:00 |
| Margem / perguntas | — | ~0:30 |

- **Não leia os slides.** Eles dão o contexto visual; a fala acrescenta a narrativa.
- **Slides de tabela (7 e 8) são rápidos:** aponte 1–2 colunas e siga; não leia a tabela.
- **Números-âncora por slide:** decore 1–2 por tela (marcados em cada seção acima).
- **Passagem de bastão:** use as frases de transição ao fim de cada bloco.
- **Se atrasar:** encurte os slides 7 e 8 (amostras de dados) — são ilustrativos.
- **Frase-marca a repetir 3×:** *"volume não é o mesmo que gravidade."*
