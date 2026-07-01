# Roteiro de Apresentação — Acidentes nas Rodovias Federais

**Duração alvo:** 15 minutos · **Apresentadores:** 3 · **Slides:** 12
**Navegação:** setas `←` / `→` (ou barra de espaço). O número do slide aparece no canto inferior direito.

> **Fio condutor (amarrar do início ao fim):** a pergunta do slide 3 — *"Onde e como investir para
> salvar mais vidas?"* — é aberta no começo e respondida no fecho. Repita a ideia
> **"volume ≠ gravidade"** em três momentos: contexto (slide 2), EDA (slide 7) e clusterização (slide 10).

> **Ordem narrativa:** contexto e pergunta → hipótese → **como construímos** (arquitetura + pipeline)
> → **o que os dados mostram** (EDA + concentração) → perfis de risco → conclusão. As evidências só
> aparecem depois de apresentarmos a base e o pipeline.

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

## Bloco 2 — Apresentador 2 · Como construímos & evidências (slides 5–8, ~5:00)

### Slide 5 — Arquitetura lógica · 60s
- **Objetivo:** dar visão de engenharia — reprodutível e em camadas.
- **Âncoras:** Fontes CSV → Polars (ETL) → Trusted (parquet) → Feature Engineering → Analytics →
  Clusterização/Priorização. Stack: Python, Polars, scikit-learn.
- **Fala:** "O trabalho é um pipeline em camadas. As três bases da PRF passam por um ETL em **Polars**,
  viram uma camada **Trusted** em parquet, ganham novas variáveis na feature engineering e chegam à
  camada **Analytics**, que alimenta as análises. Tudo reprodutível e versionado."

### Slide 6 — Da matéria-bruta ao dado analítico · 70s
- **Objetivo:** detalhar os 4 passos e a variável-chave.
- **Âncoras:** índice de gravidade = 12·mortos + 6·graves + 2·leves; trecho = UF_BR_km.
- **Fala:** "São quatro passos: tratamento, exploração, engenharia de variáveis e nova exploração.
  A variável central é o **índice de gravidade**, que pesa cada vítima — 12 para morte, 6 para ferido
  grave, 2 para leve. E segmentamos cada rodovia em **trechos**. O donut mostra que o índice separa
  bem: poucos acidentes concentram a severidade."

### Slide 7 — O risco não é aleatório (EDA) · 70s
- **Objetivo:** apresentar as evidências da EDA — agora que a base já foi apresentada.
- **Âncoras:** pico de volume às 18h; noite/amanhecer ~2× mais letais que o dia; pista simples ~10%
  fatais vs. ~5% na dupla.
- **Fala:** "Com a base tratada, a exploração confirma parte da hipótese. O volume se concentra no
  fim de tarde — mas a **letalidade** se concentra à noite e na **pista simples**, o dobro da pista
  dupla. Aqui aparece de novo a frase-chave: **volume e gravidade não são a mesma coisa.**"

### Slide 8 — O risco é concentrado · 60s
- **Objetivo:** entregar a evidência que valida a hipótese da concentração espacial.
- **Âncoras:** 10% dos trechos = 44,7% da gravidade; 36% dos trechos = 80%.
- **Fala:** "E a concentração espacial se confirma: **10% dos trechos concentram quase 45% de toda a
  gravidade.** A priorização deixa de ser opinião e vira aritmética — atacar os trechos certos
  multiplica o impacto de cada real."
- **Transição:** "Se o risco se concentra, quais são esses perfis de trecho? O [próximo apresentador]
  mostra os grupos e fecha com a resposta."

---

## Bloco 3 — Apresentador 3 · Perfis de risco & conclusão (slides 9–12, ~5:30)

### Slide 9 — Quantos perfis existem? · 55s
- **Objetivo:** justificar K=4 sem jargão excessivo.
- **Âncoras:** cotovelo + silhueta; escolha K=4.
- **Fala:** "Usamos KMeans para agrupar trechos parecidos. O método do cotovelo e a silhueta apontam
  para poucos grupos; escolhemos **quatro**, que equilibram compactação estatística e leitura de
  negócio: volume contra letalidade."

### Slide 10 — Quatro perfis e um grupo crítico · 75s
- **Objetivo:** revelar o achado central da clusterização.
- **Âncoras:** grupo crítico = 8,9% dos trechos, 87,8% fatais, 4.410 mortos ≈ 36% do total. A
  projeção PCA (87,3% da variância) mostra o grupo crítico se descolando dos demais.
- **Fala:** "Os trechos se organizam em dois eixos. Há corredores de alto volume, há a malha comum, e
  há um **grupo crítico**: raro, mas letal — 8,9% dos trechos que respondem por **mais de um terço de
  todas as mortes**. É o alvo de maior retorno em vidas."

### Slide 11 — Onde investir e o quê fazer · 75s
- **Objetivo:** traduzir tudo em recomendação por perfil de via.
- **Âncoras:** BR-101/116 (carga: ~1,5 mil mortos cada); BR-222/316/153/262/163 (letalidade + pista
  simples).
- **Fala:** "A resposta tem dois lados. Nos **corredores de carga** — BR-101 e BR-116 — o caminho é
  gestão de tráfego, fiscalização e capacidade. Nas **rodovias de pista simples e alta letalidade** —
  como a BR-316 e a BR-153 — o maior retorno em vidas vem de **duplicação, barreira central e
  iluminação**. E nos trechos pontualmente letais, intervenções dirigidas pelo perfil."

### Slide 12 — A resposta · 60s
- **Objetivo:** responder à pergunta do slide 3 e encerrar com a ressalva.
- **Âncoras:** 36% das mortes em 8,9% dos trechos (o anel à direita).
- **Fala:** "Voltando à pergunta que abriu a apresentação: investir onde o risco **se concentra**,
  não onde ele apenas **se acumula**. Três frentes: corredores de volume, pista simples letal e os
  trechos críticos. A ressalva honesta: sem dados de tráfego e de velocidade da via, a priorização
  fina ainda é limitada — é o nosso próximo passo. Obrigado."
