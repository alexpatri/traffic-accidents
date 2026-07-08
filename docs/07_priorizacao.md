# Conclusão — Priorização de Investimentos

Síntese final que aplica todas as etapas ao objetivo do projeto: **onde investir e o quê
fazer**. Código em `src/modeling/priorizacao/` (`python -m src.modeling.priorizacao.main`);
relatório completo, figuras e a lista priorizada (`trechos_prioritarios.parquet`) em
`outputs/priorizacao/`.

**Dois eixos de risco (volume ≠ gravidade).** A clusterização isolou um grupo crítico —
**8,9% dos trechos concentram ~4.410 mortos (≈36% do total)**, com baixo volume e altíssima
letalidade. A priorização trata os dois eixos separadamente.

**Vias (BRs) prioritárias:**
- **Carga de gravidade (alto volume):** **BR-101** e **BR-116** dominam (~1.500 mortos cada)
  — corredores urbanizados. Foco em **gestão de tráfego, fiscalização e capacidade**.
- **Alta letalidade (pista simples):** **BR-222, BR-135, BR-316, BR-153, BR-262, BR-163**
  combinam muita pista simples e alta % de fatais — perfil de colisão frontal. Foco em
  **duplicação + barreira central + iluminação**.

**Da evidência à proposta.** Cada trecho crítico recebe intervenções segundo seu perfil
observado: pista simples/contramão → **duplicação + barreira**; noturno/pedestre →
**iluminação/travessias**; curva-declive → **geometria/sinalização**; velocidade →
**radar + faixa de ultrapassagem**; álcool → **blitz**; acúmulo de água → **drenagem**.

**Ressalva.** As recomendações apoiam a decisão (associação, não causa) e não substituem
vistoria de engenharia; a ausência de **VMD** e **velocidade da via** limita a normalização
por exposição — principal lacuna para trabalhos futuros.
