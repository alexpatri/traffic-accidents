"""Etapa de ML Supervisionado — classificação de risco de trechos rodoviários (PRF).

Classifica cada trecho em um nível ordinal de risco (Baixa/Média/Alta/Crítica) derivado
de `indice_gravidade_medio`, a partir de **características estruturais** do segmento
(tipo de pista, traçado, uso do solo, região) — sem identificadores (`br`, `km_faixa`) e
sem variáveis de desfecho, de modo a generalizar para outras vias.

Execução: ``python -m src.modeling.ml.main``.
"""
