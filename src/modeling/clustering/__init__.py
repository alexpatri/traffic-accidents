"""Etapa de Clusterização (Aprendizado Não Supervisionado) — acidentes PRF.

Descobre perfis naturais de risco de **trechos rodoviários** (`trechos.py`).

A clusterização de acidentes foi investigada e não incluída como entrega — o KMeans não
formou perfis multivariados nítidos (apenas redescobre `tipo_pista`). O achado está
documentado no relatório (`outputs/clustering/relatorio_clustering.md`) e no README.

Execução: ``python -m src.modeling.clustering.main``.
"""
