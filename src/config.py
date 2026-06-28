"""Configuração central do ETL: caminhos, parâmetros de leitura e schema alvo.

Concentra em um único lugar todas as constantes que descrevem *o que* o pipeline
faz com os dados, mantendo `extract`/`transform`/`load` livres de números mágicos.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Caminhos
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
TRUSTED_DIR: Path = DATA_DIR / "trusted"
LOG_DIR: Path = PROJECT_ROOT / "logs"

TRUSTED_FILE: Path = TRUSTED_DIR / "acidentes_trusted.parquet"

# Datasets de origem, indexados pelo ano que representam.
RAW_FILES: dict[int, Path] = {
    2024: RAW_DIR / "datatran2024.csv",
    2025: RAW_DIR / "datatran2025.csv",
}

# --------------------------------------------------------------------------- #
# Parâmetros de leitura dos CSVs
# --------------------------------------------------------------------------- #
# Os arquivos datatran são publicados em ISO-8859-1 e separados por ponto-e-vírgula.
ENCODING: str = "latin-1"
SEPARATOR: str = ";"

# Tokens que representam ausência de valor e devem virar nulo na leitura.
NULL_TOKENS: list[str] = ["NA", "", "(null)", "null", "NULL"]

# --------------------------------------------------------------------------- #
# Padronização de nomes de colunas
# --------------------------------------------------------------------------- #
# Os cabeçalhos já chegam em snake_case; o mapa aplica apenas renomeações
# semânticas: encurtar `data_inversa` e corrigir a grafia de `metereologica`.
RENAME_MAP: dict[str, str] = {
    "data_inversa": "data",
    "condicao_metereologica": "condicao_meteorologica",
}

# --------------------------------------------------------------------------- #
# Papéis das colunas (nomes já padronizados pós-rename)
# --------------------------------------------------------------------------- #
ID_COL: str = "id"
DATE_COL: str = "data"
TIME_COL: str = "horario"

# Colunas numéricas com separador decimal "," (km também usa "," nos dois anos).
DECIMAL_COLS: list[str] = ["km", "latitude", "longitude"]

# Contagens inteiras de vítimas / envolvidos.
INT_COUNT_COLS: list[str] = [
    "pessoas",
    "mortos",
    "feridos_leves",
    "feridos_graves",
    "ilesos",
    "ignorados",
    "feridos",
    "veiculos",
]

# Coluna de origem adicionada na extração (rastreabilidade, não é feature de domínio).
SOURCE_COL: str = "ano_arquivo"

# Colunas categóricas em que um nulo residual deve virar a categoria "Não informado".
CATEGORICAL_COLS: list[str] = [
    "dia_semana",
    "uf",
    "br",
    "municipio",
    "causa_acidente",
    "tipo_acidente",
    "classificacao_acidente",
    "fase_dia",
    "sentido_via",
    "condicao_meteorologica",
    "tipo_pista",
    "tracado_via",
    "uso_solo",
    "regional",
    "delegacia",
    "uop",
]

NAO_INFORMADO: str = "Não informado"

# Ordem canônica das colunas no arquivo Trusted final.
COLUMN_ORDER: list[str] = [
    ID_COL,
    DATE_COL,
    "dia_semana",
    TIME_COL,
    "uf",
    "br",
    "km",
    "municipio",
    "causa_acidente",
    "tipo_acidente",
    "classificacao_acidente",
    "fase_dia",
    "sentido_via",
    "condicao_meteorologica",
    "tipo_pista",
    "tracado_via",
    "uso_solo",
    "pessoas",
    "mortos",
    "feridos_leves",
    "feridos_graves",
    "ilesos",
    "ignorados",
    "feridos",
    "veiculos",
    "latitude",
    "longitude",
    "regional",
    "delegacia",
    "uop",
    SOURCE_COL,
]

# --------------------------------------------------------------------------- #
# Regras de validação de consistência
# --------------------------------------------------------------------------- #
# Limites aproximados do território brasileiro (com folga) para coordenadas.
LAT_MIN, LAT_MAX = -34.0, 6.0
LON_MIN, LON_MAX = -74.0, -32.0

# Soma das parcelas de vítimas/envolvidos que deveria reproduzir `pessoas`.
VICTIM_PARTS: list[str] = [
    "mortos",
    "feridos_leves",
    "feridos_graves",
    "ilesos",
    "ignorados",
]

# Formato de persistência.
PARQUET_COMPRESSION: str = "zstd"
