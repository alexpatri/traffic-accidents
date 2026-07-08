"""Análise final de priorização: vias e trechos de maior risco + propostas.

Sintetiza as etapas anteriores para responder ao objetivo do projeto — onde investir e
o quê fazer. Produz o relatório final, figuras e a lista priorizada de trechos.

Execução:
    python -m src.modeling.priorizacao.main
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import polars as pl

from src.modeling.priorizacao import config

logger = logging.getLogger("priorizacao")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler()],
    )


def _save_fig(fig, name: str) -> None:
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=config.DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figura salva: %s", path.relative_to(config.PROJECT_ROOT))


# --------------------------------------------------------------------------- #
# Agregações
# --------------------------------------------------------------------------- #
def _perfil_por_br(ac: pl.DataFrame) -> pl.DataFrame:
    return (
        ac.group_by("br")
        .agg(
            pl.len().alias("acidentes"),
            pl.col("mortos").sum().alias("mortos"),
            pl.col("indice_gravidade").sum().alias("ig_total"),
            (pl.col("fatal").mean() * 100).round(1).alias("pct_fatal"),
            (pl.col("periodo_noturno").mean() * 100).round(0).alias("pct_noturno"),
            (pl.col("tipo_pista").eq("Simples").mean() * 100).round(0).alias("pct_simples"),
            (pl.col("uso_solo").eq("Sim").mean() * 100).round(0).alias("pct_urbano"),
        )
        .sort(["ig_total", "br"], descending=[True, False])
    )


def _perfil_por_trecho(ac: pl.DataFrame) -> pl.DataFrame:
    return ac.group_by("trecho").agg(
        pl.col("uf").first().alias("uf"),
        pl.col("br").first().alias("br"),
        pl.len().alias("acidentes"),
        pl.col("mortos").sum().alias("mortos"),
        pl.col("indice_gravidade").sum().alias("ig_total"),
        (pl.col("fatal").mean() * 100).round(0).alias("pct_fatal"),
        (pl.col("periodo_noturno").mean() * 100).round(0).alias("pct_not"),
        (pl.col("uso_solo").eq("Sim").mean() * 100).round(0).alias("pct_urb"),
        (pl.col("tem_curva").mean() * 100).round(0).alias("curva"),
        (pl.col("tem_declive").mean() * 100).round(0).alias("decl"),
        # mode().sort().first(): desempate alfabético torna a moda determinística.
        pl.col("tipo_pista").mode().sort().first().alias("pista"),
        pl.col("causa_acidente_agrupada").mode().sort().first().alias("causa"),
    )


# --------------------------------------------------------------------------- #
# Regras de intervenção (perfil do trecho -> proposta)
# --------------------------------------------------------------------------- #
def _sugerir(r: dict) -> str:
    causa = (r.get("causa") or "").lower()
    s: list[str] = []
    if r["pista"] == "Simples" and (
        r["pct_fatal"] >= config.LIM_SIMPLES_FATAL
        or "contramão" in causa or "ultrapassagem" in causa
    ):
        s.append("Duplicação + barreira central")
    if r["pct_not"] >= config.LIM_NOTURNO_PCT or "pedestre" in causa:
        s.append("Iluminação")
    if r["curva"] >= config.LIM_CURVA_PCT or r["decl"] >= config.LIM_CURVA_PCT:
        s.append("Geometria/sinalização + redutores")
    if "velocidade" in causa or "ultrapassagem" in causa:
        s.append("Radar + faixa de ultrapassagem")
    if "álcool" in causa or "alcool" in causa:
        s.append("Blitz de alcoolemia")
    if "pedestre" in causa:
        s.append("Passarela/travessia iluminada")
    elif r["pct_urb"] >= config.LIM_URBANO_PCT:
        s.append("Gestão de tráfego urbano")
    if any(k in causa for k in ("água", "agua", "chuva", "pavimento", "pista escorreg")):
        s.append("Drenagem + recuperação de pavimento")
    if not s:
        s.append("Fiscalização + campanha educativa")
    out, seen = [], set()
    for x in s:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return "; ".join(out)


def _com_intervencao(df: pl.DataFrame) -> pl.DataFrame:
    props = [_sugerir(r) for r in df.iter_rows(named=True)]
    return df.with_columns(pl.Series("intervencao", props))


# --------------------------------------------------------------------------- #
# Figuras
# --------------------------------------------------------------------------- #
def _fig_top_br(br: pl.DataFrame) -> None:
    top = br.head(config.TOP_BR)
    fig, ax = plt.subplots(figsize=config.FIGSIZE)
    ax.barh([f"BR-{b}" for b in top.get_column("br")][::-1],
            top.get_column("ig_total").to_numpy()[::-1], color=config.BAR_COLOR)
    ax.set_xlabel("Índice de gravidade total (carga de severidade)")
    ax.set_title("Vias (BRs) com maior carga de gravidade")
    ax.grid(True, axis="x", alpha=0.3)
    _save_fig(fig, "top_brs_carga")


def _fig_volume_letalidade(br: pl.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=config.FIGSIZE)
    x = br.get_column("acidentes").to_numpy()
    y = br.get_column("pct_fatal").to_numpy()
    m = br.get_column("mortos").to_numpy()
    ax.scatter(x, y, s=m / 3 + 10, alpha=0.5, color=config.BAR_COLOR)
    ax.axvline(float(br.get_column("acidentes").median()), color="grey", ls="--", alpha=0.5)
    ax.axhline(float(br.get_column("pct_fatal").median()), color="grey", ls="--", alpha=0.5)
    for row in br.head(12).iter_rows(named=True):
        ax.annotate(f"BR-{row['br']}", (row["acidentes"], row["pct_fatal"]), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Volume de acidentes (log)")
    ax.set_ylabel("% de acidentes fatais (letalidade)")
    ax.set_title("BRs: volume × letalidade (tamanho = mortos)")
    ax.grid(True, alpha=0.3)
    _save_fig(fig, "br_volume_letalidade")


# --------------------------------------------------------------------------- #
# Relatório
# --------------------------------------------------------------------------- #
def _tab(df: pl.DataFrame, cols: list[str], titulos: list[str], trunc: int = 90) -> str:
    head = "| " + " | ".join(titulos) + " |\n|" + "---|" * len(titulos)
    linhas = []
    for row in df.iter_rows(named=True):
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, str) and len(v) > trunc:
                v = v[: trunc - 1] + "…"
            vals.append(f"{v:.0f}" if isinstance(v, float) else str(v))
        linhas.append("| " + " | ".join(vals) + " |")
    return head + "\n" + "\n".join(linhas)


def _relatorio(br, hotspots, letais, c2_mortos, c2_pct, total_mortos) -> str:
    br_carga = _tab(
        br.head(config.TOP_BR),
        ["br", "acidentes", "mortos", "ig_total", "pct_fatal", "pct_simples", "pct_noturno"],
        ["BR", "Acid.", "Mortos", "Índ.total", "% fatal", "% simples", "% noturno"],
    )
    br_letal = _tab(
        br.filter(pl.col("acidentes") >= 1000).sort("pct_fatal", descending=True).head(8),
        ["br", "acidentes", "mortos", "pct_fatal", "pct_simples", "pct_noturno"],
        ["BR", "Acid.", "Mortos", "% fatal", "% simples", "% noturno"],
    )
    t_hot = _tab(
        hotspots,
        ["trecho", "acidentes", "mortos", "ig_total", "pista", "pct_urb", "intervencao"],
        ["Trecho", "Acid.", "Mortos", "Índ.tot", "Pista", "% urb", "Intervenção sugerida"],
    )
    t_let = _tab(
        letais,
        ["trecho", "acidentes", "mortos", "pct_fatal", "pista", "causa", "intervencao"],
        ["Trecho", "Acid.", "Mortos", "% fatal", "Pista", "Causa", "Intervenção sugerida"],
    )
    return f"""# Relatório Final — Priorização de Investimentos (Acidentes PRF)

Síntese das etapas do projeto (EDA → Feature Engineering → Clusterização → ML) aplicada ao
objetivo central: **onde investir e o quê fazer** para reduzir mortes e feridos na malha
rodoviária federal. Base: 2024–2025. Leituras são **associativas, não causais**.

## 1. Dois eixos de risco (não confundir volume com letalidade)

O projeto mostrou que **volume ≠ gravidade**: há corredores que concentram muitos acidentes
(carga total) e há trechos raros porém **altamente letais**. A priorização trata os dois.
A clusterização isolou um grupo crítico — **{c2_pct:.1f}% dos trechos concentram
{c2_mortos:,} mortos** ({c2_mortos / total_mortos * 100:.0f}% do total), com baixíssimo
volume e altíssima letalidade.

![BRs: volume × letalidade](figures/br_volume_letalidade.png)

## 2. Vias (BRs) prioritárias

### 2a. Maior carga de gravidade (corredores de alto volume)

{br_carga}

![Carga por BR](figures/top_brs_carga.png)

**BR-101 e BR-116** dominam a carga absoluta (~1.500 mortos cada) — corredores longos,
urbanizados e movimentados. Prioridade de **capacidade, fiscalização e gestão de tráfego**.

### 2b. Maior letalidade (rodovias de pista simples, alto risco por acidente)

{br_letal}

Rodovias como **BR-316, BR-153, BR-262, BR-163** combinam **alta % de pista simples** e
**alta letalidade** — perfil clássico de colisão frontal. Prioridade de **duplicação,
barreira central e iluminação**.

## 3. Trechos críticos

### 3a. Hotspots de carga (maior índice de gravidade total)

{t_hot}

### 3b. Trechos mais letais (≥ {config.MIN_VOL_LETAL} acidentes, maior % fatal)

{t_let}

## 4. Da evidência à proposta — matriz de intervenção

As intervenções acima são atribuídas por regras sobre o **perfil observado** de cada trecho:

| Evidência no trecho | Intervenção proposta |
|---|---|
| Pista simples + alta letalidade / contramão / ultrapassagem | **Duplicação / faixa adicional + barreira central** |
| Alta % de acidentes noturnos ou atropelamentos | **Iluminação** |
| Alta % em curva/declive | **Melhoria geométrica + sinalização/redutores** |
| Causa velocidade incompatível / ultrapassagem | **Fiscalização eletrônica (radar) + faixa de ultrapassagem** |
| Causa ingestão de álcool | **Blitz de alcoolemia** |
| Pedestre / trecho urbano | **Passarelas/travessias iluminadas + gestão de tráfego** |
| Causa chuva / acúmulo de água / pavimento | **Drenagem + recuperação do pavimento** |

## 5. Recomendação (resumo executivo)

1. **Corredores de alto volume (BR-101, BR-116):** gestão de tráfego, fiscalização de
   velocidade/distância e melhorias de capacidade nos hotspots urbanos (SC, SP, PE).
2. **Rodovias de pista simples e alta letalidade (BR-316, BR-153, BR-262, BR-163):**
   programa de **duplicação + barreira central + iluminação** — maior retorno em vidas.
3. **Trechos pontuais extremamente letais:** intervenções dirigidas pelo perfil (geometria
   em curvas/declives, radares onde a causa é velocidade, travessias onde há pedestres,
   drenagem onde há acúmulo de água).

## 6. Ressalvas

- Métricas por trecho são ruidosas em baixo volume; o índice depende dos pesos (12/6/2).
- Sem dados de **volume de tráfego (VMD)** e **velocidade da via**, a normalização por
  exposição e a priorização fina ficam limitadas — principal lacuna para trabalhos futuros.
- As recomendações apoiam a decisão; não substituem vistoria de engenharia de campo.
"""


def run() -> None:
    _setup_logging()
    logger.info("=== Análise final — priorização de investimentos ===")
    ac = pl.read_parquet(config.ACIDENTES_FILE)
    tr = pl.read_parquet(config.TRECHOS_FILE)
    cl = pl.read_parquet(config.CLUSTERS_FILE)

    br = _perfil_por_br(ac)
    perf = _perfil_por_trecho(ac)
    hotspots = _com_intervencao(
        perf.sort(["ig_total", "trecho"], descending=[True, False]).head(config.TOP_HOTSPOTS)
    )
    letais = _com_intervencao(
        perf.filter(pl.col("acidentes") >= config.MIN_VOL_LETAL)
        .sort(["pct_fatal", "mortos", "ig_total", "trecho"],
              descending=[True, True, True, False])
        .head(config.TOP_LETAIS)
    )

    # Concentração de mortes no cluster letal (etapa de clusterização).
    c2 = cl.filter(pl.col("cluster") == config.CLUSTER_LETAL).join(
        tr.select("trecho", "mortos"), on="trecho", how="left")
    c2_mortos = int(c2.get_column("mortos").sum())
    c2_pct = c2.height / cl.height * 100
    total_mortos = int(tr.get_column("mortos").sum())

    _fig_top_br(br)
    _fig_volume_letalidade(br)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (
        pl.concat([hotspots.with_columns(pl.lit("hotspot").alias("tipo")),
                   letais.with_columns(pl.lit("letal").alias("tipo"))], how="diagonal")
        .write_parquet(config.RANKING_FILE)
    )
    logger.info("Trechos prioritários salvos: %s",
                config.RANKING_FILE.relative_to(config.PROJECT_ROOT))

    conteudo = _relatorio(br, hotspots, letais, c2_mortos, c2_pct, total_mortos)
    config.REPORT_FILE.write_text(conteudo, encoding="utf-8")
    logger.info("Relatório final gravado em %s",
                config.REPORT_FILE.relative_to(config.PROJECT_ROOT))
    logger.info("=== Priorização concluída — artefatos em outputs/priorizacao/ ===")


if __name__ == "__main__":
    run()
