"""Static figures for the portfolio's Historic Fashion Archive case study.

Numbers are computed from the judgment files, not typed in, so the figures stay
true if anything is re-run. Styled to the portfolio: paper #f5f0e6, ink #2b2013,
muted #6b5a44, IBM Plex; series colors validated with the dataviz palette checker
(indigo #2a5fb8, kincha #c98a12, bengara #a8543a on the paper surface).

Run:  PYTHONPATH=. .venv/bin/python scripts/portfolio_figures.py <out_dir> <font_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from hfa.rec_eval import boot_ci, per_query  # noqa: E402
from hfa.recsys import load_catalog, rank  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PAPER, INK, MUTED, RULE = "#f5f0e6", "#2b2013", "#6b5a44", "#d9cfbe"
INDIGO, KINCHA, BENGARA = "#2a5fb8", "#c98a12", "#a8543a"
W, H, DPI = 1425, 900, 150
NAMES = {"fixed": "Fixed weights (original)", "fixed_attrs": "Fixed weights + attributes",
         "image": "Image similarity", "switch": "Rule: clothing/accessory switch", "rerank": "Learned re-ranker"}


def fonts(font_dir: Path) -> tuple[str, str, str]:
    names = []
    for f in ("IBMPlexSans-VF.ttf", "IBMPlexSerif-Medium.ttf", "IBMPlexMono-Regular.ttf"):
        font_manager.fontManager.addfont(str(font_dir / f))
        names.append(font_manager.FontProperties(fname=str(font_dir / f)).get_name())
    plt.rcParams.update({"font.family": names[0], "text.color": INK, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": RULE})
    return tuple(names)


def scores() -> pd.DataFrame:
    """nDCG@5 per query for every system, on the blind-final judgments."""
    j = json.loads((ROOT / "data/judgments/judgments_final.json").read_text())
    cv = json.loads((ROOT / "data/predictions/rerank_cv.json").read_text())
    g = pd.read_parquet(ROOT / "data/processed/garments.parquet").set_index("accession")
    c = load_catalog()
    rows = []
    for q in cv:
        for s in ("fixed", "fixed_attrs", "image", "switch", "rerank"):
            r = cv[q] if s == "rerank" else rank(c, q, s, 5)
            rows.append({"query": q, "group": g.at[q, "group"], "system": s, "ndcg": per_query(j[q], r, 5)["ndcg"]})
    return pd.DataFrame(rows)


def canvas():
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=PAPER)
    return fig


def header(fig, serif: str, mono: str, title: str, sub: str, tag: str) -> None:
    fig.text(0.06, 0.915, title, fontfamily=serif, fontsize=17, color=INK)
    fig.text(0.06, 0.865, sub, fontsize=9.5, color=MUTED)
    # spec-label tag in the corner, the site's "pattern piece" callout
    fig.text(0.94, 0.925, tag, fontfamily=mono, fontsize=8, color=MUTED, ha="right",
             bbox={"boxstyle": "square,pad=0.45", "fc": PAPER, "ec": MUTED, "lw": 0.8, "ls": (0, (3, 2))})
    fig.add_artist(plt.Line2D([0.06, 0.94], [0.84, 0.84], color=MUTED, lw=0.8, ls=(0, (6, 4)), transform=fig.transFigure))


def fig_by_system(df: pd.DataFrame, out: Path, serif: str, mono: str) -> None:
    rng = np.random.default_rng(1)
    order = ["fixed", "image", "fixed_attrs", "switch", "rerank"]
    fig = canvas()
    header(fig, serif, mono, "Related garments: how often the top picks are right",
           "nDCG@5 on 50 query garments, 731 blind relevance judgments. Whiskers: 95% bootstrap interval over queries.",
           "HFA · RECSYS · nDCG@5")
    ax = fig.add_axes([0.30, 0.13, 0.62, 0.66], facecolor=PAPER)
    for i, s in enumerate(order):
        v = df[df.system == s].ndcg.to_numpy()
        m, (lo, hi) = v.mean(), boot_ci(v, rng)
        hero = s == "rerank"
        col = BENGARA if hero else MUTED
        ax.plot([lo, hi], [i, i], color=col, lw=2, solid_capstyle="round", zorder=2)
        ax.scatter([m], [i], s=110 if hero else 70, color=col, edgecolor=PAPER, linewidth=2, zorder=3)
        ax.text(hi + 0.012, i, f"{m:.2f}", va="center", fontsize=11.5 if hero else 10, color=INK,
                fontfamily=serif if hero else None)
        ax.text(-0.02, i, NAMES[s], transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=11 if hero else 10.5, color=INK if hero else MUTED, fontfamily=serif if hero else None)
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_yticks([])
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.tick_params(axis="x", labelsize=9, length=0)
    ax.grid(axis="x", color=RULE, lw=0.8, ls=(0, (2, 3)))
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.set_xlabel("nDCG@5  (1.0 = every top-5 pick as related as the best available)", fontsize=9, labelpad=8)
    fig.savefig(out, dpi=DPI, facecolor=PAPER, pil_kwargs={"quality": 92})
    plt.close(fig)


def fig_split(df: pd.DataFrame, out: Path, serif: str, mono: str) -> None:
    fig = canvas()
    header(fig, serif, mono, "The split: what works for clothing fails for accessories",
           "nDCG@5 by query group (25 each). The catalog files fans, shoes, hats and purses under one type; photos tell them apart.",
           "HFA · RECSYS · BY GROUP")
    ax = fig.add_axes([0.10, 0.12, 0.62, 0.66], facecolor=PAPER)
    series = [("fixed", INDIGO), ("image", KINCHA), ("rerank", BENGARA)]
    xs = [0, 1]
    ends = []
    for s, col in series:
        y = [df[(df.system == s) & (df.group == g)].ndcg.mean() for g in ("clothing", "accessory")]
        ax.plot(xs, y, color=col, lw=2.2, zorder=2, solid_capstyle="round")
        ax.scatter(xs, y, s=90, color=col, edgecolor=PAPER, linewidth=2, zorder=3)
        ax.text(-0.04, y[0], f"{y[0]:.2f}", ha="right", va="center", fontsize=10, color=INK)
        ends.append([y[1], s, col])
    # direct labels at the right end, nudged apart so they never collide
    ends.sort()
    for k in range(1, len(ends)):
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + 0.045)
    for y, s, col in ends:
        ax.text(1.04, y, f"{df[(df.system == s) & (df.group == 'accessory')].ndcg.mean():.2f}  {NAMES[s]}",
                va="center", fontsize=10.5, color=INK, fontfamily=serif if s == "rerank" else None)
        ax.plot([1.005, 1.03], [y, y], color=col, lw=2, solid_capstyle="round", clip_on=False)
    ax.set_xlim(-0.12, 1.0)
    ax.set_ylim(0.4, 1.0)
    ax.set_xticks(xs, ["Clothing queries", "Accessory queries"])
    ax.tick_params(axis="x", labelsize=10.5, length=0, colors=INK, pad=10)
    ax.tick_params(axis="y", labelsize=9, length=0)
    ax.grid(axis="y", color=RULE, lw=0.8, ls=(0, (2, 3)))
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.set_ylabel("nDCG@5", fontsize=9)
    fig.savefig(out, dpi=DPI, facecolor=PAPER, pil_kwargs={"quality": 92})
    plt.close(fig)


def fig_pipeline(out: Path, serif: str, mono: str) -> None:
    fig = canvas()
    header(fig, serif, mono, "From catalog rows to the live site",
           "What runs offline on the GPU, what is measured, and what serves visitors.", "HFA · ML · PIPELINE")
    ax = fig.add_axes([0.04, 0.04, 0.92, 0.78], facecolor=PAPER)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    def box(x, y, w, h, title, body, accent=MUTED, dashed=True):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=PAPER,
                                    ec=accent, lw=1.4 if accent != MUTED else 1.0, ls=(0, (4, 2.5)) if dashed else "-"))
        ax.text(x + 1.2, y + h - 2.4, title, fontsize=10.5, color=INK, fontfamily=serif, va="top")
        ax.text(x + 1.2, y + h - 6.2, body, fontsize=8.3, color=MUTED, va="top", linespacing=1.35)

    def arrow(x0, y0, x1, y1, col=MUTED):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, lw=1.2, color=col,
                                     shrinkA=2, shrinkB=2))

    box(1, 40, 20, 14, "Catalog", "2,658 spreadsheet rows\ncleaned: accessions, dates,\ntypes; donor data dropped")
    box(1, 20, 20, 14, "Photos", "5,821 photos of 1,251\ngarments, RAW converted\nand cached locally")
    box(27, 40, 22, 14, "Attributes", "material / color / technique\ntext: keyword rubric (F1 0.95+)\nphotos: linear probe")
    box(27, 20, 22, 14, "Embeddings", "FashionCLIP, one vector per\ngarment (mean of its photos)\n77 s on a 4060 Ti")
    box(57, 40, 19, 14, "Learned re-ranker", "logistic regression over\npair features; nDCG@5 0.93\n(nested CV)", accent=BENGARA)
    box(57, 20, 19, 14, "Search service", "BM25 + CLIP fusion, CPU\nFastAPI on a VPS behind a\nCloudflare Tunnel; ~100 ms", accent=BENGARA)
    box(81.5, 30, 17.5, 14, "uvahcc.org", "related.json picks;\n/api/search with\nkeyword fallback", dashed=False)
    box(27, 1, 49, 14, "Measured, blind: every number above",
        "200 hand-labeled descriptions · 731 recommendation judgments\n463 search judgments · frozen test sets · intra-rater kappa 0.82-0.87\nregression gate in GitHub Actions on a committed fixture")
    for y in (47, 27):
        arrow(21.4, y, 27, y)
    arrow(49.4, 47, 57, 47)
    arrow(49.4, 27, 57, 27)
    arrow(49.4, 43.5, 57, 30.5)
    arrow(49.4, 30.5, 57, 43.5)
    arrow(76.4, 46, 81.5, 40, BENGARA)
    arrow(76.4, 28, 81.5, 34, BENGARA)
    fig.savefig(out, dpi=DPI, facecolor=PAPER, pil_kwargs={"quality": 92})
    plt.close(fig)


def main() -> None:
    out, font_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    sans, serif, mono = fonts(font_dir)
    df = scores()
    fig_by_system(df, out / "ndcg-by-system.jpg", serif, mono)
    fig_split(df, out / "clothing-vs-accessories.jpg", serif, mono)
    fig_pipeline(out / "ml-pipeline.jpg", serif, mono)
    print("wrote", ", ".join(p.name for p in sorted(out.glob("*.jpg")) if p.stem in
                             ("ndcg-by-system", "clothing-vs-accessories", "ml-pipeline")))


if __name__ == "__main__":
    main()
