import os
import argparse
from pathlib import Path

_mpl_cache_dir = Path.cwd() / ".mplconfig"
_mpl_cache_dir.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache_dir))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_CSV = os.path.expanduser("~/Escritorio/fairface_cribado/descriptors_join_300.csv")

MEASURES = [
    "nasal_index", "canthal_index", "fissure_biocular_ratio", "mouth_nose_ratio",
    "mouth_biocular_ratio", "nasal_tip_angle", "eyebrow_tilt", "facial_index",
    "naso_intercanthal_ratio",
]
LABELS = {  
    "nasal_index": "Índice nasal", "canthal_index": "Índice cantal",
    "fissure_biocular_ratio": "Razón hendidura/biocular", "mouth_nose_ratio": "Razón boca/nariz",
    "mouth_biocular_ratio": "Razón boca/biocular", "nasal_tip_angle": "Ángulo punta nasal",
    "eyebrow_tilt": "Inclinación de ceja", "facial_index": "Índice facial",
    "naso_intercanthal_ratio": "Razón nasointercantal",
}
AGE_ORDER = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70"]
NAVY, TEAL, TERRA, VINO, GRID = "#1F3A5F", "#3C7A89", "#C0603A", "#7A3B4A", "#DDDDDD"


def _style():
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "font.size": 11,
        "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 9.5, "text.color": "#2B2B2B", "axes.edgecolor": "#888888",
        "axes.labelcolor": "#2B2B2B", "xtick.color": "#2B2B2B", "ytick.color": "#2B2B2B",
    })


def signal_ratio(df):
    """Range of race means over the global individual std, per descriptor."""
    out = {}
    for m in MEASURES:
        rng = df.groupby("race")[m].mean().max() - df.groupby("race")[m].mean().min()
        out[m] = rng / df[m].std()
    return pd.Series(out).sort_values(ascending=False)


def fig_nasal_by_race(df, figdir):
    _style()
    order = df.groupby("race")["nasal_index"].median().sort_values().index
    data = [df.loc[df.race == r, "nasal_index"].values for r in order]
    short = {"Latino_Hispanic": "Latino", "Southeast Asian": "SE Asian", "Middle Eastern": "Mid. East."}
    fig, ax = plt.subplots(figsize=(6.3, 3.9))
    bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False,
                    medianprops=dict(color="white", lw=1.6),
                    whiskerprops=dict(color=NAVY), capprops=dict(color=NAVY), boxprops=dict(color=NAVY))
    for p in bp["boxes"]:
        p.set_facecolor(NAVY); p.set_alpha(0.85)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([short.get(r, r) for r in order], rotation=20, ha="right")
    ax.set_ylabel("Índice nasal")
    ax.grid(axis="y", ls=":", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig_demo_nasal_raza.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_age_trends(df, figdir):
    _style()
    labels = [a if a != "more than 70" else ">70" for a in AGE_ORDER]
    series = [("Índice facial", "facial_index", NAVY, "o"),
              ("Ángulo punta nasal", "nasal_tip_angle", TERRA, "s"),
              ("Razón boca/biocular", "mouth_biocular_ratio", TEAL, "^")]
    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    x = range(len(AGE_ORDER))
    for name, key, c, mk in series:
        z = (df[key] - df[key].mean()) / df[key].std()
        means = [z[df.age == a].mean() for a in AGE_ORDER]
        ax.plot(x, means, color=c, marker=mk, ms=5, lw=1.8, label=name)
    ax.axhline(0, color=GRID, lw=0.8)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Media por grupo (z-score)"); ax.set_xlabel("Grupo de edad")
    ax.grid(axis="y", ls=":", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              handletextpad=0.5, columnspacing=1.6)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig_demo_edad.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_signal(ratio, figdir):
    _style()
    r = ratio.sort_values()
    labels = [LABELS[m] for m in r.index]
    cols = [NAVY if v >= 1 else "#9AA0A6" for v in r.values]
    fig, ax = plt.subplots(figsize=(6.3, 4.1))
    ax.barh(range(len(r)), r.values, color=cols, edgecolor="none")
    ax.axvline(1.0, color=VINO, lw=1.3, ls="--")
    ax.set_yticks(range(len(r))); ax.set_yticklabels(labels)
    ax.set_xlabel("Rango entre razas / desviación individual")
    ax.grid(axis="x", ls=":", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig_demo_senal.pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Demographic analysis of the descriptors over a labelled dataset.")
    ap.add_argument("--csv", default=DEFAULT_CSV, help="joined descriptors+labels CSV")
    ap.add_argument("--figdir", default=".", help="directory for the output PDFs")
    ap.add_argument("--no-figs", action="store_true", help="only print the tables")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    print(f"n = {len(df)}\n")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("=== mean by race ===\n", df.groupby("race")[MEASURES].mean().round(2), "\n")
        print("=== mean by sex ===\n", df.groupby("gender")[MEASURES].mean().round(2), "\n")
        print("=== mean by age ===\n", df.groupby("age")[MEASURES].mean().reindex(AGE_ORDER).round(2), "\n")
    ratio = signal_ratio(df)
    print("=== between-group / within-individual ratio ===\n", ratio.round(2))

    if not args.no_figs:
        os.makedirs(args.figdir, exist_ok=True)
        fig_nasal_by_race(df, args.figdir)
        fig_age_trends(df, args.figdir)
        fig_signal(ratio, args.figdir)
        print(f"\nfigures written to {args.figdir}")


if __name__ == "__main__":
    main()
