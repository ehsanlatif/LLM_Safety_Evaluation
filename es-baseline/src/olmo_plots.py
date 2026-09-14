#!/usr/bin/env python3
"""
OLMo 3 7B — dedicated Olmo-only figures.

The shared analysis scripts (analyze.py, final_results.py, mutation_analysis.py)
mix the 12 OLMo variants into the full cross-model roster (they are green-coded
but not separated). This script produces Olmo-ONLY versions of the main safety
charts so the family can be read on its own, alongside those roster-wide plots
and the existing analysis/fig_olmo_trajectory.png.

Outputs (all under analysis/):
  - fig_olmo_overall.png        unsafe% per variant, SorryBench + AILuminate (LLM judge)
  - fig_olmo_mutation_heatmap.png   12 variants x 20 mutation styles (fulfillment %)
  - fig_olmo_resilience.png     base vs mean-mutation vs worst-case fulfillment %
  - fig_olmo_evaluators.png     LLM judge vs official judge, per variant

Reads only the analysis CSVs, reusing the model flow / labels from olmo_trajectory.
Every figure is guarded: if its source CSV has no OLMo rows yet, it is skipped
with a printed note rather than crashing. Safe to run at any pipeline stage.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import olmo_trajectory as ot  # LABEL, branch lists, STYLES, read_csv, num, AN

AN = ot.AN

# Ordered flow for the per-variant charts: Base, Instruct chain, Think chain, RL-Zero.
ORDER = ([ot.BASE]
         + ot.INSTRUCT[1:]
         + ot.THINK[1:]
         + ot.RLZERO)

# Open-Individual colour-blind-safe palette, one hue per post-training branch.
GREY, BLUE, ORANGE, GREEN = "#999999", "#0072B2", "#E69F00", "#009E73"
BRANCH = {ot.BASE: ("Base", GREY)}
BRANCH.update({m: ("Instruct", BLUE) for m in ot.INSTRUCT[1:]})
BRANCH.update({m: ("Think", ORANGE) for m in ot.THINK[1:]})
BRANCH.update({m: ("RL-Zero", GREEN) for m in ot.RLZERO})

# Mutation styles grouped into the four families (in olmo_trajectory.STYLES order).
FAMILIES = [
    ("Writing", ["slang", "uncommon_dialects", "technical_terms", "role_play",
                 "misspellings", "question"]),
    ("Persuasion", ["authority_endorsement", "misrepresentation", "logical_appeal",
                    "evidence-based_persuasion", "expert_endorsement"]),
    ("Cipher", ["ascii", "caesar", "morse", "atbash"]),
    ("Translation", ["translate-fr", "translate-ml", "translate-mr", "translate-ta",
                     "translate-zh-cn"]),
]

_RC = {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
       "savefig.dpi": 140, "savefig.bbox": "tight"}


def _labels(models):
    return [ot.LABEL[m] for m in models]


def _colors(models):
    return [BRANCH[m][1] for m in models]


def _branch_legend():
    seen, handles = set(), []
    for m in ORDER:
        name, col = BRANCH[m]
        if name not in seen:
            seen.add(name)
            handles.append(Patch(facecolor=col, label=name))
    return handles


# ---------------------------------------------------------------------------
# 1) Overall unsafe% (LLM judge), both datasets — Olmo-only version of fig1.
# ---------------------------------------------------------------------------
def fig_overall(data):
    panels = [("sb_llm", "SORRY-Bench (440)"), ("ail_llm", "AILuminate (1,200)")]
    have = {m: data.get(m, {}) for m in ORDER}
    if not any(have[m].get("sb_llm") is not None or have[m].get("ail_llm") is not None
               for m in ORDER):
        print("skip fig_olmo_overall: no LLM-judge rows for OLMo in summary_overall.csv")
        return
    plt.rcParams.update(_RC)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    ys = list(range(len(ORDER)))[::-1]  # base at top
    for ax, (metric, title) in zip(axes, panels):
        vals = [have[m].get(metric) for m in ORDER]
        ax.barh(ys, [v if v is not None else 0 for v in vals],
                color=_colors(ORDER), edgecolor="white")
        for y, v in zip(ys, vals):
            if v is not None:
                ax.text(v + 0.5, y, f"{v:.1f}", va="center", fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("Unsafe % (Claude judge, answer-level)")
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(_labels(ORDER))
    fig.legend(handles=_branch_legend(), loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("OLMo 3 (7B) — answer-level unsafe rate by variant", y=1.06)
    out = AN / "fig_olmo_overall.png"
    fig.savefig(out); plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# 2) Mutation heatmap — Olmo-only version of fig_mutation_heatmap.
# ---------------------------------------------------------------------------
def fig_mutation_heatmap():
    rows = {r["model"]: r for r in ot.read_csv(AN / "mutation_fulfillment.csv")}
    models = [m for m in ORDER if m in rows]
    if not models:
        print("skip fig_olmo_mutation_heatmap: no OLMo rows in mutation_fulfillment.csv "
              "(run the mutation judge first)")
        return
    styles = [s for _, ss in FAMILIES for s in ss]
    mat = [[ot.num(rows[m].get(s)) for s in styles] for m in models]
    plt.rcParams.update(_RC)
    fig, ax = plt.subplots(figsize=(0.42 * len(styles) + 3, 0.5 * len(models) + 2))
    im = ax.imshow([[v if v is not None else float("nan") for v in r] for r in mat],
                   aspect="auto", cmap="Reds", vmin=0, vmax=100)
    ax.set_xticks(range(len(styles)))
    ax.set_xticklabels(styles, rotation=90, fontsize=7)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(_labels(models))
    for i, r in enumerate(mat):
        for j, v in enumerate(r):
            if v is not None:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6,
                        color="white" if v > 55 else "black")
    # Family separators + labels along the top.
    x = 0
    for name, ss in FAMILIES:
        if x:
            ax.axvline(x - 0.5, color="black", lw=1)
        ax.text(x + len(ss) / 2 - 0.5, -1.2, name, ha="center", fontsize=8, weight="bold")
        x += len(ss)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Fulfillment %")
    ax.set_title("OLMo 3 (7B) — mutation fulfillment by style (official ft-Mistral judge)",
                 pad=24)
    out = AN / "fig_olmo_mutation_heatmap.png"
    fig.savefig(out); plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# 3) Resilience gap — base vs mean-mutation vs worst-case fulfillment.
# ---------------------------------------------------------------------------
def fig_resilience(data):
    rows = {r["model"]: r for r in ot.read_csv(AN / "mutation_fulfillment.csv")}
    models = [m for m in ORDER if m in rows]
    if not models:
        print("skip fig_olmo_resilience: no OLMo rows in mutation_fulfillment.csv")
        return
    styles = [s for _, ss in FAMILIES for s in ss]
    base, mean_m, worst = [], [], []
    for m in models:
        vals = [ot.num(rows[m].get(s)) for s in styles]
        vals = [v for v in vals if v is not None]
        base.append(data.get(m, {}).get("sb_off"))     # unmutated official fulfillment
        mean_m.append(sum(vals) / len(vals) if vals else None)
        worst.append(max(vals) if vals else None)
    plt.rcParams.update(_RC)
    fig, ax = plt.subplots(figsize=(0.9 * len(models) + 2, 5))
    xs = range(len(models))
    w = 0.26
    for k, (series, lbl, col) in enumerate([
            (base, "Base (unmutated)", GREY),
            (mean_m, "Mean over 20 mutations", BLUE),
            (worst, "Worst-case mutation", ORANGE)]):
        ax.bar([x + (k - 1) * w for x in xs],
               [v if v is not None else 0 for v in series], w, label=lbl, color=col)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(_labels(models), rotation=45, ha="right")
    ax.set_ylabel("SORRY-Bench fulfillment %")
    ax.set_title("OLMo 3 (7B) — safety resilience under linguistic mutation")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    out = AN / "fig_olmo_resilience.png"
    fig.savefig(out); plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# 4) Cross-evaluator agreement — LLM judge vs official judge, per variant.
# ---------------------------------------------------------------------------
def fig_evaluators(data):
    have_off = any(data.get(m, {}).get("sb_off") is not None
                   or data.get(m, {}).get("ail_off") is not None for m in ORDER)
    if not have_off:
        print("skip fig_olmo_evaluators: no official-judge rows for OLMo yet "
              "(run eval_sorrybench.py / eval_ailuminate.py)")
        return
    plt.rcParams.update(_RC)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    xs = range(len(ORDER))
    w = 0.4
    # SORRY-Bench: LLM unsafe% vs official fulfillment%.
    a = axes[0]
    a.bar([x - w / 2 for x in xs], [data.get(m, {}).get("sb_llm") or 0 for m in ORDER],
          w, label="LLM judge (unsafe%)", color=BLUE)
    a.bar([x + w / 2 for x in xs], [data.get(m, {}).get("sb_off") or 0 for m in ORDER],
          w, label="Official ft-Mistral (fulfillment%)", color=GREY)
    a.set_title("SORRY-Bench")
    # AILuminate: LLM unsafe% vs official (LlamaGuard-2) unsafe%.
    b = axes[1]
    b.bar([x - w / 2 for x in xs], [data.get(m, {}).get("ail_llm") or 0 for m in ORDER],
          w, label="LLM judge (unsafe%)", color=BLUE)
    b.bar([x + w / 2 for x in xs], [data.get(m, {}).get("ail_off") or 0 for m in ORDER],
          w, label="Official LlamaGuard-2 (unsafe%)", color=GREY)
    b.set_title("AILuminate")
    for ax in axes:
        ax.set_xticks(list(xs))
        ax.set_xticklabels(_labels(ORDER), rotation=45, ha="right")
        ax.set_ylabel("%")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("OLMo 3 (7B) — unified LLM judge vs official judges", y=1.02)
    out = AN / "fig_olmo_evaluators.png"
    fig.savefig(out); plt.close(fig)
    print(f"wrote {out}")


def main():
    AN.mkdir(exist_ok=True)
    data = ot.load()
    fig_overall(data)
    fig_mutation_heatmap()
    fig_resilience(data)
    fig_evaluators(data)


if __name__ == "__main__":
    main()
