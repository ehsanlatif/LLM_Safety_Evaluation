#!/usr/bin/env python3
"""
Combined manuscript Figure 1:
  (a) cross-family answer-level unsafe rate (the 7 audited models, both suites
      pooled, coloured by class) -- the observation; and
  (b) the Qwen3 same-weights reasoning toggle (unsafe rate vs size, thinking
      OFF vs ON, both suites pooled) -- the causal confirmation.

Writes: paper/figs/fig1_combined.png
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter

import analyze as az
import analyze_toggle as at

CROSS = ["opus-4.8", "sonnet-5", "gpt-5.5", "qwen-3b", "llama-3.2-3b",
         "qwen-1.5b", "deepseek-r1-1.5b", "vibethinker-1.5b", "vibethinker-3b"]


def cls(m):
    info = az.MODEL_INFO[m]
    if info["klass"] == "Large closed":
        return "Large closed", az.OI["blue"]
    if info["reasoning"]:
        return "Small open, reasoning", az.OI["vermillion"]
    return "Small open, non-reasoning", az.OI["orange"]


def main():
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 140,
                         "savefig.dpi": 200, "savefig.bbox": "tight"})
    data = az.load()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))

    # ---- (a) cross-family pooled unsafe rate ----
    rowsA = []
    for m in CROSS:
        recs = data[m]["ailuminate"] + data[m]["sorry-bench"]
        u, t = az.count(recs, "answer")
        rowsA.append((m, 100 * u / t if t else 0))
    rowsA.sort(key=lambda r: r[1])
    ys = list(range(len(rowsA)))
    axA.barh(ys, [p for _, p in rowsA], color=[cls(m)[1] for m, _ in rowsA],
             height=0.62, zorder=3)
    for y, (_, p) in zip(ys, rowsA):
        axA.text(p + 1.2, y, f"{p:.0f}%", va="center", fontsize=9)
    axA.set_yticks(ys)
    axA.set_yticklabels([az.MODEL_INFO[m]["label"] for m, _ in rowsA])
    axA.invert_yaxis()
    axA.set_xlim(0, max(p for _, p in rowsA) + 10)
    axA.set_xlabel("Answer-level unsafe rate (%), both suites pooled")
    axA.set_title("(a) Cross-family audit", weight="bold")
    axA.grid(axis="y", visible=False)
    seen = {}
    for m, _ in rowsA:
        n, c = cls(m)
        seen[n] = c
    axA.legend(handles=[Patch(color=c, label=n) for n, c in seen.items()],
               frameon=False, fontsize=8, loc="upper right")

    # ---- (b) Qwen3 same-weights toggle, pooled ----
    pooled = sorted((r for r in at.build_rows(data) if r["dataset"] == "pooled"),
                    key=lambda r: r["params"])
    xs = [r["params"] for r in pooled]
    xlab = [f"{r['size'].rstrip('b')}B" for r in pooled]
    pn = [r["nothink"] for r in pooled]
    pt = [r["think"] for r in pooled]
    nerr = [[r["nothink"] - r["nlo"] for r in pooled], [r["nhi"] - r["nothink"] for r in pooled]]
    terr = [[r["think"] - r["tlo"] for r in pooled], [r["thi"] - r["think"] for r in pooled]]
    axB.fill_between(xs, pn, pt, color=az.OI["vermillion"], alpha=0.12, zorder=1)
    axB.errorbar(xs, pn, yerr=nerr, marker="o", color=az.OI["blue"], capsize=3, lw=2,
                 zorder=3, label="Thinking OFF")
    axB.errorbar(xs, pt, yerr=terr, marker="s", color=az.OI["vermillion"], capsize=3, lw=2,
                 zorder=3, label="Thinking ON")
    for r in pooled:
        cap = max(r["nhi"], r["thi"])
        axB.annotate(f"+{r['tax']:.1f}", (r["params"], cap + 1.8), ha="center",
                     va="bottom", fontsize=9, color=az.OI["vermillion"], zorder=5,
                     bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
    axB.set_xscale("log")
    axB.set_xticks(xs)
    axB.set_xticklabels(xlab)
    axB.xaxis.set_minor_formatter(NullFormatter())
    axB.minorticks_off()
    axB.set_xlim(0.5, 9.5)
    axB.set_ylim(0, max(r["thi"] for r in pooled) + 8)
    axB.set_xlabel("Qwen3 size (params, B; log scale)")
    axB.set_ylabel("Answer-level unsafe rate (%)")
    axB.set_title("(b) Qwen3 same-weights reasoning toggle", weight="bold")
    axB.legend(frameon=False, loc="upper right")
    axB.grid(which="major", alpha=0.3)

    fig.savefig("paper/figs/fig1_combined.png")
    plt.close(fig)
    print("wrote paper/figs/fig1_combined.png")


if __name__ == "__main__":
    main()
