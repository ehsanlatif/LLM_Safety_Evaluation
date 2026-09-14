#!/usr/bin/env python3
"""
Finalize the Qwen3 reasoning-toggle study: the size x reasoning result.

Because each size is ONE set of weights run with thinking OFF vs ON, reasoning is
the only variable — this is the clean ablation the cross-family models can't give.
For each size we report the ANSWER-LEVEL unsafe rate (primary metric, from
grade_responses.py's split schema) with 95% Wilson CIs, the reasoning "safety tax"
(think - nothink, in percentage points), the thinking variant's answer coverage
(truncation gap), and its CoT-level unsafe rate. Per dataset and pooled.

Reads : results/graded/qwen3-*__*.jsonl  (via analyze.load)
Writes: analysis/qwen3_toggle_summary.csv
        analysis/fig_qwen3_toggle.png        (unsafe vs size: thinking vs non-thinking)
        analysis/fig_qwen3_reasoning_tax.png  (the think-nothink gap vs size)
"""
from __future__ import annotations

import csv

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

import analyze as az  # reuse load(), count(), wilson(), coverage(), OUT, OI, DS_LABEL

OUT = az.OUT
OI = az.OI
DS_LABEL = dict(az.DS_LABEL, pooled="Both datasets pooled")
SIZES = [("0.6b", 0.6), ("1.7b", 1.7), ("4b", 4.0), ("8b", 8.0)]
DATASETS = ["ailuminate", "sorry-bench"]


def rate(recs, channel):
    u, t = az.count(recs, channel)
    pct = 100 * u / t if t else float("nan")
    lo, hi = az.wilson(u, t)
    return pct, lo, hi, t


def build_rows(data):
    rows = []
    for size, params in SIZES:
        nm, tm = f"qwen3-{size}-nothink", f"qwen3-{size}-think"
        for ds in DATASETS + ["pooled"]:
            if ds == "pooled":
                rn = data[nm]["ailuminate"] + data[nm]["sorry-bench"]
                rt = data[tm]["ailuminate"] + data[tm]["sorry-bench"]
            else:
                rn, rt = data[nm][ds], data[tm][ds]
            pn, nlo, nhi, nn = rate(rn, "answer")
            pt, tlo, thi, tn = rate(rt, "answer")
            cotp, _, _, _ = rate(rt, "cot")
            rows.append(dict(size=size, params=params, dataset=ds,
                             nothink=pn, nlo=nlo, nhi=nhi, nn=nn,
                             think=pt, tlo=tlo, thi=thi, tn=tn,
                             tax=pt - pn, cov=az.coverage(rt), cot=cotp))
    return rows


def write_csv(rows):
    with (OUT / "qwen3_toggle_summary.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["size", "params_B", "dataset",
                    "nothink_unsafe_pct", "nothink_ci_lo", "nothink_ci_hi", "nothink_n",
                    "think_unsafe_pct", "think_ci_lo", "think_ci_hi", "think_n",
                    "reasoning_tax_pp", "think_answer_coverage_pct", "think_cot_unsafe_pct"])
        for r in rows:
            w.writerow([r["size"], r["params"], r["dataset"],
                        f'{r["nothink"]:.2f}', f'{r["nlo"]:.2f}', f'{r["nhi"]:.2f}', r["nn"],
                        f'{r["think"]:.2f}', f'{r["tlo"]:.2f}', f'{r["thi"]:.2f}', r["tn"],
                        f'{r["tax"]:+.2f}', f'{r["cov"]:.1f}', f'{r["cot"]:.2f}'])


def _size_axis(ax, xs, xlab):
    """Discrete size ticks on a log axis, with the colliding minor labels off."""
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(xlab)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.minorticks_off()
    ax.set_xlim(0.5, 9.5)
    ax.set_xlabel("Model size (billions of parameters, log scale)")


def fig_lines(rows):
    xs = [p for _, p in SIZES]
    xlab = [s.rstrip("b") + "B" for s, _ in SIZES]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), sharey=True)
    # shared headroom so the gap labels (placed above the upper CI cap) never clip
    top = max(max(r["nhi"], r["thi"]) for r in rows if r["dataset"] in DATASETS)
    for ax, ds in zip(axes, DATASETS):
        sub = sorted((r for r in rows if r["dataset"] == ds), key=lambda r: r["params"])
        pn = [r["nothink"] for r in sub]
        pt = [r["think"] for r in sub]
        nerr = [[r["nothink"] - r["nlo"] for r in sub], [r["nhi"] - r["nothink"] for r in sub]]
        terr = [[r["think"] - r["tlo"] for r in sub], [r["thi"] - r["think"] for r in sub]]
        ax.fill_between(xs, pn, pt, color=OI["vermillion"], alpha=0.12, zorder=1)
        ax.errorbar(xs, pn, yerr=nerr, marker="o", color=OI["blue"], capsize=3, lw=2,
                    zorder=3, label="Non-thinking")
        ax.errorbar(xs, pt, yerr=terr, marker="s", color=OI["vermillion"], capsize=3, lw=2,
                    zorder=3, label="Thinking")
        # gap label sits above the upper error-bar cap, bottom-anchored, with a
        # faint white halo so it stays legible even when close to a line.
        for r in sub:
            cap = max(r["nhi"], r["thi"])
            ax.annotate(f"+{r['tax']:.1f}", (r["params"], cap + 2.4), ha="center",
                        va="bottom", fontsize=9, color=OI["vermillion"], zorder=5,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                                  alpha=0.7))
        _size_axis(ax, xs, xlab)
        ax.set_title(DS_LABEL.get(ds, ds), weight="bold")
        ax.grid(True, which="major", alpha=0.3)
    axes[0].set_ylim(0, top + 9)
    axes[0].set_ylabel("Answer-level unsafe rate (%) — lower is safer")
    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle("Qwen3 same-weights toggle: thinking raises the unsafe rate at every size\n"
                 "(shaded = reasoning safety tax; it shrinks as the model scales)",
                 y=1.07, fontsize=13, weight="bold")
    fig.savefig(OUT / "fig_qwen3_toggle.png", bbox_inches="tight"); plt.close(fig)


def fig_tax(rows):
    xs = [p for _, p in SIZES]
    xlab = [s.rstrip("b") + "B" for s, _ in SIZES]
    fig, ax = plt.subplots(figsize=(7.8, 5))
    for ds, color, mk in [("ailuminate", OI["blue"], "o"),
                          ("sorry-bench", OI["green"], "s"),
                          ("pooled", OI["vermillion"], "D")]:
        sub = sorted((r for r in rows if r["dataset"] == ds), key=lambda r: r["params"])
        ax.plot(xs, [r["tax"] for r in sub], marker=mk, color=color, lw=2,
                label=DS_LABEL.get(ds, ds))
    ax.axhline(0, color="#888", lw=1)
    _size_axis(ax, xs, xlab)
    ax.set_ylabel("Reasoning safety tax (pp) = thinking − non-thinking")
    ax.set_title("Reasoning penalty vs scale", weight="bold")
    ax.legend(frameon=False); ax.grid(which="major", alpha=0.3)
    fig.savefig(OUT / "fig_qwen3_reasoning_tax.png", bbox_inches="tight"); plt.close(fig)


def print_table(rows):
    print("\n" + "=" * 70)
    print("QWEN3 SIZE x REASONING — answer-level unsafe% (both datasets pooled)")
    print("=" * 70)
    print(f"{'size':6s} {'non-think':>11s} {'think':>11s} {'tax(pp)':>9s} {'think cov':>10s}")
    for size, _ in SIZES:
        r = next(x for x in rows if x["size"] == size and x["dataset"] == "pooled")
        print(f"{size:6s} {r['nothink']:10.1f}% {r['think']:10.1f}% "
              f"{r['tax']:+8.1f} {r['cov']:9.1f}%")


def main():
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 140,
                         "savefig.dpi": 200, "savefig.bbox": "tight"})
    rows = build_rows(az.load())
    write_csv(rows)
    fig_lines(rows)
    fig_tax(rows)
    print_table(rows)
    print(f"\nWrote qwen3_toggle_summary.csv + 2 figures to {OUT}/")


if __name__ == "__main__":
    main()
