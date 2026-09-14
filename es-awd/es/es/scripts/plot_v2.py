#!/usr/bin/env python3
"""Plots for experiment v2 (Opus reward, balanced data, AWD on/off).
Reads both runs' metrics.jsonl + the results/*.summary.json eval files and writes
PNG figures to results/figs_v2/. CVD-safe Okabe-Ito palette; recessive grid."""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIGS = RES / "figs_v2"
FIGS.mkdir(parents=True, exist_ok=True)

# Okabe-Ito (colorblind-safe)
C = {"base": "#5A5A5A", "noawd": "#0072B2", "awd": "#D55E00",
     "sorry": "#0072B2", "ailu": "#009E73", "pooled": "#CC79A7", "grid": "#D9D9D9"}
INK, MUTED = "#222222", "#666666"

def _newest(pattern):
    dirs = sorted((ROOT / "experiments").glob(pattern), key=lambda p: p.name)
    if not dirs:
        raise FileNotFoundError(pattern)
    return dirs[-1]  # timestamp is the trailing sort key -> newest last

NOAWD_DIR = _newest("es-safety-*opus-noawd-*")
AWD_DIR = _newest("es-safety-*opus-awdl210-*")

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": "#BBBBBB", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.7,
    "axes.axisbelow": True, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "figure.dpi": 130,
})


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def load_metrics(d):
    train, ev = [], []
    for line in (d / "metrics.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "train/reward/mean" in r:
            train.append(r)
        if "eval/avgunsafe_rate" in r:
            ev.append(r)
    return train, ev


def safety(tag):
    s = json.loads((RES / f"{tag}__safety_test.summary.json").read_text())["summaries"]
    sb, al = s["sorry_bench"], s["ailuminate"]
    su, sn = sb["answer_unsafe_rate"], sb["answer_n"]
    au, an = al["answer_unsafe_rate"], al["answer_n"]
    k, n = round(su * sn) + round(au * an), sn + an
    pooled, plo, phi = wilson(k, n)
    return {"sorry": (su, sb["answer_ci"]), "ailu": (au, al["answer_ci"]),
            "pooled": (pooled, [plo, phi])}


def reasoning(tag):
    for ds in ("reasoning_probe_200", "reasoning_probe"):
        f = RES / f"{tag}__{ds}.summary.json"
        if f.is_file():
            return json.loads(f.read_text())["summaries"][ds]["accuracy"]
    raise FileNotFoundError(f"no reasoning summary for {tag}")


def benign(tag):
    s = json.loads((RES / f"{tag}__benign_probe.summary.json").read_text())["summaries"]
    return s[next(iter(s))]["answer_unsafe_rate"]


MODELS = [("baseline", "vibe_base_v2", C["base"]),
          ("ES no-AWD", "vibe_es_noawd", C["noawd"]),
          ("ES +AWD", "vibe_es_awd", C["awd"])]


# ---------------------------------------------------------------- Fig 1: headline safety
def fig_safety():
    groups = ["SORRY-Bench", "AILuminate", "Pooled"]
    keys = ["sorry", "ailu", "pooled"]
    data = {tag: safety(tag) for _l, tag, _c in MODELS}
    x = np.arange(len(groups))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for i, (label, tag, col) in enumerate(MODELS):
        vals = [data[tag][k][0] * 100 for k in keys]
        los = [(data[tag][k][0] - data[tag][k][1][0]) * 100 for k in keys]
        his = [(data[tag][k][1][1] - data[tag][k][0]) * 100 for k in keys]
        xi = x + (i - 1) * w
        bars = ax.bar(xi, vals, w, color=col, label=label, zorder=3)
        ax.errorbar(xi, vals, yerr=[los, his], fmt="none", ecolor="#444444",
                    elinewidth=1.1, capsize=3, zorder=4)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + max(his) + 1.2, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("Answer-level unsafe %  (↓ safer)")
    ax.set_title("Plain-chat safety on the untouched test split — Claude-Opus judge (95% Wilson CI)",
                 fontsize=11.5, color=INK)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.10))
    ax.set_ylim(0, 40); ax.margins(x=0.02)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_safety_before_after.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 2: capability
def fig_capability():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 4.2))
    labels = [l for l, _t, _c in MODELS]
    cols = [c for _l, _t, c in MODELS]
    acc = [reasoning(t) * 100 for _l, t, _c in MODELS]
    a1.bar(labels, acc, color=cols, zorder=3, width=0.6)
    for i, v in enumerate(acc):
        a1.text(i, v + 0.6, f"{v:.0f}%", ha="center", va="bottom", fontsize=10, color=INK)
    a1.set_ylabel("MATH-50 accuracy %  (↑ reasoning kept)")
    a1.set_title("Reasoning (capability)", fontsize=11)
    a1.set_ylim(80, 100)
    ben = [benign(t) * 100 for _l, t, _c in MODELS]
    a2.bar(labels, ben, color=cols, zorder=3, width=0.6)
    for i, v in enumerate(ben):
        a2.text(i, v + 0.3, f"{v:.0f}%", ha="center", va="bottom", fontsize=10, color=INK)
    a2.set_ylabel("Benign unsafe %  (over-refusal proxy)")
    a2.set_title("Benign over-refusal probe", fontsize=11)
    a2.set_ylim(0, 10)
    for ax in (a1, a2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="x", labelrotation=12)
    fig.suptitle("Capability preserved? — reasoning & over-refusal", fontsize=12.5, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_capability.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 3: training dynamics
def fig_dynamics():
    tn, en = load_metrics(NOAWD_DIR)
    ta, ea = load_metrics(AWD_DIR)

    def series(rows, key):
        return [r["global_step"] for r in rows], [r[key] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    # (a) reward mean ± std
    for rows, col, lab in [(tn, C["noawd"], "no-AWD"), (ta, C["awd"], "+AWD")]:
        xs, ys = series(rows, "train/reward/mean")
        sd = [r["train/reward/std"] for r in rows]
        axes[0].plot(xs, ys, color=col, lw=2, label=lab, zorder=3)
        axes[0].fill_between(xs, np.array(ys) - np.array(sd), np.array(ys) + np.array(sd),
                             color=col, alpha=0.12, zorder=2)
    axes[0].set_title("Train reward (mean ± std)", fontsize=11)
    axes[0].set_xlabel("iteration"); axes[0].set_ylabel("granular reward  (0–1)")
    axes[0].legend(frameon=False, loc="lower right")
    # (b) response length
    for rows, col, lab in [(tn, C["noawd"], "no-AWD"), (ta, C["awd"], "+AWD")]:
        xs, ys = series(rows, "train/response-length/mean")
        axes[1].plot(xs, ys, color=col, lw=2, label=lab, zorder=3)
    axes[1].set_title("Mean response length", fontsize=11)
    axes[1].set_xlabel("iteration"); axes[1].set_ylabel("tokens")
    axes[1].legend(frameon=False, loc="upper right")
    # (c) in-loop val unsafe-rate
    for rows, col, lab in [(en, C["noawd"], "no-AWD"), (ea, C["awd"], "+AWD")]:
        xs = [r["global_step"] for r in rows]
        ys = [r["eval/avgunsafe_rate"] * 100 for r in rows]
        axes[2].plot(xs, ys, color=col, lw=2, marker="o", ms=7, label=lab, zorder=3)
    axes[2].set_title("In-loop val unsafe % (small val)", fontsize=11)
    axes[2].set_xlabel("iteration"); axes[2].set_ylabel("val unsafe %")
    axes[2].legend(frameon=False, loc="upper right")
    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Training dynamics — Opus granular reward, teacher-shaping 0", fontsize=12.5, y=1.03)
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_training_dynamics.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_safety()
    fig_capability()
    fig_dynamics()
    print(f"wrote figures to {FIGS}")
    for p in sorted(FIGS.glob("*.png")):
        print(" ", p.name)
