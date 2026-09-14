#!/usr/bin/env python3
"""Dedicated training + validation plots for both runs (no-AWD vs +AWD):
reward, response length (raw + rolling mean), aggregate val unsafe%, and per-benchmark
val unsafe%. Writes results/figs_v2/fig4_train_val.png."""
import json, glob
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "results" / "figs_v2"; FIGS.mkdir(parents=True, exist_ok=True)
NO = {"c": "#0072B2", "lab": "no-AWD"}; AW = {"c": "#D55E00", "lab": "+AWD"}
INK, MUTED, GRID = "#222222", "#666666", "#D9D9D9"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#BBBBBB", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "figure.dpi": 130})


def newest(pat): return sorted(glob.glob(str(ROOT / "experiments" / pat)))[-1]
def load(pat):
    tr, ev = [], []
    for l in open(Path(newest(pat)) / "metrics.jsonl"):
        r = json.loads(l)
        if "train/reward/mean" in r: tr.append(r)
        if "eval/avgunsafe_rate" in r: ev.append(r)
    return tr, ev
def roll(y, w=9):
    y = np.asarray(y, float); h = w // 2; out = np.empty_like(y)
    for i in range(len(y)):                       # shrinking window at edges (no boundary dip)
        out[i] = y[max(0, i - h):min(len(y), i + h + 1)].mean()
    return out

trN, evN = load("es-safety-*opus-noawd-*")
trA, evA = load("es-safety-*opus-awdl210-*")
def col(tr, key): return [r["global_step"] for r in tr], [r[key] for r in tr]

fig, ax = plt.subplots(2, 2, figsize=(12.5, 8))

# (a) train reward: raw faint + rolling bold -> shows early rise then noisy plateau
for tr, s in [(trN, NO), (trA, AW)]:
    x, y = col(tr, "train/reward/mean")
    ax[0,0].plot(x, y, color=s["c"], lw=0.8, alpha=0.22)
    ax[0,0].plot(x, roll(y), color=s["c"], lw=2.2, label=s["lab"])
ax[0,0].set_title("Train reward — raw (faint) + rolling mean (bold)", fontsize=11)
ax[0,0].set_xlabel("iteration"); ax[0,0].set_ylabel("granular reward (0–1)")
ax[0,0].legend(frameon=False, loc="lower right")
ax[0,0].annotate("coherent rise\n(easy refusals learned)", xy=(25, 0.775), xytext=(60, 0.70),
    fontsize=8.5, color=MUTED, arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
ax[0,0].annotate("noisy plateau\n(random walk near optimum)", xy=(220, 0.805), xytext=(150, 0.74),
    fontsize=8.5, color=MUTED, arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))

# (b) response length: raw + rolling -> length collapse, AWD retains more
for tr, s in [(trN, NO), (trA, AW)]:
    x, y = col(tr, "train/response-length/mean")
    ax[0,1].plot(x, y, color=s["c"], lw=0.8, alpha=0.22)
    ax[0,1].plot(x, roll(y), color=s["c"], lw=2.2, label=s["lab"])
ax[0,1].set_title("Mean response length — length collapse (AWD milder)", fontsize=11)
ax[0,1].set_xlabel("iteration"); ax[0,1].set_ylabel("tokens")
ax[0,1].legend(frameon=False, loc="upper right")

# (c) aggregate val unsafe% -> best (lowest) at iter 150 for both
for ev, s in [(evN, NO), (evA, AW)]:
    x = [r["global_step"] for r in ev]; y = [r["eval/avgunsafe_rate"] * 100 for r in ev]
    ax[1,0].plot(x, y, color=s["c"], lw=2, marker="o", ms=7, label=s["lab"])
ax[1,0].axvline(150, color=MUTED, ls="--", lw=1)
ax[1,0].text(152, ax[1,0].get_ylim()[1]*0.96, "best val @150\n(over-optimization after)", fontsize=8.5,
    color=MUTED, va="top")
ax[1,0].set_title("Validation unsafe % (held-out, mean model θ)", fontsize=11)
ax[1,0].set_xlabel("iteration"); ax[1,0].set_ylabel("val unsafe %")
ax[1,0].legend(frameon=False, loc="lower right")

# (d) per-benchmark val unsafe% -> AILuminate drives the late rise
for ev, s in [(evN, NO), (evA, AW)]:
    x = [r["global_step"] for r in ev]
    ax[1,1].plot(x, [r["eval/sorry_bench/unsafe_rate"]*100 for r in ev], color=s["c"], lw=2, marker="o", ms=6, label=f"{s['lab']} · SORRY")
    ax[1,1].plot(x, [r["eval/ailuminate/unsafe_rate"]*100 for r in ev], color=s["c"], lw=2, ls="--", marker="^", ms=6, label=f"{s['lab']} · AILu")
ax[1,1].set_title("Validation unsafe % by benchmark", fontsize=11)
ax[1,1].set_xlabel("iteration"); ax[1,1].set_ylabel("val unsafe %")
ax[1,1].legend(frameon=False, fontsize=8.5, ncol=2, loc="upper left")

for a in ax.flat:
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
fig.suptitle("Training & validation dynamics — 300 iterations (no-AWD vs +AWD)", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(FIGS / "fig4_train_val.png", bbox_inches="tight")
print("wrote", FIGS / "fig4_train_val.png")
