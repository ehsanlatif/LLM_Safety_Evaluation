#!/usr/bin/env python3
"""Plot GRPO training curves from a run's metrics.jsonl.

Usage: plot_grpo.py <RL/experiments/<run>>   # writes <run>/plots/grpo_training.png
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

run = Path(sys.argv[1])
rows = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines() if l.strip()]
tr = [r for r in rows if "train/reward/mean" in r]
ev = [r for r in rows if "eval/avg/safe_rate" in r]

st = [r["global_step"] for r in tr]
rew = [r["train/reward/mean"] for r in tr]
ln = [r["train/response-length/mean"] for r in tr]
kl = [r["train/kl"] for r in tr]
gn = [r["train/grad_norm"] for r in tr]


def smooth(x, k=10):
    x = np.array(x, float)
    return [float(x[max(0, i - k + 1):i + 1].mean()) for i in range(len(x))]


es = [r["global_step"] for r in ev]
e_avg = [r["eval/avg/safe_rate"] for r in ev]
e_sorry = [r.get("eval/sorry_bench/safe_rate") for r in ev]
e_ailu = [r.get("eval/ailuminate/safe_rate") for r in ev]

fig, ax = plt.subplots(2, 2, figsize=(13, 8))
fig.suptitle(f"GRPO safety fine-tuning — VibeThinker-3B\n{run.name}", fontsize=11)

ax[0, 0].plot(st, rew, color="#bbb", lw=0.8, label="per-step")
ax[0, 0].plot(st, smooth(rew), color="#1f77b4", lw=2, label="10-step MA")
ax[0, 0].set_title("Train reward (Opus, harmful=safety + benign=helpfulness)")
ax[0, 0].set_xlabel("iteration"); ax[0, 0].set_ylabel("mean reward"); ax[0, 0].legend(); ax[0, 0].grid(alpha=.3)

ax[0, 1].plot(es, e_avg, "o-", color="#2ca02c", lw=2, label="avg safe-rate")
if any(x is not None for x in e_sorry):
    ax[0, 1].plot(es, e_sorry, "s--", color="#d62728", alpha=.8, label="SORRY-Bench")
if any(x is not None for x in e_ailu):
    ax[0, 1].plot(es, e_ailu, "^--", color="#9467bd", alpha=.8, label="AILuminate")
best = max(ev, key=lambda r: r["eval/avg/safe_rate"])
ax[0, 1].axvline(best["global_step"], color="k", ls=":", alpha=.6)
ax[0, 1].annotate(f"best {best['eval/avg/safe_rate']:.3f}\n@step {best['global_step']}",
                  (best["global_step"], best["eval/avg/safe_rate"]),
                  textcoords="offset points", xytext=(8, -28), fontsize=9)
ax[0, 1].set_title("In-loop val safe-rate (safety_val_small, greedy)")
ax[0, 1].set_xlabel("iteration"); ax[0, 1].set_ylabel("safe-rate"); ax[0, 1].legend(); ax[0, 1].grid(alpha=.3)

ax[1, 0].plot(st, ln, color="#bbb", lw=0.8)
ax[1, 0].plot(st, smooth(ln), color="#ff7f0e", lw=2)
ax[1, 0].set_title("Response length (tokens) — watch for length collapse")
ax[1, 0].set_xlabel("iteration"); ax[1, 0].set_ylabel("mean tokens"); ax[1, 0].grid(alpha=.3)

ax2 = ax[1, 1]
ax2.plot(st, kl, color="#1f77b4", lw=1.5, label="KL(policy‖ref)")
ax2.set_xlabel("iteration"); ax2.set_ylabel("KL", color="#1f77b4"); ax2.grid(alpha=.3)
ax3 = ax2.twinx()
ax3.plot(st, gn, color="#e377c2", lw=1, alpha=.6, label="grad-norm")
ax3.set_ylabel("grad-norm", color="#e377c2")
ax2.set_title("KL-to-reference & grad-norm (stability)")
ax2.legend(loc="upper left"); ax3.legend(loc="upper right")

plt.tight_layout(rect=[0, 0, 1, 0.95])
out = run / "plots"; out.mkdir(exist_ok=True)
path = out / "grpo_training.png"
plt.savefig(path, dpi=130)
print(f"saved {path}")
