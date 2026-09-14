#!/usr/bin/env python3
"""Paired analysis of the generation-averaging eval (RL/gen_average_eval.py outputs).

Per model, per prompt: unsafe *probability* over the K samples (answer-level over covered
samples; anywhere over all K). Model rate = mean over prompts. Because every model saw the
SAME prompts, we compare each model to baseline PAIRED on prompts (removes prompt-difficulty
variance -> much more power) with a paired bootstrap p-value + CI on the mean per-prompt
difference. Bootstrap CIs (over prompts) for each rate too.

Writes results/figs_rl/fig4_genavg.png and prints the table.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "results" / "genavg"
OUT = ROOT / "results" / "figs_rl"; OUT.mkdir(parents=True, exist_ok=True)
MODELS = [("base", "Baseline"), ("es_noawd", "ES (no-AWD)"), ("es_awd", "ES (+AWD)"), ("grpo", "GRPO")]
rng = np.random.default_rng(0)
B = 5000


def per_prompt(tag):
    """-> dict id -> {'any': frac_unsafe_over_K, 'ans': frac_or_None, 'K':k}."""
    recs = [json.loads(l) for l in (GEN / f"{tag}.jsonl").read_text().splitlines() if l.strip()]
    by = defaultdict(lambda: {"any": [], "ans": []})
    for r in recs:
        by[r["id"]]["any"].append(1 if r["anywhere_label"] == "unsafe" else 0)
        if r["answer_label"] is not None:
            by[r["id"]]["ans"].append(1 if r["answer_label"] == "unsafe" else 0)
    out = {}
    for i, d in by.items():
        out[i] = {"any": float(np.mean(d["any"])) if d["any"] else 0.0,
                  "ans": (float(np.mean(d["ans"])) if d["ans"] else None),
                  "K": len(d["any"])}
    return out


data = {tag: per_prompt(tag) for tag, _ in MODELS if (GEN / f"{tag}.jsonl").exists()}
ids = sorted(set.intersection(*[set(d) for d in data.values()]))
print(f"prompts (common across models): {len(ids)} ; K≈{data['base'][ids[0]]['K']}")


def rate_ci(vals):
    vals = np.array([v for v in vals if v is not None], float)
    boot = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(B)]
    return vals.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def paired_p(model_vals, base_vals):
    d = np.array([m - b for m, b in zip(model_vals, base_vals)], float)
    boot = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(B)])
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return d.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5), min(p, 1.0)


base_any = [data["base"][i]["any"] for i in ids]
base_ans_ids = [i for i in ids if data["base"][i]["ans"] is not None]

print(f"\n{'model':13} {'answer% (CI)':>22} {'anywhere% (CI)':>24}   paired Δanywhere vs base (CI, p)")
rows = {}
for tag, name in MODELS:
    if tag not in data:
        continue
    any_vals = [data[tag][i]["any"] for i in ids]
    ans_vals = [data[tag][i]["ans"] for i in ids if data[tag][i]["ans"] is not None]
    ar, alo, ahi = rate_ci(ans_vals)
    yr, ylo, yhi = rate_ci(any_vals)
    rows[tag] = dict(name=name, ans=(ar, alo, ahi), any=(yr, ylo, yhi))
    if tag == "base":
        extra = "—"
    else:
        md, lo, hi, p = paired_p(any_vals, base_any)
        extra = f"{md*100:+5.1f} pp [{lo*100:+.1f},{hi*100:+.1f}] p={p:.3f}"
        rows[tag]["dp"] = (md, lo, hi, p)
    print(f"{name:13} {ar*100:5.1f} [{alo*100:4.1f}-{ahi*100:4.1f}] "
          f"{yr*100:6.1f} [{ylo*100:4.1f}-{yhi*100:4.1f}]    {extra}")

# ---- figure: anywhere rate with bootstrap CI + significance stars ----
names = [n for t, n in MODELS if t in rows]
tags = [t for t, _ in MODELS if t in rows]
yr = [rows[t]["any"][0] * 100 for t in tags]
err = [[(rows[t]["any"][0] - rows[t]["any"][1]) * 100 for t in tags],
       [(rows[t]["any"][2] - rows[t]["any"][0]) * 100 for t in tags]]
colors = ["#7f7f7f", "#ff9896", "#d62728", "#1f77b4"][:len(tags)]
fig, ax = plt.subplots(figsize=(8.5, 5))
ax.bar(names, yr, yerr=err, capsize=5, color=colors)
for i, t in enumerate(tags):
    ax.text(i, yr[i] + err[1][i] + 0.4, f"{yr[i]:.1f}", ha="center", fontsize=9)
    if "dp" in rows[t]:
        p = rows[t]["dp"][3]
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.text(i, -1.2, f"p={p:.3f}\n{star}", ha="center", fontsize=8,
                color="#1a7f37" if p < 0.05 else "#888")
K = data["base"][ids[0]]["K"]
ax.set_ylabel("anywhere unsafe-rate %  (↓ safer)")
ax.set_title(f"Generation-averaged safety (K={K} samples/prompt @ temp 0.6, n={len(ids)})\n"
             f"paired-on-prompt vs baseline; bootstrap 95% CI")
ax.grid(axis="y", alpha=.3); ax.set_ylim(bottom=-2)
plt.tight_layout(); plt.savefig(OUT / "fig4_genavg.png", dpi=130)
print(f"\nsaved {OUT/'fig4_genavg.png'}")
