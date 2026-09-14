#!/usr/bin/env python3
"""Aggregate the final GRPO vs baseline (vs ES) eval into a table + figures.

Reads results/<tag>__safety_test.summary.json, __reasoning_probe_200, __benign_probe.
Writes results/figs_rl/fig1_safety.png, fig2_capability.png and prints the table.
"""
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "figs_rl"; OUT.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("vibe_base_v2", "Baseline"),
    ("vibe_es_noawd", "ES (no-AWD)"),
    ("vibe_es_awd", "ES (+AWD)"),
    ("vibe_grpo", "GRPO"),
]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def two_prop_z(k1, n1, k2, n2):
    if min(n1, n2) == 0:
        return float("nan"), float("nan")
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p2 - p1) / se
    from math import erfc
    pval = erfc(abs(z) / math.sqrt(2))
    return z, pval


def load(tag):
    d = {}
    st = json.load(open(RES / f"{tag}__safety_test.summary.json"))["summaries"]
    ans_k = ans_n = any_k = 0
    per = {}
    for split in ("sorry_bench", "ailuminate"):
        s = st[split]
        ku = round(s["answer_unsafe_rate"] * s["answer_n"])
        au = round(s["anywhere_unsafe_rate"] * s["n"])
        per[split] = (s["answer_unsafe_rate"], ku, s["answer_n"], au, s["n"])
        ans_k += ku; ans_n += s["answer_n"]; any_k += au
    d["per"] = per
    d["ans"] = (ans_k, ans_n)          # pooled answer-level (covered subset)
    d["any"] = (any_k, 200)            # pooled anywhere over full n=200
    d["reason"] = json.load(open(RES / f"{tag}__reasoning_probe_200.summary.json"))[
        "summaries"]["reasoning_probe_200"]["accuracy"]
    bp = json.load(open(RES / f"{tag}__benign_probe.summary.json"))["summaries"]["benign_probe"]
    d["benign"] = bp["answer_unsafe_rate"]
    return d


data = {tag: load(tag) for tag, _ in MODELS}
base = data["vibe_base_v2"]

print(f"\n{'model':14} {'SORRY ans%':>11} {'AILu ans%':>10} {'POOL ans%':>18} {'anywhere%':>16} {'MATH':>6} {'benign%':>8}")
for tag, name in MODELS:
    d = data[tag]
    ak, an = d["ans"]; yk, yn = d["any"]
    pa, la, ha = wilson(ak, an); py, ly, hy = wilson(yk, yn)
    s = d["per"]["sorry_bench"][0] * 100
    a = d["per"]["ailuminate"][0] * 100
    zc = ""
    if tag != "vibe_base_v2":
        bz, bp_ = two_prop_z(*base["ans"], ak, an)
        yz, yp_ = two_prop_z(*base["any"], yk, yn)
        zc = f"  [ans p={bp_:.3f}, any p={yp_:.3f} vs base]"
    print(f"{name:14} {s:10.1f}  {a:9.1f}  {pa*100:6.1f} [{la*100:4.1f}-{ha*100:4.1f}]  "
          f"{py*100:5.1f} [{ly*100:4.1f}-{hy*100:4.1f}]  {d['reason']*100:5.1f}  {d['benign']*100:6.1f}{zc}")
print()

# ---- fig1: safety (pooled answer + anywhere), baseline vs GRPO (+ES context) ----
names = [n for _, n in MODELS]
ans_p, ans_lo, ans_hi, any_p, any_lo, any_hi = [], [], [], [], [], []
for tag, _ in MODELS:
    ak, an = data[tag]["ans"]; yk, yn = data[tag]["any"]
    p, lo, hi = wilson(ak, an); ans_p.append(p * 100); ans_lo.append((p - lo) * 100); ans_hi.append((hi - p) * 100)
    p, lo, hi = wilson(yk, yn); any_p.append(p * 100); any_lo.append((p - lo) * 100); any_hi.append((hi - p) * 100)

x = np.arange(len(names)); w = 0.38
colors = ["#7f7f7f", "#ff9896", "#d62728", "#1f77b4"]
fig, ax = plt.subplots(figsize=(9, 5.2))
ax.bar(x - w / 2, ans_p, w, yerr=[ans_lo, ans_hi], capsize=4, color=colors, label="answer-level (covered)")
ax.bar(x + w / 2, any_p, w, yerr=[any_lo, any_hi], capsize=4, color=colors, alpha=0.45, hatch="//",
       label="anywhere (n=200)")
for i, v in enumerate(ans_p): ax.text(i - w / 2, v + 0.4, f"{v:.1f}", ha="center", fontsize=8)
for i, v in enumerate(any_p): ax.text(i + w / 2, v + 0.4, f"{v:.1f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(names)
ax.set_ylabel("unsafe-rate %  (↓ safer)")
ax.set_title("Plain-chat safety on untouched test split (Opus judge, greedy)\nGRPO best@step150 vs baseline vs ES")
ax.legend(); ax.grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig(OUT / "fig1_safety.png", dpi=130); print(f"saved {OUT/'fig1_safety.png'}")

# ---- fig2: capability ----
reason = [data[t]["reason"] * 100 for t, _ in MODELS]
benign = [data[t]["benign"] * 100 for t, _ in MODELS]
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
ax[0].bar(names, reason, color=colors); ax[0].set_title("MATH-200 accuracy (↑ better)")
ax[0].set_ylim(80, 100); ax[0].grid(axis="y", alpha=.3)
for i, v in enumerate(reason): ax[0].text(i, v + 0.2, f"{v:.1f}", ha="center", fontsize=9)
ax[1].bar(names, benign, color=colors); ax[1].set_title("Benign over-refusal unsafe% (↓ better)")
ax[1].set_ylim(0, 5); ax[1].grid(axis="y", alpha=.3)
for i, v in enumerate(benign): ax[1].text(i, v + 0.1, f"{v:.1f}", ha="center", fontsize=9)
for a in ax: a.tick_params(axis="x", rotation=20)
plt.tight_layout(); plt.savefig(OUT / "fig2_capability.png", dpi=130); print(f"saved {OUT/'fig2_capability.png'}")
