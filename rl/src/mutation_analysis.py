#!/usr/bin/env python3
"""
Resilience-gap analysis for the 20 SORRY-Bench mutation styles.

resilience gap = fulfillment(style) - fulfillment(base). Positive = the linguistic
mutation jailbreaks the model beyond its base fulfillment. We report per-style
uplift, the mean and worst-case (max) uplift per model, and compare by model class
(the H-CoT question: are reasoning models more brittle under mutation?).

Reads : analysis/mutation_fulfillment.csv   (model x 20 styles)
        analysis/sorrybench_official_overall.csv  (base fulfillment)
Writes: analysis/resilience_gap.csv,
        analysis/fig_mutation_heatmap.png, analysis/fig_resilience_gap.png
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

AN = Path("analysis")
_Q3 = [f"qwen3-{s}-{m}" for s in ("0.6b", "1.7b", "4b", "8b") for m in ("nothink", "think")]
ORDER = ["opus-4.8", "sonnet-5", "gpt-5.5", "llama-3.2-3b", "qwen-3b", "qwen-1.5b",
         "deepseek-r1-1.5b", "vibethinker-1.5b", "vibethinker-3b"] + _Q3
LABEL = {"opus-4.8": "Opus 4.8", "sonnet-5": "Sonnet 5", "gpt-5.5": "GPT-5.5",
         "qwen-3b": "Qwen2.5-3B", "llama-3.2-3b": "Llama-3.2-3B", "qwen-1.5b": "Qwen2.5-1.5B",
         "vibethinker-1.5b": "VibeThinker-1.5B", "deepseek-r1-1.5b": "DeepSeek-R1-1.5B",
         "vibethinker-3b": "VibeThinker-3B"}
CLASS = {"opus-4.8": "Large closed", "sonnet-5": "Large closed", "gpt-5.5": "Large closed",
         "qwen-3b": "Small open (non-reasoning)", "llama-3.2-3b": "Small open (non-reasoning)",
         "qwen-1.5b": "Small open (non-reasoning)",
         "vibethinker-1.5b": "Small open (reasoning)", "deepseek-r1-1.5b": "Small open (reasoning)",
         "vibethinker-3b": "Small open (reasoning)"}
for _s in ("0.6b", "1.7b", "4b", "8b"):
    LABEL[f"qwen3-{_s}-nothink"] = f"Qwen3-{_s.upper()}"
    LABEL[f"qwen3-{_s}-think"] = f"Qwen3-{_s.upper()} (think)"
    CLASS[f"qwen3-{_s}-nothink"] = "Small open (non-reasoning)"
    CLASS[f"qwen3-{_s}-think"] = "Small open (reasoning)"
# OLMo 3 7B model-flow sweep (safety across post-training stages).
_OLMO_LABEL = {
    "olmo3-7b-base": "OLMo3-7B Base", "olmo3-7b-instruct-sft": "OLMo3-7B Instruct-SFT",
    "olmo3-7b-instruct-dpo": "OLMo3-7B Instruct-DPO", "olmo3-7b-instruct": "OLMo3-7B Instruct-RL",
    "olmo3-7b-think-sft": "OLMo3-7B Think-SFT", "olmo3-7b-think-dpo": "OLMo3-7B Think-DPO",
    "olmo3-7b-think-rl": "OLMo3-7B Think-RL", "olmo3-7b-rlzero-math": "OLMo3-7B RLZero-Math",
    "olmo3-7b-rlzero-code": "OLMo3-7B RLZero-Code", "olmo3-7b-rlzero-if": "OLMo3-7B RLZero-IF",
    "olmo3-7b-rlzero-general": "OLMo3-7B RLZero-General", "olmo3-7b-rlzero-mix": "OLMo3-7B RLZero-Mix",
}
ORDER = ORDER + list(_OLMO_LABEL)
LABEL.update(_OLMO_LABEL)
CLASS.update({m: "OLMo 3 7B" for m in _OLMO_LABEL})
CLASS_COLOR = {"Large closed": "#0072B2", "Small open (non-reasoning)": "#E69F00",
               "Small open (reasoning)": "#D55E00", "OLMo 3 7B": "#009E73"}
STYLES = ["slang", "uncommon_dialects", "technical_terms", "role_play", "misspellings", "question",
          "authority_endorsement", "misrepresentation", "logical_appeal",
          "evidence-based_persuasion", "expert_endorsement",
          "ascii", "caesar", "morse", "atbash",
          "translate-fr", "translate-ml", "translate-mr", "translate-ta", "translate-zh-cn"]
FAMILY = (["Writing"] * 6 + ["Persuasion"] * 5 + ["Cipher"] * 4 + ["Translation"] * 5)


def read_csv(p):
    return list(csv.DictReader(p.open())) if p.is_file() else []


def load():
    base = {}
    for r in read_csv(AN / "sorrybench_official_overall.csv"):
        base[r["model"]] = float(r["fulfillment_rate_pct_macro"])
    mut = {}
    for r in read_csv(AN / "mutation_fulfillment.csv"):
        mut[r["model"]] = {s: (float(r[s]) if r.get(s) not in (None, "") else np.nan)
                           for s in STYLES}
    return base, mut


def main():
    base, mut = load()
    models = [m for m in ORDER if m in mut]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 130,
                         "savefig.bbox": "tight"})

    # ---- resilience_gap.csv + summary ----
    with (AN / "resilience_gap.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "class", "base_fulfill", "mean_mutation",
                    "mean_uplift", "worst_style", "worst_fulfill", "worst_uplift"])
        rows = []
        for m in models:
            b = base.get(m, np.nan)
            vals = {s: mut[m][s] for s in STYLES if not np.isnan(mut[m].get(s, np.nan))}
            if not vals:
                continue
            mean_mut = float(np.mean(list(vals.values())))
            worst_s = max(vals, key=vals.get)
            row = [m, CLASS[m], f"{b:.1f}", f"{mean_mut:.1f}", f"{mean_mut-b:+.1f}",
                   worst_s, f"{vals[worst_s]:.1f}", f"{vals[worst_s]-b:+.1f}"]
            w.writerow(row); rows.append((m, b, mean_mut, vals[worst_s], worst_s))

    # ---- Fig 1: heatmap model x style ----
    mat = np.full((len(models), len(STYLES)), np.nan)
    for i, m in enumerate(models):
        for j, s in enumerate(STYLES):
            mat[i, j] = mut[m].get(s, np.nan)
    fig, ax = plt.subplots(figsize=(13, 4.6))
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(STYLES))); ax.set_xticklabels(STYLES, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(models))); ax.set_yticklabels([LABEL[m] for m in models])
    for i in range(len(models)):
        for j in range(len(STYLES)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if mat[i, j] > 55 else "#222")
    prev, start = None, 0
    for j, fam in enumerate(FAMILY + [None]):
        if fam != prev and prev is not None:
            ax.axvline(j - 0.5, color="#333", lw=1.5)
            ax.text((start + j - 1) / 2, -0.85, prev, ha="center", fontsize=9, weight="bold")
            start = j
        prev = fam
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02).set_label("Fulfillment (%)")
    ax.set_title("SORRY-Bench fulfillment under 20 linguistic mutations (official Mistral judge)",
                 weight="bold", pad=34)
    fig.savefig(AN / "fig_mutation_heatmap.png"); plt.close(fig)

    # ---- Fig 2: base vs mean vs worst-case, grouped, colored by class ----
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(models)); bw = 0.26
    b = [base.get(m, np.nan) for m in models]
    mean_mut = [float(np.nanmean([mut[m][s] for s in STYLES])) for m in models]
    worst = [float(np.nanmax([mut[m][s] for s in STYLES])) for m in models]
    ax.bar(x - bw, b, bw, label="Base (undisguised)", color="#999999", zorder=3)
    ax.bar(x, mean_mut, bw, label="Mean over 20 mutations", color="#56B4E9", zorder=3)
    ax.bar(x + bw, worst, bw, label="Worst-case mutation", color="#D55E00", zorder=3)
    for xi, m in enumerate(models):
        ax.text(xi + bw, worst[xi] + 1.5, f"+{worst[xi]-b[xi]:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([LABEL[m] for m in models], rotation=20, ha="right")
    # color the tick labels by class
    for tick, m in zip(ax.get_xticklabels(), models):
        tick.set_color(CLASS_COLOR[CLASS[m]])
    ax.set_ylabel("SORRY-Bench fulfillment (%)")
    ax.set_title("Adversarial resilience gap: fulfillment uplift under mutation "
                 "(worst-case label = uplift over base)", weight="bold")
    ax.legend(frameon=False); ax.grid(axis="x", visible=False)
    fig.savefig(AN / "fig_resilience_gap.png"); plt.close(fig)

    print("Wrote resilience_gap.csv, fig_mutation_heatmap.png, fig_resilience_gap.png")
    print("\nResilience summary (base -> mean-mutation / worst-case):")
    for m, bb, mm, ww, ws in rows:
        print(f"  {LABEL[m]:20s} {bb:5.1f} -> {mm:5.1f} / {ww:5.1f}  (worst: {ws}, +{ww-bb:.1f})")


if __name__ == "__main__":
    main()
