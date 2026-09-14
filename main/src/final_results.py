#!/usr/bin/env python3
"""
FINAL combined results — merge all three evaluators into master tables + figures.

Evaluators:
  1. LLM-judge (claude-haiku-4-5), answer-level  — unified harm metric, both datasets
     (analysis/summary_overall.csv)
  2. AILuminate official method — LlamaGuard-2-8b annotator + MLCommons S*/5-tier grade
     (analysis/ailuminate_official_overall.csv, ..._by_hazard.csv)
  3. SORRY-Bench official — fine-tuned Mistral-7B judge, macro fulfillment rate
     (analysis/sorrybench_official_overall.csv)

Note the metrics measure DIFFERENT things:
  - AILuminate (both LlamaGuard-2 and my LLM-judge) and SORRY-Bench-via-LLM-judge
    measure HARM (did the response enable/deliver harmful content).
  - SORRY-Bench OFFICIAL measures FULFILLMENT across a 44-category taxonomy that
    includes benign categories (advice, opinions, mild insults) -> systematically
    higher, and NOT a pure harm measure. Rankings are broadly preserved.

Writes: analysis/FINAL_master_table.csv, analysis/fig_final_evaluators.png,
        analysis/fig_final_ailuminate_grades.png
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

AN = Path("analysis")
_QWEN3 = ["qwen3-0.6b-nothink", "qwen3-0.6b-think", "qwen3-1.7b-nothink", "qwen3-1.7b-think",
          "qwen3-4b-nothink", "qwen3-4b-think", "qwen3-8b-nothink", "qwen3-8b-think"]
ORDER = ["opus-4.8", "sonnet-5", "gpt-5.5", "llama-3.2-3b", "qwen-3b",
         "deepseek-r1-1.5b", "vibethinker-1.5b", "qwen-1.5b", "vibethinker-3b"] + _QWEN3
LABEL = {"opus-4.8": "Claude Opus 4.8", "sonnet-5": "Claude Sonnet 5", "gpt-5.5": "GPT-5.5",
         "qwen-3b": "Qwen2.5-3B", "vibethinker-1.5b": "VibeThinker-1.5B",
         "llama-3.2-3b": "Llama-3.2-3B", "deepseek-r1-1.5b": "DeepSeek-R1-1.5B",
         "qwen-1.5b": "Qwen2.5-1.5B", "vibethinker-3b": "VibeThinker-3B"}
KLASS = {"opus-4.8": "Large closed", "sonnet-5": "Large closed", "gpt-5.5": "Large closed",
         "qwen-3b": "Small open (non-reason)", "llama-3.2-3b": "Small open (non-reason)",
         "vibethinker-1.5b": "Small open (reason)", "deepseek-r1-1.5b": "Small open (reason)",
         "qwen-1.5b": "Small open (non-reason)", "vibethinker-3b": "Small open (reason)"}
for _s in ("0.6b", "1.7b", "4b", "8b"):
    LABEL[f"qwen3-{_s}-nothink"] = f"Qwen3-{_s.upper()}"
    LABEL[f"qwen3-{_s}-think"] = f"Qwen3-{_s.upper()} (think)"
    KLASS[f"qwen3-{_s}-nothink"] = "Small open (non-reason)"
    KLASS[f"qwen3-{_s}-think"] = "Small open (reason)"
# OLMo 3 7B model-flow sweep (safety across post-training stages).
_OLMO_LABEL = {
    "olmo3-7b-base": "OLMo3-7B Base", "olmo3-7b-instruct-sft": "OLMo3-7B Instruct-SFT",
    "olmo3-7b-instruct-dpo": "OLMo3-7B Instruct-DPO", "olmo3-7b-instruct": "OLMo3-7B Instruct-RL",
    "olmo3-7b-think-sft": "OLMo3-7B Think-SFT", "olmo3-7b-think-dpo": "OLMo3-7B Think-DPO",
    "olmo3-7b-think-rl": "OLMo3-7B Think-RL", "olmo3-7b-rlzero-math": "OLMo3-7B RLZero-Math",
    "olmo3-7b-rlzero-code": "OLMo3-7B RLZero-Code", "olmo3-7b-rlzero-if": "OLMo3-7B RLZero-IF",
    "olmo3-7b-rlzero-general": "OLMo3-7B RLZero-General", "olmo3-7b-rlzero-mix": "OLMo3-7B RLZero-Mix",
}
_OLMO = list(_OLMO_LABEL)
LABEL.update(_OLMO_LABEL)
KLASS.update({m: "OLMo 3 7B" for m in _OLMO})
ORDER = ORDER + _OLMO
OI = dict(blue="#0072B2", orange="#E69F00", green="#009E73", vermillion="#D55E00",
          sky="#56B4E9", purple="#CC79A7", grey="#999999")
GRADE_COLOR = {"Excellent": "#1a9850", "Very Good": "#91cf60", "Good": "#fee08b",
               "Fair": "#fc8d59", "Poor": "#d73027", "": "#dddddd"}


def read_csv(path):
    if not path.is_file():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def load_metrics():
    """Return model -> metric_key -> value(float or str)."""
    M = {m: {} for m in ORDER}

    # 1. LLM-judge answer-level, both datasets
    for r in read_csv(AN / "summary_overall.csv"):
        m = r["model"]
        if m not in M:
            continue
        key = "llmjudge_ail" if r["dataset"] == "ailuminate" else "llmjudge_sb"
        M[m][key] = float(r["answer_unsafe_pct"])

    # 2. AILuminate official (LlamaGuard-2)
    for r in read_csv(AN / "ailuminate_official_overall.csv"):
        if r["model"] in M:
            M[r["model"]]["ail_lg_unsafe"] = float(r["unsafe_pct"])
            M[r["model"]]["ail_lg_grade"] = r["grade"]

    # 3. SORRY-Bench official fulfillment
    for r in read_csv(AN / "sorrybench_official_overall.csv"):
        if r["model"] in M:
            M[r["model"]]["sb_official_fulfill"] = float(r["fulfillment_rate_pct_macro"])
    return M


def master_table(M):
    cols = ["model", "class",
            "AIL_LLMjudge_unsafe%", "AIL_LlamaGuard2_unsafe%", "AIL_LlamaGuard2_grade",
            "SB_LLMjudge_unsafe%", "SB_official_fulfillment%"]
    with (AN / "FINAL_master_table.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(cols)
        for m in ORDER:
            d = M[m]
            w.writerow([LABEL[m], KLASS[m],
                        _f(d.get("llmjudge_ail")), _f(d.get("ail_lg_unsafe")),
                        d.get("ail_lg_grade", ""),
                        _f(d.get("llmjudge_sb")), _f(d.get("sb_official_fulfill"))])
    # print
    print("\n" + "=" * 96)
    print("FINAL MASTER TABLE")
    print("=" * 96)
    hdr = f"{'Model':16s} {'Class':12s} | {'AIL LLMj':>8s} {'AIL LG2':>8s} {'grade':>10s} | {'SB LLMj':>7s} {'SB offic':>8s}"
    print(hdr); print("-" * 96)
    for m in ORDER:
        d = M[m]
        print(f"{LABEL[m]:16s} {KLASS[m]:12s} | {_f(d.get('llmjudge_ail')):>8s} "
              f"{_f(d.get('ail_lg_unsafe')):>8s} {d.get('ail_lg_grade',''):>10s} | "
              f"{_f(d.get('llmjudge_sb')):>7s} {_f(d.get('sb_official_fulfill')):>8s}")
    print("\nAIL = AILuminate (harm); SB = SORRY-Bench. LLMj/LG2 unsafe% & SB official fulfillment%.")


def _f(v):
    return f"{v:.1f}" if isinstance(v, (int, float)) else "-"


def fig_evaluators(M):
    """Two panels: per model, harm judges on each dataset side by side."""
    fig, (axA, axS) = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=False)
    x = np.arange(len(ORDER)); bw = 0.38

    # AILuminate: LLM-judge vs LlamaGuard-2 (both harm)
    v_llm = [M[m].get("llmjudge_ail", np.nan) for m in ORDER]
    v_lg = [M[m].get("ail_lg_unsafe", np.nan) for m in ORDER]
    axA.bar(x - bw/2, v_llm, bw, label="LLM-judge (Claude-Haiku)", color=OI["blue"], zorder=3)
    axA.bar(x + bw/2, v_lg, bw, label="LlamaGuard-2 (AILuminate)", color=OI["orange"], zorder=3)
    for xi, m in enumerate(ORDER):
        g = M[m].get("ail_lg_grade", "")
        if g:
            axA.text(xi + bw/2, (M[m].get("ail_lg_unsafe") or 0) + 1.2, g, ha="center",
                     fontsize=7.5, rotation=90, va="bottom", color="#333")
    axA.set_title("AILuminate — unsafe rate (two harm evaluators)", weight="bold")
    axA.set_ylabel("Unsafe response rate (%)  —  lower is safer")
    axA.set_xticks(x); axA.set_xticklabels([LABEL[m] for m in ORDER], rotation=20, ha="right")
    axA.legend(frameon=False, fontsize=9); axA.grid(axis="x", visible=False)

    # SORRY-Bench: LLM-judge harm vs official fulfillment
    v_llm_sb = [M[m].get("llmjudge_sb", np.nan) for m in ORDER]
    v_off = [M[m].get("sb_official_fulfill", np.nan) for m in ORDER]
    axS.bar(x - bw/2, v_llm_sb, bw, label="LLM-judge harm (Claude-Haiku)", color=OI["blue"], zorder=3)
    axS.bar(x + bw/2, v_off, bw, label="Official fulfillment (ft-Mistral-7B)", color=OI["green"], zorder=3)
    axS.set_title("SORRY-Bench — harm vs. fulfillment", weight="bold")
    axS.set_ylabel("Rate (%)")
    axS.set_xticks(x); axS.set_xticklabels([LABEL[m] for m in ORDER], rotation=20, ha="right")
    axS.legend(frameon=False, fontsize=9); axS.grid(axis="x", visible=False)

    fig.suptitle("Final cross-evaluator comparison", y=1.02, fontsize=13, weight="bold")
    fig.savefig(AN / "fig_final_evaluators.png", bbox_inches="tight", dpi=130)
    plt.close(fig)


def fig_ailuminate_grades(M):
    """AILuminate per-hazard grade grid (Excellent..Poor) if by_hazard exists."""
    rows = read_csv(AN / "ailuminate_official_by_hazard.csv")
    if not rows:
        return
    hazards = []
    grid = {m: {} for m in ORDER}
    for r in rows:
        h = r["hazard"]
        if h not in hazards:
            hazards.append(h)
        if r["model"] in grid:
            grid[r["model"]][h] = r["grade"]
    fig, ax = plt.subplots(figsize=(13, 3.6))
    for i, m in enumerate(ORDER):
        for j, h in enumerate(hazards):
            g = grid[m].get(h, "")
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=GRADE_COLOR.get(g, "#ddd"),
                                       edgecolor="white", lw=1.5))
            ax.text(j + 0.5, i + 0.5, g[:2] if g else "-", ha="center", va="center",
                    fontsize=7, color="#111")
    ax.set_xlim(0, len(hazards)); ax.set_ylim(0, len(ORDER))
    ax.set_xticks(np.arange(len(hazards)) + 0.5); ax.set_xticklabels(hazards, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(ORDER)) + 0.5); ax.set_yticklabels([LABEL[m] for m in ORDER])
    ax.invert_yaxis(); ax.set_title("AILuminate per-hazard grade (LlamaGuard-2 self-consistent)", weight="bold")
    handles = [Patch(color=c, label=g) for g, c in GRADE_COLOR.items() if g]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=5, frameon=False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    fig.savefig(AN / "fig_final_ailuminate_grades.png", bbox_inches="tight", dpi=130)
    plt.close(fig)


def main():
    M = load_metrics()
    if not any("ail_lg_unsafe" in M[m] for m in ORDER):
        print("WARNING: ailuminate_official_overall.csv not found — run "
              "eval_ailuminate.py --phase grade first. Building with available metrics.")
    master_table(M)
    fig_evaluators(M)
    fig_ailuminate_grades(M)
    print(f"\nWrote FINAL_master_table.csv + figures to {AN}/")


if __name__ == "__main__":
    main()
