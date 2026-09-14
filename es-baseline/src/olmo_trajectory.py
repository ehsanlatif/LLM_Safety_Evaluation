#!/usr/bin/env python3
"""
OLMo 3 7B safety trajectory — the headline deliverable of the sweep.

Reads the analysis CSVs produced by analyze.py / eval_sorrybench.py /
eval_ailuminate.py / mutation_analysis.py and produces:
  - analysis/fig_olmo_trajectory.png : unsafe-% along each post-training branch
    (Base -> SFT -> DPO -> RL) for Instruct and Think, plus the RL-Zero cluster,
    on both SorryBench and AILuminate (unified Claude judge, answer-level).
  - REPORT_olmo.md : data-driven write-up (per-variant table + auto observations).

Safe to run before all rows exist: missing models are simply skipped.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AN = Path("analysis")

# The model flow (friendly names). Base is the shared root of both chains.
BASE = "olmo3-7b-base"
INSTRUCT = [BASE, "olmo3-7b-instruct-sft", "olmo3-7b-instruct-dpo", "olmo3-7b-instruct"]
THINK = [BASE, "olmo3-7b-think-sft", "olmo3-7b-think-dpo", "olmo3-7b-think-rl"]
RLZERO = ["olmo3-7b-rlzero-math", "olmo3-7b-rlzero-code", "olmo3-7b-rlzero-if",
          "olmo3-7b-rlzero-general", "olmo3-7b-rlzero-mix"]
STAGES = ["Base", "SFT", "DPO", "RL"]
ALL_OLMO = INSTRUCT + THINK[1:] + RLZERO

LABEL = {
    "olmo3-7b-base": "Base", "olmo3-7b-instruct-sft": "Instruct-SFT",
    "olmo3-7b-instruct-dpo": "Instruct-DPO", "olmo3-7b-instruct": "Instruct-RL",
    "olmo3-7b-think-sft": "Think-SFT", "olmo3-7b-think-dpo": "Think-DPO",
    "olmo3-7b-think-rl": "Think-RL", "olmo3-7b-rlzero-math": "RLZero-Math",
    "olmo3-7b-rlzero-code": "RLZero-Code", "olmo3-7b-rlzero-if": "RLZero-IF",
    "olmo3-7b-rlzero-general": "RLZero-General", "olmo3-7b-rlzero-mix": "RLZero-Mix",
}
PERSUASION = ["authority_endorsement", "misrepresentation", "logical_appeal",
              "evidence-based_persuasion", "expert_endorsement"]
STYLES = ["slang", "uncommon_dialects", "technical_terms", "role_play", "misspellings",
          "question", "authority_endorsement", "misrepresentation", "logical_appeal",
          "evidence-based_persuasion", "expert_endorsement", "ascii", "caesar", "morse",
          "atbash", "translate-fr", "translate-ml", "translate-mr", "translate-ta",
          "translate-zh-cn"]


def read_csv(p):
    return list(csv.DictReader(p.open())) if p.is_file() else []


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load():
    """Return per-model metrics dict keyed by friendly name."""
    # Unified Claude judge, answer-level unsafe% per (model, dataset).
    llm = {}
    for r in read_csv(AN / "summary_overall.csv"):
        llm.setdefault(r["model"], {})[r["dataset"]] = num(r.get("answer_unsafe_pct"))
    sb_off = {r["model"]: num(r.get("fulfillment_rate_pct_macro"))
              for r in read_csv(AN / "sorrybench_official_overall.csv")}
    ail_off = {r["model"]: (num(r.get("unsafe_pct")), r.get("grade", ""))
               for r in read_csv(AN / "ailuminate_official_overall.csv")}
    mut = {}
    for r in read_csv(AN / "mutation_fulfillment.csv"):
        mut[r["model"]] = {s: num(r.get(s)) for s in STYLES}

    out = {}
    for m in ALL_OLMO:
        d = llm.get(m, {})
        mrow = mut.get(m, {})
        mvals = [v for v in mrow.values() if v is not None]
        pvals = [mrow[s] for s in PERSUASION if mrow.get(s) is not None]
        out[m] = {
            "sb_llm": d.get("sorry-bench"),
            "ail_llm": d.get("ailuminate"),
            "sb_off": sb_off.get(m),
            "ail_off": ail_off.get(m, (None, ""))[0],
            "ail_grade": ail_off.get(m, (None, ""))[1],
            "mut_worst": max(mvals) if mvals else None,
            "mut_persuasion": (sum(pvals) / len(pvals)) if pvals else None,
        }
    return out


def _series(models, metric, data):
    xs, ys = [], []
    for i, m in enumerate(models):
        v = data.get(m, {}).get(metric)
        if v is not None:
            xs.append(i); ys.append(v)
    return xs, ys


def make_figure(data):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 140,
                         "savefig.bbox": "tight"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    panels = [("sb_llm", "SORRY-Bench (440)"), ("ail_llm", "AILuminate (1,200)")]
    for ax, (metric, title) in zip(axes, panels):
        xi, yi = _series(INSTRUCT, metric, data)
        xt, yt = _series(THINK, metric, data)
        if xi:
            ax.plot(xi, yi, "-o", color="#0072B2", lw=2, label="Instruct branch")
        if xt:
            ax.plot(xt, yt, "-s", color="#E69F00", lw=2, label="Think branch")
        # RL-Zero cluster at an offset x past the RL stage.
        for j, m in enumerate(RLZERO):
            v = data.get(m, {}).get(metric)
            if v is not None:
                ax.scatter(3.5 + j * 0.12, v, color="#009E73", zorder=4)
                ax.annotate(LABEL[m].replace("RLZero-", ""), (3.5 + j * 0.12, v),
                            fontsize=6, rotation=90, ha="center", va="bottom")
        ax.set_xticks(range(len(STAGES)))
        ax.set_xticklabels(STAGES)
        ax.set_title(title)
        ax.set_xlabel("Post-training stage")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Unsafe % (Claude judge, answer-level)")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], marker="o", ls="", color="#009E73"))
    labels.append("RL-Zero (single-domain)")
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("OLMo 3 (7B) safety across the post-training model flow", y=1.09)
    out = AN / "fig_olmo_trajectory.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def _f(v, suf="%"):
    return f"{v:.1f}{suf}" if isinstance(v, (int, float)) else "—"


def _delta(data, chain, metric):
    a = data.get(chain[0], {}).get(metric)
    b = data.get(chain[-1], {}).get(metric)
    if a is None or b is None:
        return None
    return b - a


def write_report(data):
    have = [m for m in ALL_OLMO if any(v is not None for v in data.get(m, {}).values())]
    lines = []
    lines.append("# OLMo 3 (7B) — Safety Across the Post-Training Model Flow\n")
    lines.append(
        "Safety evaluation of the OLMo 3 7B family, tracing how refusal behavior evolves "
        "across each post-training stage (Base → SFT → DPO → RL) for the **Instruct** and "
        "**Think** branches, plus the experimental **RL-Zero** runs. Benchmarks: SORRY-Bench "
        "(440 base + 20 linguistic mutations) and AILuminate (1,200 demo prompts). Graders: "
        "unified Claude judge (`claude-haiku-4-5`, answer-level, **primary**), official "
        "SORRY-Bench ft-Mistral-7B (fulfillment %), and LlamaGuard-2-8B (AILuminate grade). "
        "Lower unsafe/fulfillment % = safer.\n")
    if not have:
        lines.append("\n_No results yet — run the pipeline, then re-run `olmo_trajectory.py`._\n")
        (Path("REPORT_olmo.md")).write_text("\n".join(lines))
        print("wrote REPORT_olmo.md (empty — no rows yet)")
        return

    lines.append("\n## Per-variant safety\n")
    lines.append("| Variant | SB unsafe% (LLM) | AIL unsafe% (LLM) | AIL grade | SB fulfillment% (official) | Mutation worst-case% | Persuasion mean% |")
    lines.append("|---|---|---|---|---|---|---|")
    for m in ALL_OLMO:
        d = data.get(m, {})
        if m not in have:
            continue
        lines.append(f"| {LABEL[m]} | {_f(d['sb_llm'])} | {_f(d['ail_llm'])} | "
                     f"{d['ail_grade'] or '—'} | {_f(d['sb_off'])} | "
                     f"{_f(d['mut_worst'])} | {_f(d['mut_persuasion'])} |")

    lines.append("\n## Observations (auto-computed)\n")
    for name, chain in (("Instruct", INSTRUCT), ("Think", THINK)):
        for metric, ds in (("sb_llm", "SORRY-Bench"), ("ail_llm", "AILuminate")):
            dlt = _delta(data, chain, metric)
            if dlt is not None:
                verb = "reduced" if dlt < 0 else "increased"
                lines.append(f"- **{name} branch, {ds}:** unsafe% {verb} by "
                             f"{abs(dlt):.1f} pp from Base to RL "
                             f"({_f(data[chain[0]].get(metric))} → {_f(data[chain[-1]].get(metric))}).")
    # Reasoning tax at the final RL stage.
    for metric, ds in (("sb_llm", "SORRY-Bench"), ("ail_llm", "AILuminate")):
        i = data.get("olmo3-7b-instruct", {}).get(metric)
        t = data.get("olmo3-7b-think-rl", {}).get(metric)
        if i is not None and t is not None:
            gap = t - i
            lines.append(f"- **Reasoning tax ({ds}):** final Think is "
                         f"{'less' if gap > 0 else 'more'} safe than final Instruct by "
                         f"{abs(gap):.1f} pp ({_f(t)} vs {_f(i)}).")
    # Safest / least-safe by primary SB LLM metric.
    ranked = [(m, data[m].get("sb_llm")) for m in have if data[m].get("sb_llm") is not None]
    if ranked:
        ranked.sort(key=lambda x: x[1])
        lines.append(f"- **Safest (SB, LLM judge):** {LABEL[ranked[0][0]]} ({_f(ranked[0][1])}); "
                     f"**least safe:** {LABEL[ranked[-1][0]]} ({_f(ranked[-1][1])}).")
    lines.append("\n## Figure\n\n![OLMo 3 7B safety trajectory](analysis/fig_olmo_trajectory.png)\n")
    lines.append("\n_See also `analysis/FINAL_master_table.csv` (cross-evaluator), "
                 "`analysis/mutation_fulfillment.csv` (per-style), and "
                 "`analysis/fig_mutation_heatmap.png`._\n")
    Path("REPORT_olmo.md").write_text("\n".join(lines))
    print(f"wrote REPORT_olmo.md ({len(have)}/{len(ALL_OLMO)} variants with data)")


def main():
    data = load()
    make_figure(data)
    write_report(data)


if __name__ == "__main__":
    main()
