#!/usr/bin/env python3
"""
Temperature / response-diversity vs safety — analysis for the OLMo-3-7B-Instruct-SFT
sampling sweep (see run_temp_sweep.sh).

Reads the isolated sweep outputs under results/temp_sweep/ and produces:
  - analysis/fig_temp_curve.png    unsafe% vs temperature (top_p=0.9 curve), LLM
                                   judge + official judges, SorryBench & AILuminate
  - analysis/fig_temp_methods.png  diversity-threshold comparison at T=1.0
                                   (top_p=0.9 vs top_k=50 vs min_p=0.1)
  - REPORT_temperature.md          per-config table + auto-computed observations

Metrics (all "higher = less safe" except SB-official is fulfillment%, same sense):
  - LLM judge (primary): answer-level unsafe% from graded/<run>__<ds>.jsonl
  - SorryBench official : fulfillment% (ft-Mistral) from analysis/sorrybench_official_overall.csv
  - AILuminate official : LlamaGuard-2 unsafe% from official_ailuminate/<run>.jsonl
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SW = Path("results/temp_sweep")
AN = Path("analysis")
BLUE, ORANGE, GREY, GREEN = "#0072B2", "#E69F00", "#999999", "#009E73"
_RC = {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
       "savefig.dpi": 140, "savefig.bbox": "tight"}


def read_csv(p):
    return list(csv.DictReader(p.open())) if p.is_file() else []


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def llm_unsafe_pct(run, dataset):
    """Answer-level unsafe% from the LLM-judge graded file (safe/unsafe only)."""
    p = SW / "graded" / f"{run}__{dataset}.jsonl"
    if not p.is_file():
        return None
    u = n = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        lab = json.loads(line).get("answer_label")
        if lab in ("safe", "unsafe"):
            n += 1
            u += (lab == "unsafe")
    return 100.0 * u / n if n else None


def ail_official_unsafe_pct(run):
    """LlamaGuard-2 unsafe% from the annotate output (valid & not-safe)."""
    p = SW / "official_ailuminate" / f"{run}.jsonl"
    if not p.is_file():
        return None
    u = n = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("is_valid"):
            n += 1
            u += (not r.get("is_safe"))
    return 100.0 * u / n if n else None


def load():
    manifest = read_csv(SW / "manifest.csv")
    sb_off = {r["model"]: num(r.get("fulfillment_rate_pct_macro"))
              for r in read_csv(SW / "analysis" / "sorrybench_official_overall.csv")}
    rows = []
    for m in manifest:
        run = m["run"]
        rows.append({
            "run": run, "temperature": num(m["temperature"]),
            "top_p": num(m["top_p"]), "top_k": num(m["top_k"]),
            "min_p": num(m["min_p"]), "arm": m["arm"],
            "sb_llm": llm_unsafe_pct(run, "sorry-bench"),
            "ail_llm": llm_unsafe_pct(run, "ailuminate"),
            "sb_off": sb_off.get(run),
            "ail_off": ail_official_unsafe_pct(run),
        })
    return rows


def _method_label(r):
    if r["top_k"] and r["top_k"] > 0:
        return f"top_k={int(r['top_k'])}"
    if r["min_p"] and r["min_p"] > 0:
        return f"min_p={r['min_p']:g}"
    return f"top_p={r['top_p']:g}"


# ---------------------------------------------------------------------------
def fig_curve(rows):
    curve = sorted([r for r in rows if r["arm"] == "temp_curve"],
                   key=lambda r: r["temperature"])
    if not curve:
        print("skip fig_temp_curve: no temp_curve rows"); return
    xs = [r["temperature"] for r in curve]
    plt.rcParams.update(_RC)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Left: unified LLM judge (primary).
    a = axes[0]
    a.plot(xs, [r["sb_llm"] for r in curve], "-o", color=BLUE, lw=2, label="SORRY-Bench")
    a.plot(xs, [r["ail_llm"] for r in curve], "-s", color=ORANGE, lw=2, label="AILuminate")
    a.set_title("Unified LLM judge (answer-level)")
    a.set_ylabel("Unsafe %")
    # Right: official judges.
    b = axes[1]
    b.plot(xs, [r["sb_off"] for r in curve], "-o", color=BLUE, lw=2,
           label="SORRY-Bench ft-Mistral (fulfillment%)")
    b.plot(xs, [r["ail_off"] for r in curve], "-s", color=ORANGE, lw=2,
           label="AILuminate LlamaGuard-2 (unsafe%)")
    b.set_title("Official judges")
    b.set_ylabel("%")
    for ax in axes:
        ax.set_xlabel("Temperature (top_p=0.9)")
        ax.axvline(0.0, color=GREY, ls=":", lw=1)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("OLMo-3-7B-Instruct-SFT — safety vs decoding temperature", y=1.02)
    out = AN / "fig_temp_curve.png"
    fig.savefig(out); plt.close(fig)
    print(f"wrote {out}")


def fig_methods(rows):
    # The three diversity thresholds at T=1.0.
    at1 = {r["run"]: r for r in rows if r["temperature"] == 1.0}
    order = ["olmosft-t10-p90", "olmosft-t10-k50", "olmosft-t10-mp10"]
    picks = [at1[k] for k in order if k in at1]
    if len(picks) < 2:
        print("skip fig_temp_methods: need the T=1.0 method configs"); return
    labels = [_method_label(r) for r in picks]
    metrics = [("sb_llm", "SB unsafe% (LLM)", BLUE),
               ("ail_llm", "AIL unsafe% (LLM)", ORANGE),
               ("sb_off", "SB fulfillment% (official)", GREEN),
               ("ail_off", "AIL unsafe% (LG2)", GREY)]
    plt.rcParams.update(_RC)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(picks))
    w = 0.2
    for k, (key, lbl, col) in enumerate(metrics):
        ax.bar([xi + (k - 1.5) * w for xi in x],
               [r.get(key) if r.get(key) is not None else 0 for r in picks],
               w, label=lbl, color=col)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("%")
    ax.set_xlabel("Response-diversity threshold (all at T=1.0)")
    ax.set_title("OLMo-3-7B-Instruct-SFT — safety by diversity-threshold method (T=1.0)")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    out = AN / "fig_temp_methods.png"
    fig.savefig(out); plt.close(fig)
    print(f"wrote {out}")


def _f(v):
    return f"{v:.1f}" if isinstance(v, (int, float)) else "—"


def write_report(rows):
    curve = sorted([r for r in rows if r["arm"] == "temp_curve"],
                   key=lambda r: r["temperature"])
    L = ["# OLMo-3-7B-Instruct-SFT — Safety vs Decoding Temperature\n",
         "Does sampling temperature (and the response-diversity threshold) change the "
         "safety of the safest OLMo variant? Base config: `Olmo-3-7B-Instruct-SFT`, "
         "seed=0, 1024-token budget, SorryBench (440) + AILuminate (1,200). Graders: "
         "unified Claude judge (answer-level, **primary**), official SorryBench "
         "ft-Mistral (fulfillment%), LlamaGuard-2 (unsafe%). Higher = less safe "
         "(SorryBench-official is fulfillment%, same direction).\n",
         "\nGrid grounded in the decoding-safety literature: **Arm 1** sweeps temperature "
         "at the standard nucleus threshold `top_p=0.9` (Holtzman 2019; Huang et al., "
         "ICLR 2024); **Arm 2** compares diversity thresholds at T=1.0 — `top_k=50` "
         "(more exploitable than top-p per 2024-25 jailbreak-oracle work) and `min_p=0.1` "
         "(Nguyen et al., ICLR 2025).\n",
         "\n## Per-config safety\n",
         "| Config | T | Threshold | SB unsafe% (LLM) | AIL unsafe% (LLM) | "
         "SB fulfillment% (official) | AIL unsafe% (LG2) |",
         "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['run']} | {r['temperature']:g} | {_method_label(r)} | "
                 f"{_f(r['sb_llm'])} | {_f(r['ail_llm'])} | {_f(r['sb_off'])} | "
                 f"{_f(r['ail_off'])} |")

    L.append("\n## Observations (auto-computed)\n")
    if len(curve) >= 2 and curve[0]["temperature"] == 0.0:
        g, hi = curve[0], curve[-1]
        for key, ds in (("sb_llm", "SORRY-Bench"), ("ail_llm", "AILuminate")):
            a, b = g.get(key), hi.get(key)
            if a is not None and b is not None:
                verb = "increased" if b > a else "decreased"
                L.append(f"- **{ds} (LLM judge):** unsafe% {verb} by {abs(b-a):.1f} pp from "
                         f"greedy (T=0) to T={hi['temperature']:g} ({a:.1f}% → {b:.1f}%).")
        # Monotonicity of the SB LLM curve.
        seq = [r["sb_llm"] for r in curve if r["sb_llm"] is not None]
        if len(seq) == len([r for r in curve]):
            mono = all(seq[i] <= seq[i+1] + 1e-9 for i in range(len(seq)-1))
            L.append(f"- **SORRY-Bench trend:** {'monotonically increasing' if mono else 'non-monotonic'} "
                     f"in temperature.")
    # Method comparison at T=1.0.
    at1 = {r["run"]: r for r in rows if r["temperature"] == 1.0}
    trip = [(k, at1[k]) for k in ("olmosft-t10-p90", "olmosft-t10-k50", "olmosft-t10-mp10")
            if k in at1]
    ranked = [(r["sb_llm"], _method_label(r)) for _, r in trip if r["sb_llm"] is not None]
    if len(ranked) >= 2:
        ranked.sort()
        L.append(f"- **Diversity threshold at T=1.0 (SB, LLM):** safest = {ranked[0][1]} "
                 f"({ranked[0][0]:.1f}%); least safe = {ranked[-1][1]} ({ranked[-1][0]:.1f}%).")
    L.append("\n## Figures\n\n![safety vs temperature](analysis/fig_temp_curve.png)\n"
             "\n![diversity-threshold methods](analysis/fig_temp_methods.png)\n")
    Path("REPORT_temperature.md").write_text("\n".join(L))
    print("wrote REPORT_temperature.md")


def main():
    AN.mkdir(exist_ok=True)
    rows = load()
    if not rows:
        print("no manifest/rows found under results/temp_sweep — run run_temp_sweep.sh first")
        return
    fig_curve(rows)
    fig_methods(rows)
    write_report(rows)


if __name__ == "__main__":
    main()
