#!/usr/bin/env python3
"""Aggregate experiment-v2 eval summaries into one comparison table:
baseline vs ES(no-AWD) vs ES(+AWD), on the safety axis (Opus-judge answer-level
unsafe% on the test split) and the reasoning axis (MATH probe accuracy) plus the
benign over-refusal probe. Reads results/<tag>__<dataset>.summary.json."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
TAGS = [("baseline", "vibe_base_v2"), ("ES no-AWD", "vibe_es_noawd"), ("ES +AWD", "vibe_es_awd")]


def load(tag, dataset):
    f = RES / f"{tag}__{dataset}.summary.json"
    if not f.is_file():
        return None
    return json.loads(f.read_text()).get("summaries", {})


def pct(x):
    return f"{100 * x:.1f}%" if isinstance(x, (int, float)) else "—"


def main():
    print(f"\n{'model':<12} | {'SORRY unsafe':>13} | {'AILu unsafe':>12} | "
          f"{'pooled':>7} | {'MATH acc':>9} | {'benign unsafe':>13}")
    print("-" * 80)
    for label, tag in TAGS:
        saf = load(tag, "safety_test") or {}
        s = saf.get("sorry_bench", {})
        a = saf.get("ailuminate", {})
        su, au = s.get("answer_unsafe_rate"), a.get("answer_unsafe_rate")
        sn, an = s.get("answer_n", 0), a.get("answer_n", 0)
        pooled = ((su * sn + au * an) / (sn + an)) if (su is not None and au is not None and sn + an) else None
        rea_all = load(tag, "reasoning_probe_200") or load(tag, "reasoning_probe") or {}
        rea = rea_all.get("reasoning_probe_200") or rea_all.get("reasoning_probe", {})
        acc = rea.get("accuracy")
        ben = load(tag, "benign_probe") or {}
        bkey = next(iter(ben), None)
        bunsafe = ben.get(bkey, {}).get("answer_unsafe_rate") if bkey else None
        print(f"{label:<12} | {pct(su):>13} | {pct(au):>12} | {pct(pooled):>7} | "
              f"{pct(acc):>9} | {pct(bunsafe):>13}")
    print("\n(lower unsafe% = safer; higher MATH acc = reasoning preserved. "
          "AWD's aim: hold MATH acc while safety improves.)")
    ck = RES / "v2_checkpoints.txt"
    if ck.is_file():
        print("\n" + ck.read_text())


if __name__ == "__main__":
    main()
