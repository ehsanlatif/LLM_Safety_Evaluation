#!/usr/bin/env python3
"""Build a 200-sample MATH reasoning probe (datasets/reasoning_probe_200) so a
small AWD reasoning effect (~2-4 pp) becomes measurable. Same source/schema as the
50-sample probe in build_safety_datasets.py; deterministic (seed 42)."""
import random
from pathlib import Path
from datasets import Dataset, load_from_disk

ROOT = Path(__file__).resolve().parent.parent
MATH_SUITE = Path("$HOME/es_reasoning/datasets/evaluation_suite")
N = 200
SEED = 42

math = load_from_disk(str(MATH_SUITE))["math"]
idx = list(range(len(math)))
random.Random(SEED).shuffle(idx)
sel = idx[:N]
rows = [{"problem": math[i]["problem"], "answer": math[i]["answer"],
         "meta": {"id": f"math-{i}", "source": "math"}} for i in sel]
out = ROOT / "datasets" / "reasoning_probe_200"
Dataset.from_list(rows).save_to_disk(str(out))
print(f"saved {len(rows)} MATH problems -> {out}")
