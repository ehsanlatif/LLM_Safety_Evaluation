#!/usr/bin/env python3
"""Derive a small in-loop validation set from safety_val to bound Opus judge cost
during ES (the population-eval scores every member on this set). Default 20+20=40."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
from datasets import Dataset, DatasetDict, load_from_disk

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
src = load_from_disk(str(ROOT / "datasets" / "safety_val"))
out = DatasetDict({name: Dataset.from_list([d[i] for i in range(min(N, len(d)))])
                   for name, d in src.items()})
out.save_to_disk(str(ROOT / "datasets" / "safety_val_small"))
print({name: len(d) for name, d in out.items()}, "-> datasets/safety_val_small")
