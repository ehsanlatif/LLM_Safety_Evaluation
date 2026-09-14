#!/usr/bin/env python3
"""
Render reward + safety-eval curves for an ES fine-tuning run from its metrics.jsonl.
(Complements the live W&B panels; works offline.)

Usage:
    plot_training.py <experiment_dir>          # dir containing metrics.jsonl
    plot_training.py --latest                  # newest run under experiments/
Outputs <dir>/plots/{reward_curve.png, safety_eval_curve.png}.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = ROOT / "experiments"


def load_metrics(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def series(rows, key, xkey="global_step"):
    xs, ys = [], []
    for r in rows:
        if key in r and xkey in r and isinstance(r[key], (int, float)):
            xs.append(r[xkey]); ys.append(r[key])
    # sort by x
    pts = sorted(zip(xs, ys))
    return [p[0] for p in pts], [p[1] for p in pts]


def resolve_dir(args):
    if args.dir:
        return Path(args.dir)
    runs = [p.parent for p in EXPERIMENTS.glob("*/metrics.jsonl")]
    if not runs:
        raise SystemExit(f"no metrics.jsonl under {EXPERIMENTS}")
    return max(runs, key=lambda p: (p / "metrics.jsonl").stat().st_mtime)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default=None)
    ap.add_argument("--latest", action="store_true")
    args = ap.parse_args()

    run = resolve_dir(args)
    rows = load_metrics(run / "metrics.jsonl")
    if not rows:
        raise SystemExit(f"empty metrics in {run}")
    out = run / "plots"; out.mkdir(exist_ok=True)
    print(f"run={run}  rows={len(rows)}")

    # ---- reward curve ----
    xm, ym = series(rows, "train/reward/mean")
    xlo, ylo = series(rows, "train/reward/min")
    xhi, yhi = series(rows, "train/reward/max")
    if xm:
        plt.figure(figsize=(8, 4.5))
        plt.plot(xm, ym, color="#2266cc", lw=2, label="mean reward")
        if xlo and xhi and len(ylo) == len(yhi) == len(ym):
            plt.fill_between(xm, ylo, yhi, color="#2266cc", alpha=0.15, label="min–max")
        plt.xlabel("ES iteration"); plt.ylabel("LlamaGuard reward (=safe fraction)")
        plt.title("ES training reward"); plt.grid(alpha=0.3); plt.legend()
        plt.tight_layout(); plt.savefig(out / "reward_curve.png", dpi=130); plt.close()
        print("wrote", out / "reward_curve.png")

    # ---- safety eval curves (safe_rate per split + overall) ----
    safe_keys = sorted({k for r in rows for k in r
                        if k.startswith("eval/") and k.endswith("safe_rate")})
    avg_keys = [k for k in {k for r in rows for k in r} if "avgpass@1/mean" in k]
    if safe_keys or avg_keys:
        plt.figure(figsize=(8, 4.5))
        for k in safe_keys:
            xs, ys = series(rows, k)
            if xs:
                label = k.replace("eval/", "").replace("/safe_rate", "")
                plt.plot(xs, ys, marker="o", ms=3, label=f"{label} safe-rate")
        for k in avg_keys:
            xs, ys = series(rows, k)
            if xs:
                plt.plot(xs, ys, "--", color="k", alpha=0.6, label="overall mean reward")
        plt.xlabel("ES iteration"); plt.ylabel("held-out safe fraction (↑ safer)")
        plt.ylim(-0.02, 1.02); plt.title("Held-out safety during ES"); plt.grid(alpha=0.3); plt.legend()
        plt.tight_layout(); plt.savefig(out / "safety_eval_curve.png", dpi=130); plt.close()
        print("wrote", out / "safety_eval_curve.png")


if __name__ == "__main__":
    main()
