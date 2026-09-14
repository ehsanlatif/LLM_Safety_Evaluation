#!/usr/bin/env python3
"""Analysis figures for the ES-safety experiment -> results/figs/*.png"""
import json, glob, os
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "results" / "figs"; FIGS.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 130})

C = {"base": "#6b7280", "es": "#2563eb", "qwen": "#059669", "guard": "#7c3aed"}


def metrics():
    run = max(glob.glob(str(ROOT / "experiments/es-safety-*/metrics.jsonl")), key=os.path.getmtime)
    return [json.loads(l) for l in open(run) if l.strip()]


def summ(tag, split):
    return json.load(open(ROOT / f"results/{tag}__safety_test.summary.json"))["summaries"][split]


# ---------- Fig 1: fine-tuning dynamics ----------
rows = metrics()
tr = [r for r in rows if "train/reward/mean" in r]
it = [r["global_step"] for r in tr]
rm = [r["train/reward/mean"] for r in tr]
rlo = [r.get("train/reward/min") for r in tr]; rhi = [r.get("train/reward/max") for r in tr]
ln = [r.get("train/response-length/mean") for r in tr]
ev = [r for r in rows if "eval/avgsafe_rate" in r]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
ax1.plot(it, rm, color=C["es"], lw=2, label="reward (mean)")
ax1.fill_between(it, rlo, rhi, color=C["es"], alpha=0.15, label="reward (min–max)")
ax1.axhline(1.0, color="k", ls=":", lw=1, alpha=0.6)
ax1.text(1, 1.005, "pure-safe = 1.0 (above = concise-refusal bonus)", fontsize=8, color="k")
ax1.set_xlabel("ES iteration"); ax1.set_ylabel("LlamaGuard reward", color=C["es"])
axl = ax1.twinx(); axl.plot(it, ln, color="#d97706", lw=2, ls="--", label="response length")
axl.set_ylabel("response length (tokens)", color="#d97706"); axl.grid(False)
ax1.set_title("Fine-tuning dynamics: reward ↑, length ↓ (saturation signal)")
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = axl.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")

for key, lab in [("eval/sorry_bench/safe_rate", "SORRY-Bench"),
                 ("eval/ailuminate/safe_rate", "AILuminate"),
                 ("eval/avgsafe_rate", "avg")]:
    xs = [r["global_step"] for r in ev if key in r]; ys = [r[key] for r in ev if key in r]
    ax2.plot(xs, ys, marker="o", label=lab)
ax2.set_xlabel("ES iteration"); ax2.set_ylabel("held-out VAL safe-rate (LlamaGuard)")
ax2.set_ylim(0.85, 1.0); ax2.set_title("In-loop val safety (selection proxy)"); ax2.legend(fontsize=9)
fig.tight_layout(); fig.savefig(FIGS / "fine_tuning_dynamics.png"); plt.close(fig)

# ---------- Fig 2: safety before/after (plain-chat test, Claude judge) ----------
models = [("vibe_base", "VibeThinker-3B\n(plain, baseline)", C["base"]),
          ("vibe_es", "ES iter50\n(plain)", C["es"]),
          ("qwen_base", "Qwen2.5-3B\n(reference)", C["qwen"]),
          ("vibe_base_guardrail", "VibeThinker\n+guardrail (ceiling)", C["guard"])]
splits = ["sorry_bench", "ailuminate"]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for ax, split in zip(axes, splits):
    xs = range(len(models))
    vals = [summ(t, split)["answer_unsafe_rate"] * 100 for t, _, _ in models]
    err = [[(summ(t, split)["answer_unsafe_rate"] - summ(t, split)["answer_ci"][0]) * 100 for t, _, _ in models],
           [(summ(t, split)["answer_ci"][1] - summ(t, split)["answer_unsafe_rate"]) * 100 for t, _, _ in models]]
    bars = ax.bar(xs, vals, color=[c for _, _, c in models], yerr=err, capsize=4, alpha=0.9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_xticks(list(xs)); ax.set_xticklabels([m[1] for m in models], fontsize=8)
    ax.set_ylabel("answer-level UNSAFE % (↓ safer)"); ax.set_ylim(0, 42)
    ax.set_title(f"{split} (plain-chat, Claude judge, n=100)")
fig.suptitle("ES fine-tuning lowers plain-chat unsafe rate (guardrail used in TRAINING only)", fontsize=12)
fig.tight_layout(); fig.savefig(FIGS / "safety_before_after.png"); plt.close(fig)

# ---------- Fig 3: capability + channel ----------
fig, (axa, axb) = plt.subplots(1, 2, figsize=(13, 4.6))
# channel: answer vs anywhere, base vs ES (avg of two splits)
def avg(tag, field):
    d = json.load(open(ROOT / f"results/{tag}__safety_test.summary.json"))["summaries"]
    return sum(d[s][field] for s in splits) / 2 * 100
labels = ["answer-level", "anywhere (incl. CoT)"]
base_v = [avg("vibe_base", "answer_unsafe_rate"), avg("vibe_base", "anywhere_unsafe_rate")]
es_v = [avg("vibe_es", "answer_unsafe_rate"), avg("vibe_es", "anywhere_unsafe_rate")]
x = range(len(labels)); w = 0.36
axa.bar([i - w/2 for i in x], base_v, w, label="baseline", color=C["base"])
axa.bar([i + w/2 for i in x], es_v, w, label="ES iter50", color=C["es"])
for i, (bv, ev_) in enumerate(zip(base_v, es_v)):
    axa.text(i - w/2, bv + 0.4, f"{bv:.1f}", ha="center", fontsize=9)
    axa.text(i + w/2, ev_ + 0.4, f"{ev_:.1f}", ha="center", fontsize=9)
axa.set_xticks(list(x)); axa.set_xticklabels(labels); axa.set_ylabel("unsafe % (avg of both benchmarks)")
axa.set_title("Answer channel improves more than CoT"); axa.legend()

# capability: reasoning acc + benign over-refusal
cats = ["Reasoning\n(MATH/50 acc)", "Benign\nover-refusal"]
base_c = [98.0, 0.0]; es_c = [96.0, 0.0]
x = range(len(cats))
axb.bar([i - w/2 for i in x], base_c, w, label="baseline", color=C["base"])
axb.bar([i + w/2 for i in x], es_c, w, label="ES iter50", color=C["es"])
for i, (bv, ev_) in enumerate(zip(base_c, es_c)):
    axb.text(i - w/2, bv + 1, f"{bv:.0f}%", ha="center", fontsize=9)
    axb.text(i + w/2, ev_ + 1, f"{ev_:.0f}%", ha="center", fontsize=9)
axb.set_xticks(list(x)); axb.set_xticklabels(cats); axb.set_ylabel("%"); axb.set_ylim(0, 105)
axb.set_title("No capability cost: reasoning held, no over-refusal"); axb.legend()
fig.tight_layout(); fig.savefig(FIGS / "capability_and_channel.png"); plt.close(fig)

print("wrote:", *[str(p) for p in FIGS.glob("*.png")], sep="\n  ")
