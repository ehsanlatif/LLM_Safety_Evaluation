# ES fine-tuning for safety alignment of VibeThinker-3B

**Question.** Can Evolution-Strategies (ES) fine-tuning with *very few samples* (30 prompts,
1000 iterations) — the ES-AWD recipe (arXiv:2605.30148 / `kschweig/es-awd`, implemented in
`~/es_reasoning`) — install safety into a reasoning model's **weights**, measured by an
independent **Claude-as-judge** metric?

**Model.** `WeiboAI/VibeThinker-3B` — a math/reasoning model built on Qwen2.5-3B that emits
`<think>…</think>` then an answer. It is not safety-aligned: by default it treats a harmful
request as a problem to *solve and complete*.

**Reward (training).** Local **LlamaGuard-2-8B** on a dedicated GPU, batch-scoring every ES
rollout (answer channel primary; harmful `<think>` capped at 0.5; truncated = 0.0; small
teacher-style refusal bonus). Free, fast, runs inside the ES loop.

**Metric (evaluation).** The exact answer-level unsafe-rate from `saftyAnalysis/grade_responses.py`
— unified **Claude judge `claude-haiku-4-5`**, splitting `<think>` CoT from the user-facing
answer. Lower = safer. 95% Wilson intervals.

---

## Method

### Data — 3-way split, category-stratified, fully disjoint (verified overlap = 0)
Source prompt sets already in the repo: **SORRY-Bench** (440, 44 harm categories) and
**AILuminate** (1,200, 15 hazard categories). `scripts/build_safety_datasets.py` (seed 42):

| Split | Size | Use |
|---|--:|---|
| **train** | 30 (15 SORRY + 15 AILuminate) | ES training prompts (one per distinct category) |
| **val** | 100 (50 + 50) | in-loop LlamaGuard monitoring + checkpoint selection **only** |
| **test** | 200 (100 + 100) | **untouched** until the final plain-chat Claude eval |
| benign_probe | 32 | over-refusal check |
| reasoning_probe | 50 (MATH) | capability-regression check |

The 30 train rows are `{"problem": <harmful prompt>, "answer": "safe", "meta": {...}}`; the
`"answer"` field is an unused placeholder (the reward is LlamaGuard, not a gold label).

### The safety guardrail (training-time only)
A system prompt (`apply_safety_template` in `es_core/train.py`) that puts a safety check **at the
front of the model's thinking trail**: *"…think inside `<think>`; first decide whether the request
is safe; refuse violence/weapons/illegal/hacking/fraud/harassment/hate/CSAM/self-harm incl.
hypothetical/fictional/roleplay framings; then answer."* ES tunes the weights so that gate fires.

**Design decision (user-directed): train *with* the guardrail, evaluate *without* it.** The final
metric uses the **plain chat template** — identical for baseline and ES-tuned — so the number
reflects safety **internalized into the weights**, not the training-time prompt.

### ES configuration
`--population-size 30 --n-iterations 1000 --sigma 0.001 --mu 1` (antithetic noise, z-score reward
shaping), 7 vLLM engines for rollouts + 1 GPU for the LlamaGuard reward actor (8×A100-80GB).
Live reward/eval curves in **wandb** (`es-finetuning` project) and locally in
`experiments/<run>/metrics.jsonl` → `scripts/plot_training.py`.

---

## Baselines — plain chat, **test** split, Claude judge (answer-level unsafe %, 95% CI)

| Model | SORRY-Bench | AILuminate | Coverage |
|---|--:|--:|--:|
| **VibeThinker-3B (plain chat)** — *the model we fine-tune* | **30.3%** [22.1–40.0] | **16.3%** [10.7–24.1] | 99 / 98% |
| Qwen2.5-3B-Instruct (plain chat) — aligned reference | 8.0% | 14.0% | 100% |
| VibeThinker-3B + guardrail **prompt** (ceiling reference) | 6.0% | 7.1% | 100 / 99% |

**Notes.**
- VibeThinker-**3B** is much safer than the VibeThinker-**1.5B** in the earlier `saftyAnalysis`
  study (~54%): the 3B is built on the safety-aligned Qwen2.5-3B, so it inherits a stronger prior.
- The **guardrail prompt alone** (6–7%) beats even Qwen — VibeThinker is fully *capable* of
  refusing; it just lacks the prior by default. This is why the guardrail is a training scaffold
  and the honest test is plain-chat. Example (same prompt): plain chat reasons about policy then
  complies → **unsafe**; guardrail reasons "evaluate if safe" → refuses → **safe**.

**ES target:** move VibeThinker's *plain-chat* test numbers from ~30 / 16% toward the ~6 / 7%
ceiling, **without** the prompt and without wrecking reasoning.

---

## ES results — stopped at iter ~66, best (val-selected) checkpoint = **iter 50**

Training was stopped early (over-saturation signal, below). All eval numbers below are
**plain-chat, untouched test split, Claude judge** — the guardrail was in *training only*.

### Headline: safety (answer-level unsafe %, ↓ safer; 95% Wilson CI)

| Model (plain chat) | SORRY-Bench | AILuminate | Pooled |
|---|--:|--:|--:|
| **VibeThinker-3B — baseline** | 28.3% [20.4–37.8] | 20.8% [13.9–30.0] | **24.6%** |
| **VibeThinker-3B — ES iter50** | **17.5%** [11.2–26.3] | **15.2%** [9.3–23.9] | **16.4%** |
| Qwen2.5-3B-Instruct (reference) | 9.0% | 12.0% | 10.5% |
| VibeThinker + guardrail *prompt* (ceiling) | 7.1% | 9.0% | 8.1% |

**ES lowered plain-chat unsafe rate 24.6% → 16.4% (−8.2 pp, −33% relative, pooled z = 2.01,
p ≈ 0.045)** — with the guardrail present only during training. The gain is **internalized into
the weights**, not a prompt effect. ES closed ~half the gap between the plain baseline and the
guardrail-prompt ceiling, in **30 samples / 50 iterations**.

### No capability cost (the over-saturation check)
| Probe | Baseline | ES iter50 |
|---|--:|--:|
| Reasoning (MATH/50 accuracy) | 98% (49/50) | 96% (48/50) — noise |
| Benign over-refusal (refusal on 32 harmless prompts) | 0% | **0%** |
| Coverage (final answer produced) | 99 / 96% | 97 / 92% |

### Channel effect
Averaged over both benchmarks, ES cut the **answer** channel (24.6→16.4%) more than the
**anywhere/CoT** channel (24→21%): the model learned to refuse in the user-facing answer faster
than to stop *exploring* harm in its private `<think>`. (Expected — the CoT penalty is a capped 0.5
signal vs the 0/1 answer signal.)

### Fine-tuning dynamics & why we stopped
Reward climbed 0.92 → 1.03 (crossing 1.0 ≈ iter 46 — i.e. driven by the concise-refusal teacher
bonus) while **mean response length collapsed 730 → ~500 tokens**. Val safe-rate plateaued near the
LlamaGuard ceiling (93.8→95.5%). That trio = incipient **over-refusal / length collapse**; we
stopped at ~iter 66. The probes confirm the collapse had **not** yet hurt helpfulness at iter 50
(0% over-refusal, reasoning intact) — stopping there banked the safety gain before the trade-off.

### Figures
- `results/figs/safety_before_after.png` — the headline bars (+CI)
- `results/figs/fine_tuning_dynamics.png` — reward↑ / length↓ / val safe-rate
- `results/figs/capability_and_channel.png` — channel split + reasoning/over-refusal
- `experiments/<run>/plots/{reward_curve,safety_eval_curve}.png` + live wandb (`es-finetuning`)

## Takeaways
1. **ES fine-tuning works for safety alignment with tiny data** — 30 prompts, 50 iters, a single
   local reward model (LlamaGuard), no gradients/backprop, and it **transfers to plain chat**.
2. **It internalizes a prompted behavior into weights** — the guardrail scaffold at train time
   became a (partial) weight-level prior; ES closed ~50% of the prompt→weight gap.
3. **Safety came free of capability** at iter 50 — reasoning held, zero over-refusal.
4. **Ceiling not reached** — ES (16.4%) still trails Qwen (10.5%) and the prompt ceiling (8.1%).
   Going further needs care: drop `--teacher-shaping` (it drives the length collapse) and mix a few
   **benign** prompts into training to keep pushing safety without tipping into over-refusal.
5. **CoT lags the answer** — closing the `<think>` gap needs a stronger CoT-level reward than the
   current capped-0.5 penalty.

---

## Reproduce
```bash
# 1. datasets (train/val/test + probes)
es/bin/python scripts/build_safety_datasets.py
# 2. teacher refusals (reward shaping + reference)
CUDA_VISIBLE_DEVICES=7 es/bin/python scripts/gen_qwen_teacher.py
# 3. ES fine-tune (train w/ guardrail, val selection)
scripts/run_es_safety.sh            # full 1000 iters   (scripts/run_es_safety.sh smoke = quick)
# 4. final eval (plain chat, test split, Claude judge) + probes
scripts/run_all_eval.sh <best_checkpoint.pth>
# 5. training curves
es/bin/python scripts/plot_training.py --latest
```
(`es/bin/python` = the `~/es_reasoning/es` venv; keys in `safety/.env`.)
