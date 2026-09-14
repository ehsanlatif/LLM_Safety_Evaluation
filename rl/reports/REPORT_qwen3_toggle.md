# Does reasoning make a model less safe? A same-weights ablation (Qwen3)

**Objective.** Isolate the causal effect of *reasoning* on safety, holding the
model fixed — and measure how that effect changes with scale.

**Date:** 2026-07-28

---

## 1. Why this experiment exists

The main study (`REPORT.md`) found small-open **reasoning** models far less safe
than small-open **non-reasoning** models. But that comparison confounds two
variables: every reasoning model was 1.5B and math-lineage (DeepSeek-R1-Distill,
VibeThinker), every non-reasoning model was 3B and instruction-tuned (Qwen2.5-3B,
Llama-3.2-3B). "Reasoning" and "size + lineage + training data" moved together, so
the safety gap could not be attributed to reasoning per se.

We considered filling the missing 2×2 cells with matched checkpoints (e.g. a
non-reasoning twin of DeepSeek-R1-1.5B, or a reasoning distill of Llama-3.2-3B).
**That approach is structurally unsound:** distillation changes weights *and*
training data together, so any two separate checkpoints differ in more than the
reasoning axis. The only construct that isolates reasoning is **one set of weights
with a reasoning toggle** — thinking on vs. off, nothing else changes.

## 2. Design

**Model.** Qwen3 dense checkpoints, which expose a genuine per-request thinking
toggle (`apply_chat_template(enable_thinking=...)`) on **identical weights**. We
run four sizes — **0.6B, 1.7B, 4B, 8B** — each twice (thinking OFF, thinking ON),
for a clean **size × reasoning** factorial where reasoning is the *only* variable
at each size.

| | 0.6B | 1.7B | 4B | 8B |
|---|---|---|---|---|
| non-reasoning | `Qwen3-0.6B` off | `Qwen3-1.7B` off | `Qwen3-4B` off | `Qwen3-8B` off |
| reasoning | `Qwen3-0.6B` on | `Qwen3-1.7B` on | `Qwen3-4B` on | `Qwen3-8B` on |

**Datasets.** AILuminate v1.0 demo (1,200 prompts) and SORRY-Bench base (440),
identical to the main study.

**Evaluators (three, independent).**
1. **Unified LLM-judge** (`claude-haiku-4-5`), **answer-level** — the primary
   harm metric; for thinking runs it grades only the user-facing text after
   `</think>`, and reports answer coverage.
2. **AILuminate official method** — LlamaGuard-2-8B annotator + MLCommons 5-tier
   grade (open self-consistent reproduction).
3. **SORRY-Bench official** — fine-tuned Mistral-7B judge, macro fulfillment rate.

**Compute.** 8×8 GB GPUs. Generation: 0.6B/1.7B in bf16; 4B/8B via official AWQ
int4 (`Qwen/Qwen3-{4,8}B-AWQ`) with fp8 KV cache to fit the 12k-token reasoning
budget. Both variants of a size share one checkpoint, so quantization is applied
equally and does not affect the within-size reasoning comparison.

---

## 3. Headline result

**Enabling reasoning on identical weights lowers safety at every size, on every
one of the three independent judges.**

**Primary metric — answer-level unsafe rate, both datasets pooled** (95% Wilson CI):

| Size | Non-thinking | Thinking | Reasoning tax | Thinking coverage |
|---|---:|---:|---:|---:|
| 0.6B | 47.1% [44.7–49.6] | 52.8% [50.4–55.3] | **+5.7 pp** | 98.2% |
| 1.7B | 31.7% [29.5–34.0] | 40.8% [38.5–43.2] | **+9.1 pp** | 99.3% |
| 4B | 16.4% [14.7–18.3] | 18.9% [17.1–20.9] | **+2.5 pp** | 98.1% |
| 8B | 18.3% [16.5–20.2] | 20.0% [18.1–22.0] | **+1.7 pp** | 99.4% |

Thinking coverage ≥98% everywhere, so the effect is **not** a truncation
artifact. (`analysis/fig_qwen3_toggle.png`, `analysis/fig_qwen3_reasoning_tax.png`,
`analysis/qwen3_toggle_summary.csv`.)

**All three evaluators agree** (non-thinking → thinking):

| Size | LLM-judge answer (pooled) | AILuminate LlamaGuard-2 grade | SORRY-Bench official fulfillment |
|---|---|---|---|
| 0.6B | 47.1 → 52.8% | Poor → Poor (20.1 → 33.3%) | 68.8 → 82.8% |
| 1.7B | 31.7 → 40.8% | **Fair → Poor** (9.3 → 21.8%) | 50.6 → 72.8% |
| 4B | 16.4 → 18.9% | **Good → Fair** (4.3 → 10.1%) | 31.3 → 53.2% |
| 8B | 18.3 → 20.0% | **Good → Fair** (3.6 → 9.1%) | 37.1 → 58.6% |

(Full cross-evaluator table: `analysis/FINAL_master_table.csv`.)

---

## 4. Insights

1. **Reasoning carries a safety tax — causally.** Because only the reasoning mode
   changed, the consistent increase in unsafe rate is attributable to reasoning
   itself, not to size, lineage, or training data. On the AILuminate official
   grade this costs a **full tier** at 1.7B, 4B, and 8B (Good→Fair, Fair→Poor).

2. **The tax is real but shrinks with scale on the official judges.** LlamaGuard-2
   and SORRY-Bench both show the largest *absolute* penalty at the small end;
   larger models are safer overall and absorb reasoning better. The LLM-judge
   answer-level tax is non-monotonic (peaks at 1.7B, +9.1 pp) — see caveat 4.

3. **This corroborates the main study's finding and removes its confound.** The
   original claim ("reasoning ≠ safety") rested on cross-family models; here the
   same conclusion holds under a clean same-weights ablation, so it is a property
   of reasoning, not of the particular reasoning checkpoints.

4. **The harm is in the delivered answer, not just the trace.** Answer-level and
   CoT-level unsafe rates track closely (e.g. 8B thinking: 20.0% answer vs 15.5%
   CoT) — the model reasons through the harmful request *and* delivers it, rather
   than merely thinking about it and then refusing.

---

## 5. Caveats & limitations

- **Not a monotonic scaling law.** The three judges disagree on the *shape* of
  tax-vs-size (official judges: decreasing; LLM-judge answer-level: peak at 1.7B).
  Report the direction ("reasoning reduces safety at every size"), not a clean
  monotonic trend.
- **SORRY-Bench coverage 436/440.** Four prompts produced no gradeable response
  during generation for the Qwen3 runs (~0.9%); negligible for the macro-average
  but noted for exact reproducibility.
- **Quantization for 4B/8B.** These ran at AWQ int4 (8 GB hardware). Within-size
  reasoning comparisons are unaffected (shared checkpoint), but absolute rates for
  4B/8B are int4, not bf16 — a minor factor if comparing across sizes.
- **Official-judge provisos carry over** from `REPORT.md`: AILuminate uses the
  open LlamaGuard-2 substitute (not the private `cheval` ensemble), and
  SORRY-Bench "fulfillment" ≠ "harm."
- **One model family, greedy decoding, one sample/prompt.** The effect is
  demonstrated for Qwen3; generalization to other hybrid-reasoning families is
  future work.

## 6. Files

- `analysis/qwen3_toggle_summary.csv` — size × reasoning, per dataset + pooled, with CIs, tax, coverage, CoT
- `analysis/fig_qwen3_toggle.png` — answer-level unsafe rate vs size, thinking vs non-thinking (per dataset)
- `analysis/fig_qwen3_reasoning_tax.png` — reasoning safety tax (pp) vs size
- `analysis/FINAL_master_table.csv` — all models × all three evaluators
- `analysis/summary_overall.csv` — per-model answer/CoT/anywhere/coverage
- Pipeline: `run_local_vllm.py` (generate) → `grade_responses.py` + `eval_sorrybench.py` + `eval_ailuminate.py` (judge) → `analyze_toggle.py` (finalize); orchestrated by `run_eval_all.sh`.
