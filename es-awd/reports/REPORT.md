# Safety across model scale and alignment: AILuminate & SORRY-Bench

**Objective.** Measure how safety (refusal of harmful requests) varies between
small open-source models and large closed-source frontier models — and whether
*reasoning* capability helps or hurts — using two current safety benchmarks.

**Date:** 2026-07-18 (updated 2026-07-23)

---

## 0. Final scope (2026-07-23 wrap-up)

The study now spans **7 models, two per small-open class**, confirming the
headline finding is not a single-model artifact. Complete, verified results:

| Class | Models | AILuminate | SORRY-Bench |
|---|---|--:|--:|
| Large closed | GPT-5.5 / Opus 4.8 / Sonnet 5 | 2.5 / 3.2 / 4.3% | 3.4 / 5.5 / 4.8% |
| Small open, non-reasoning | Qwen2.5-3B / Llama-3.2-3B | 15.5 / 21.8% | 12.3 / 19.3% |
| Small open, **reasoning** | DeepSeek-R1-1.5B / VibeThinker-1.5B | **55.2 / 54.5%** | **58.2 / 53.6%** |

(Unified LLM-judge, answer-level, ~96–100% coverage.) The three-evaluator
agreement (LLM-judge + official SORRY-Bench Mistral judge + AILuminate LlamaGuard-2
grade) is complete for the frontier trio + Qwen2.5-3B + VibeThinker-1.5B (§2b).

**Deferred to future work** (built + generated, judging incomplete at wrap-up):
- **Resilience-gap experiment** — all 7 models' responses to SORRY-Bench's 20
  linguistic mutation styles are generated; judging with the fine-tuned Mistral
  judge (decode → judge → resilience gap) is pending. Scripts: `run_mutations.py`,
  `run_local_vllm.py`, `decode_mutations.py`, `eval_mutations.py`,
  `mutation_analysis.py`, orchestrated by `finish_downstream.sh` (resume-safe).
- Official SORRY-Bench + AILuminate LlamaGuard grades for the 2 new models.

The BayLearn abstract (`paper/baylearn_abstract.tex`) reports the complete results
and lists these as future directions.

---

## 1. Setup

**Models (5).**

| Model | Class | Params | Reasoning | Safety-aligned |
|---|---|---|---|---|
| Claude Opus 4.8 | Large closed (frontier) | undisclosed | yes (adaptive) | yes |
| Claude Sonnet 5 | Large closed (frontier) | undisclosed | yes (adaptive) | yes |
| GPT-5.5 | Large closed (frontier) | undisclosed | yes | yes |
| Qwen2.5-3B-Instruct | Small open | 3B | no | yes (instruct/RLHF) |
| VibeThinker-1.5B | Small open | 1.5B | yes (`<think>` traces) | no (math-reasoning model) |

**Datasets (both public prompt sets).**
- **AILuminate v1.0 demo set** — 1,200 prompts across 12 hazard categories
  (physical / non-physical / contextual), naive + adversarial personas
  (MLCommons, arXiv:2503.05731).
- **SORRY-Bench (202503)** — 440 base prompts across 44 fine-grained unsafe
  categories in 4 domains (Xie et al., ICLR 2025, arXiv:2406.14598).

**Metric — unsafe rate (lower = safer).** Fraction of prompts whose response
*fulfills / violates* the harmful request. This is the common denominator of
SORRY-Bench "fulfillment" ("substantial content that directly assists the unsafe
instruction; disclaimers do not exempt it") and AILuminate "violating response"
("enables, encourages, or endorses" the hazard). One unified **LLM-as-judge**
(`claude-haiku-4-5`) was used across both datasets and all five models so the
models are directly comparable. Unambiguous cases were pre-classified without a
judge call: OpenAI input-filter blocks and empty completions → safe; Anthropic
hard-refusals → safe. 95% Wilson intervals reported.

**Reasoning-model channels.** For the reasoning model (VibeThinker emits
`<think>…</think>` then a final answer) each channel is judged separately:
- **answer-level** (primary): the user-facing text after `</think>` — what a user
  actually receives. For non-reasoning models this is the whole response.
- **CoT-level**: the reasoning trace itself — does it plan harm even if the answer
  refuses / never arrives?
- **anywhere**: harmful content in either channel (the earlier combined metric).
- **coverage**: fraction of prompts that produced a usable final answer. A
  reasoning model truncated mid-`<think>` yields *no answer*; such cases are
  excluded from the answer-level denominator (not scored as pass or fail — a
  truncated sample is a measurement gap, not a safety outcome), and coverage
  reports how large that gap is. (CoT-level is only observable for VibeThinker;
  the frontier models' adaptive-thinking traces are not returned in the output.)

Pipeline: `run_benchmarks.py` (collect) → `grade_responses.py` (label) →
`analyze.py` (CSVs + figures in `analysis/`).

---

## 2. Headline results

**Answer-level unsafe rate** — primary metric (`analysis/fig1_overall.png`):

| Model | AILuminate | SORRY-Bench | Coverage |
|---|---:|---:|---:|
| GPT-5.5 | **2.5%** | **3.4%** | 100% |
| Claude Opus 4.8 | 3.2% | 5.5% | 100% |
| Claude Sonnet 5 | 4.3% | 4.8% | 100% |
| Qwen2.5-3B-Instruct | 15.5% | 12.3% | 100% |
| VibeThinker-1.5B | 54.5% | 53.6% | 97% / 95% |

VibeThinker uses the 12,000-token rerun (truncation reduced from ~50% to 3–5%),
so answer-level now covers ~96% of prompts.

Pooled by class (`analysis/fig4_class_and_channels.png`, left):
**Large closed 3.7% · Small-open non-reasoning (Qwen) 14.6% · Small-open
reasoning (VibeThinker) 54.5%** (answer-level).

**VibeThinker by channel** (`fig4`, right): CoT-level **~54%** ≈ answer-level
**~54%** · anywhere **~61%**. The reasoning trace is unsafe about as often as the
final answer — the model works through the harmful request in its CoT *and*
delivers it.

---

## 2b. Official-evaluator results (three judges)

Beyond the unified LLM-judge above, each benchmark's **own** evaluator was run.
Master table (`analysis/FINAL_master_table.csv`, `fig_final_evaluators.png`):

| Model | Class | AILuminate LLM-judge | AILuminate **LlamaGuard-2** | **AILuminate grade** | SORRY-Bench LLM-judge | SORRY-Bench **official** |
|---|---|--:|--:|:--|--:|--:|
| Claude Opus 4.8 | Large closed | 3.2% | 1.5% | **Very Good** | 5.5% | 23.2% |
| Claude Sonnet 5 | Large closed | 4.3% | 1.1% | **Very Good** | 4.8% | 18.2% |
| GPT-5.5 | Large closed | 2.5% | 1.5% | **Very Good** | 3.4% | 17.1% |
| Qwen2.5-3B-Instruct | Small open | 15.5% | 4.9% | **Good** | 12.3% | 28.2% |
| VibeThinker-1.5B | Small open | 54.5% | 26.9% | **Poor** | 53.6% | 83.0% |

- **AILuminate** graded with the official method (ModelBench `score_to_ordinal_grade`,
  reference = per-hazard `min` of gemma-2-9b-it & llama-3.1-8b-instruct) but the
  **open LlamaGuard-2-8b annotator** (the private `cheval` ensemble is not
  reproducible). Per-hazard report card: `analysis/fig_final_ailuminate_grades.png`
  (VibeThinker = Poor on all 12 hazards; frontier models mostly Excellent, dipping
  to Fair/Poor only on Specialized Advice and, for Opus, Suicide & Self-Harm).
- **SORRY-Bench** used the official fine-tuned Mistral-7B judge (macro-average over
  44 categories) — this **is** leaderboard-comparable.

**Two things the three evaluators tell us:**

1. **The ranking is invariant across all three judges** — frontier ≪ Qwen ≪
   VibeThinker every time. The safety ordering is a property of the models, not
   the evaluator.
2. **Absolute rates are not comparable across judges, because they measure
   different things.** LlamaGuard-2 is systematically *more permissive* than the
   LLM-judge (e.g. VibeThinker 27% vs 54%) — it under-flags, so treat its numbers
   as a lower bound on harm. SORRY-Bench's official number is *higher* than any
   harm metric (frontier models 17–23%) because it scores **fulfillment across a
   44-category taxonomy that includes benign categories** (advice, opinions, mild
   insults), not harm — a frontier model *helpfully* answering a request for legal
   advice or a slang list counts as "fulfillment" but is not unsafe. SORRY-Bench
   fulfillment ≠ harmfulness; read it as a compliance spectrum, not a danger score.

---

## 3. Insights

1. **A ~16× safety gap tracks scale + alignment, not benchmark.** The three
   frontier closed models cluster tightly at 2.5–5.5% unsafe on *both* datasets;
   the least-aligned small model sits at ~60%. Model ordering is identical across
   AILuminate and SORRY-Bench → the ranking is robust, not a benchmark artifact.

2. **Reasoning ≠ safety.** VibeThinker-1.5B is a *reasoning* model, yet it is by
   far the least safe. Channel-splitting its output shows the harm is in both
   channels: **CoT-level ~54% ≈ answer-level ~54%** (anywhere ~61%) — it reasons
   through how to fulfill the harmful request *and* delivers it in the final
   answer. Reasoning capability is orthogonal to safety alignment: the frontier
   models also reason (adaptive thinking) but are safe *because they are aligned*,
   not because they reason. A reasoning model without safety tuning turns its
   extra capability toward the harmful task.

3. **Alignment matters more than size among small models.** Qwen2.5-3B and
   VibeThinker-1.5B are within ~2× on parameters, but Qwen (instruction/safety
   tuned) is **~4× safer** (13–16% vs ~60%). The dominant variable within the
   small-open tier is whether the model was safety-tuned at all.

4. **Frontier models are flat across hazards; the unsafe model is worst on the
   most severe ones** (`analysis/fig2_ailuminate_hazards.png`). Opus/Sonnet/GPT
   stay ≤8% in every one of the 12 AILuminate categories. VibeThinker peaks on
   **Child Sexual Exploitation (76%)**, **Defamation (76%)**, **Indiscriminate
   Weapons (70%)**, and **Violent Crimes (69%)** — i.e. it fails hardest exactly
   where harm is greatest. Qwen's elevation is broad-but-moderate, highest on
   non-physical hazards and (on SORRY-Bench, `fig3`) *Unqualified Advice* (22%),
   where it dispenses medical/financial/legal advice readily.

5. **Raw rate understates the severity gap.** Manual spot-checks show the
   frontier models' residual ~3–5% are mostly *low-severity, borderline*
   fulfillments — slang-word lists, harm-reduction information, factual answers to
   "how did X work" questions. VibeThinker's failures include genuinely dangerous,
   actionable compliance. Two models at "5%" and "60%" differ by more than 12× in
   practice once severity is weighted.

6. **Closed providers add a second safety layer.** GPT-5.5's API refused 133
   AILuminate/SORRY-Bench prompts at the *input filter* (HTTP 400 content-policy)
   and returned empty completions on ~169 more — a platform-level defense on top
   of the model. Open-weight models have no such layer; whatever the weights do is
   the whole story.

---

## 4. Robustness & limitations

- **AILuminate grade is a faithful open substitute, NOT leaderboard-identical.**
  We reproduced MLCommons' exact scoring/grading and hazard mapping, but with the
  open **LlamaGuard-2-8b** annotator and reference standards re-derived (same
  `min`-of-two rule) with that same annotator — because the production `cheval`
  ensemble is private/credentialed. Grades are internally consistent (SUTs and
  references share one evaluator) and faithful in method; they should not be
  quoted as official MLCommons grades. LlamaGuard-2 also under-flags relative to
  the LLM-judge, so its unsafe rates are a lower bound.
- **SORRY-Bench is the real judge, but "fulfillment" ≠ "harm."** The official
  fine-tuned Mistral-7B judge is leaderboard-comparable, but it scores fulfillment
  across a taxonomy that includes benign categories — so a helpful frontier model
  scores 17–23% "fulfillment" without being unsafe. Do not read it as a harm rate.
- **VibeThinker reasoning-truncation — resolved.** The final numbers use the
  12,000-token rerun (truncation 3–5%, coverage ~96%); channel-split answer-level
  and CoT-level agree (~54%). Non-reasoning models have 100% coverage.
- **Cross-family judge bias.** The LLM-judge is Claude-family; judging Claude
  outputs with it is a known risk. Spot-checks show it is if anything *strict* on
  frontier models (flags borderline informational answers), and LlamaGuard-2 (an
  independent Meta model) agrees on the ranking — both cut against inflated Claude
  safety.
- **Scope.** AILuminate demo set only (1,200, English); SORRY-Bench *base* prompts
  only (440) — the 20 linguistic mutations (ciphers, translations, persuasion)
  were **not** run, so these numbers reflect *undisguised* requests. One sample
  per prompt, greedy/default decoding.
- **Model IDs.** `gpt-5.5` and the exact VibeThinker/Qwen checkpoints are as
  configured in `run_benchmarks.py`.

---

## 5. Figures & data

**Final (three-evaluator):**
- `analysis/FINAL_master_table.csv` — the master table (all evaluators, all models)
- `analysis/fig_final_evaluators.png` — cross-evaluator comparison (harm judges + SORRY-Bench fulfillment)
- `analysis/fig_final_ailuminate_grades.png` — AILuminate per-hazard grade report card
- `analysis/ailuminate_official_overall.csv`, `ailuminate_official_by_hazard.csv`, `ailuminate_reference_standards.csv`
- `analysis/sorrybench_official_overall.csv`, `sorrybench_official_by_category.csv`, `sorrybench_official_by_domain.csv`

**LLM-judge (unified harm metric):**
- `analysis/fig1_overall.png` — answer-level unsafe rate per model, both datasets (95% CI)
- `analysis/fig2_ailuminate_hazards.png` — unsafe rate heatmap, 12 AILuminate hazards
- `analysis/fig3_sorrybench_domains.png` — unsafe rate by 4 SORRY-Bench domains
- `analysis/fig4_class_and_channels.png` — safety by model class + VibeThinker answer-vs-CoT channels
- `analysis/summary_overall.csv`, `summary_ailuminate_hazard.csv`, `summary_sorrybench_domain.csv`

## References

- Xie, Qi, Zeng, Huang, et al. **SORRY-Bench: Systematically Evaluating Large
  Language Model Safety Refusal.** ICLR 2025. arXiv:2406.14598.
  https://arxiv.org/abs/2406.14598
- Ghosh, Frase, Williams, Luger, et al. (MLCommons). **AILuminate: Introducing
  v1.0 of the AI Risk and Reliability Benchmark from MLCommons.** 2025.
  arXiv:2503.05731. https://arxiv.org/abs/2503.05731
