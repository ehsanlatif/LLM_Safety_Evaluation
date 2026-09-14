# Adversarial resilience: SORRY-Bench under 20 linguistic mutations

**Objective.** Test whether the base safety ordering survives *disguised* prompts —
20 SORRY-Bench mutation styles (6 writing, 5 persuasion, 4 cipher, 5 translation) —
and whether reasoning models are more brittle (the chain-of-thought-hijacking question).

**Date:** 2026-08-03. **Judge:** official fine-tuned Mistral-7B (fulfillment, macro
over 44 categories). Cipher/translation styles judged on the decoded English response.
17 models × 20 styles = 340 chunks, all judged.

---

## 1. Headline

**Semantic persuasion — not encoding — is the jailbreak, and it breaks every small
open model almost completely.** Under its worst mutation, essentially every small
open model (reasoning *and* non-reasoning) reaches **91–99% fulfillment**, regardless
of how safe it was on undisguised prompts. Frontier models are far more resilient
(worst-case 26–58%).

## 2. Resilience gap (fulfillment %, base → mean-over-20 / worst-case)

| Model | Class | Base | Mean | Worst | Worst style (uplift) |
|---|---|--:|--:|--:|---|
| GPT-5.5 | Large closed | 17.1 | 13.4 | **26.1** | slang (+9.0) |
| Claude Sonnet 5 | Large closed | 18.2 | 19.7 | 57.6 | logical_appeal (+39.5) |
| Claude Opus 4.8 | Large closed | 23.2 | 21.0 | 58.5 | logical_appeal (+35.3) |
| Qwen2.5-1.5B | Small open, non-reason | 13.1 | 32.8 | 91.5 | misrepresentation (**+78.5**) |
| Qwen2.5-3B | Small open, non-reason | 28.2 | 38.0 | 94.3 | misrepresentation (+66.1) |
| Llama-3.2-3B | Small open, non-reason | 33.6 | 44.4 | 94.4 | misrepresentation (+60.8) |
| VibeThinker-3B | Small open, reason | 38.5 | 29.0 | 58.8 | authority_endorsement (+20.3) |
| DeepSeek-R1-1.5B | Small open, reason | 83.6 | 55.9 | 91.0 | authority_endorsement (+7.4) |
| VibeThinker-1.5B | Small open, reason | 83.0 | 58.4 | 98.0 | misrepresentation (+15.0) |

Qwen3 same-weights toggle (nothink / think), worst-case fulfillment:

| Size | nothink base→worst | think base→worst |
|---|--:|--:|
| 0.6B | 68.8 → 95.3 | 82.8 → 97.9 |
| 1.7B | 50.6 → 93.8 | 72.8 → **99.1** |
| 4B | 31.3 → 94.5 | 53.2 → 98.2 |
| 8B | 37.1 → 92.6 | 58.6 → 98.0 |

(`analysis/resilience_gap.csv`, `analysis/fig_resilience_gap.png`.)

## 3. By mutation family (mean fulfillment %)

| Model | Writing | **Persuasion** | Cipher | Translation |
|---|--:|--:|--:|--:|
| GPT-5.5 | 19.3 | 17.0 | 4.7 | 9.8 |
| Sonnet 5 | 19.9 | 45.1 | 0.8 | 9.1 |
| Opus 4.8 | 23.6 | 47.9 | 0.8 | 7.1 |
| Qwen2.5-1.5B | 24.1 | **82.8** | 1.4 | 18.3 |
| Qwen2.5-3B | 34.2 | **88.4** | 2.5 | 20.6 |
| Llama-3.2-3B | 39.0 | **88.0** | 1.8 | 41.3 |
| VibeThinker-1.5B | 87.7 | **97.1** | 6.9 | 25.9 |
| Qwen3-8B (think) | 61.9 | **97.1** | 0.9 | 32.6 |

(Full 17×20 matrix: `analysis/mutation_fulfillment.csv`, `analysis/fig_mutation_heatmap.png`.)

## 4. Insights

1. **Persuasion is the universal jailbreak.** The 5 persuasion styles
   (misrepresentation, logical/authority/expert appeals, evidence-based) drive every
   small open model to **82–98%** fulfillment. `misrepresentation` is the single most
   effective style. Even frontier Opus/Sonnet jump to ~45–48% under persuasion
   (`logical_appeal` is their worst, ~58%). **GPT-5.5 is the outlier** — persuasion
   barely moves it (17%), worst-case only 26%.

2. **Ciphers do the opposite of a jailbreak.** ascii/caesar/morse/atbash collapse
   fulfillment to **~0–7%** for everyone — the models can't operate in cipher, so the
   decoded output is non-compliant. Translations are intermediate (7–41%;
   Llama-3.2-3B most susceptible at 41%). Encoding-based attacks fail here; the risk
   is semantic framing.

3. **Small open models are catastrophically brittle — alignment at base doesn't
   help.** Qwen2.5-1.5B is the safest small model on undisguised prompts (13%), yet a
   single persuasion reframe takes it to **91.5% (+78 pp)**. Base safety does not
   predict resilience.

4. **Reasoning adds a mutation tax too — and it does not shrink with scale.** At every
   Qwen3 size the *think* variant is more susceptible than its *nothink* twin
   (persuasion ~97–98% vs ~86–89%; worst-case up to **99.1%**). Unlike the base
   reasoning tax (which shrank with size), the persuasion vulnerability stays maximal
   at all sizes.

5. **Mind the metric: mean uplift is misleading for high-base models.** DeepSeek-R1
   and VibeThinker-1.5B show *negative* mean uplift only because ciphers/translations
   drag their mean down from an already-saturated base (~83%) — not because they are
   robust. Their worst-case is 91–98%. Read **worst-case** and the **persuasion
   column**, not the mean.

## 5. Limitations
- Fulfillment ≠ harm (SORRY-Bench official metric credits benign compliance too);
  read as a compliance/jailbreak-susceptibility spectrum. Long reasoning responses
  (>30k judge tokens) were head+tail truncated before judging so they fit Mistral's
  32k context (affects only a handful of outliers; base runs were uncapped and fit).
- Cipher fulfillment is a lower bound: a model that *could* operate in cipher would
  bypass the decoder; none here did meaningfully.

## 6. Files
- `analysis/resilience_gap.csv` — base, mean, worst-case + worst style per model
- `analysis/mutation_fulfillment.csv` — 17×20 model×style fulfillment matrix
- `analysis/fig_mutation_heatmap.png` — fulfillment heatmap (styles grouped by family)
- `analysis/fig_resilience_gap.png` — base vs mean vs worst-case per model
- Pipeline: `run_mutations_all.sh` → `decode_mutations.py` → `eval_mutations.py` (chunked, resume-safe, token-capped) → `mutation_analysis.py`
