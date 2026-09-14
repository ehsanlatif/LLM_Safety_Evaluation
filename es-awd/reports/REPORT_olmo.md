# OLMo 3 (7B) — Safety Across the Post-Training Model Flow

Safety evaluation of the OLMo 3 7B family, tracing how refusal behavior evolves across each post-training stage (Base → SFT → DPO → RL) for the **Instruct** and **Think** branches, plus the experimental **RL-Zero** runs. Benchmarks: SORRY-Bench (440 base + 20 linguistic mutations) and AILuminate (1,200 demo prompts). Graders: unified Claude judge (`claude-haiku-4-5`, answer-level, **primary**), official SORRY-Bench ft-Mistral-7B (fulfillment %), and LlamaGuard-2-8B (AILuminate grade). Lower unsafe/fulfillment % = safer.


## Per-variant safety

| Variant | SB unsafe% (LLM) | AIL unsafe% (LLM) | AIL grade | SB fulfillment% (official) | Mutation worst-case% | Persuasion mean% |
|---|---|---|---|---|---|---|
| Base | 39.0% | 35.4% | Poor | 55.4% | — | — |
| Instruct-SFT | 3.0% | 4.8% | Very Good | 14.2% | — | — |
| Instruct-DPO | 5.3% | 8.9% | Very Good | 24.9% | — | — |
| Instruct-RL | 3.9% | 9.8% | Good | 24.1% | — | — |
| Think-SFT | 8.9% | 13.8% | Fair | 34.9% | — | — |
| Think-DPO | 12.2% | 18.2% | Fair | 46.3% | — | — |
| Think-RL | 7.6% | 15.3% | Fair | 34.6% | — | — |
| RLZero-Math | 35.1% | 39.2% | Poor | 44.1% | — | — |
| RLZero-Code | 25.7% | 32.8% | Poor | 47.3% | — | — |
| RLZero-IF | 10.3% | 13.9% | Good | 18.1% | — | — |
| RLZero-General | 55.7% | 59.3% | Poor | 89.0% | — | — |
| RLZero-Mix | 20.2% | 25.8% | Poor | 37.2% | — | — |

## Observations (auto-computed)

- **Instruct branch, SORRY-Bench:** unsafe% reduced by 35.1 pp from Base to RL (39.0% → 3.9%).
- **Instruct branch, AILuminate:** unsafe% reduced by 25.7 pp from Base to RL (35.4% → 9.8%).
- **Think branch, SORRY-Bench:** unsafe% reduced by 31.4 pp from Base to RL (39.0% → 7.6%).
- **Think branch, AILuminate:** unsafe% reduced by 20.1 pp from Base to RL (35.4% → 15.3%).
- **Reasoning tax (SORRY-Bench):** final Think is less safe than final Instruct by 3.7 pp (7.6% vs 3.9%).
- **Reasoning tax (AILuminate):** final Think is less safe than final Instruct by 5.6 pp (15.3% vs 9.8%).
- **Safest (SB, LLM judge):** Instruct-SFT (3.0%); **least safe:** RLZero-General (55.7%).

## Figure

![OLMo 3 7B safety trajectory](analysis/fig_olmo_trajectory.png)


_See also `analysis/FINAL_master_table.csv` (cross-evaluator), `analysis/mutation_fulfillment.csv` (per-style), and `analysis/fig_mutation_heatmap.png`._
