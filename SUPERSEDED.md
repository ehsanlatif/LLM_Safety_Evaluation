# Superseded analyses

The v2 reports (`REPORT_v2.md`, `results/report_v2.html`) have been removed. They
were interim summaries written before the repeated-generation study and their
headline figures are contradicted by the label records committed here:

| quantity | v2 report | current (K=8, 6,400 generations) |
|---|---|---|
| baseline pooled answer unsafe | 13.6% | 16.6% |
| ES+AWD pooled answer unsafe | 6.5% | 10.4% |
| MATH-200 baseline / ES / ES+AWD | 95.0 / 92.5 / 90.5 | 93.0 / 91.5 / 88.0 |

No measurement was deleted; only the derived write-ups were. The authoritative
artifacts are the per-generation label records under `rl/rl/results/genavg/`.

## Two greedy evaluation runs

This repository contains two separate greedy executions of the same four
checkpoints. Their full-response unsafe counts out of 200 disagree:

| | baseline | ES | ES+AWD | GRPO |
|---|---|---|---|---|
| run in `es-*/es/es/results` (+ GRPO copy) | 14 | 25 | 11 | 20 |
| run in `rl/rl/rl/results` | 22 | 23 | 18 | 11 |

The recorded checkpoint paths are identical and both used temperature 0, the
same 8192-token cap and the same judge. At most 1 response in 100 is byte
identical across the two. This is batch-dependent floating-point
nondeterminism, and it is why the paper's results come from the K=8 repeated
generation study rather than from either greedy run. Both are kept
deliberately; neither identifies an arm ordering on its own.
