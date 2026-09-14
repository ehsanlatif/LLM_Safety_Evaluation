# LLM Safety Evaluation

Anonymized code and results for a double-blind submission on the safety of small
open reasoning models. The project was originally organized as several git
branches; for anonymous hosting (single-branch), each branch is provided here as
a **top-level folder**:

| Folder | Contents | Key finding |
|---|---|---|
| [`main/`](main/) | Cross-benchmark, cross-evaluator **safety audit** (7+ models, 3 evaluators, prompt mutations). | Alignment sets the safety level; reasoning adds a causal "safety tax"; persuasion beats encoding as a jailbreak. |
| [`es-baseline/`](es-baseline/) | **Evolution Strategies (no-AWD)** safety fine-tuning of VibeThinker-3B → `es/`. | Plain ES does essentially nothing to safety (null control). |
| [`es-awd/`](es-awd/) | **ES + Anchored Weight Decay (AWD)** → `es/`. | AWD is the difference-maker: ~50% relative cut in the plain-chat unsafe rate. |
| [`rl/`](rl/) | **GRPO** counterpart with the same reward/eval → `rl/`. | Under a noise-reduced, generation-averaged, paired eval, **ES+AWD is the only significant safety win** (−5.5 pp anywhere, p<0.001); **GRPO ≈ baseline** (p=0.857). |

Each folder is a **self-contained snapshot** of its branch (it includes the shared
audit code plus that branch's experiment directory), so it can be read and run on
its own. The RL experiment write-up and reproducibility spec are in
[`rl/rl/REPORT.md`](rl/rl/REPORT.md) and [`rl/rl/REPRODUCIBILITY.md`](rl/rl/REPRODUCIBILITY.md).

> Anonymized for double-blind review: author identity, institution, and absolute
> paths have been removed from all files and git history.
