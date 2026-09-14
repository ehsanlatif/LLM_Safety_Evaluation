# BayLearn 2026 abstract — build & submission notes

Files: `baylearn_abstract.tex`, `references.bib`, `figs/`.

## Format requirements (from baylearn-org.github.io)
- **2-page PDF**, **NeurIPS 2023 format**; **+1 page** allowed for references/acks only.
- **Double-blind**: no author names/affiliations/identifying links. Keep the
  `Anonymous Author(s)` block; don't add a de-anonymizing repo URL.
- **Non-archival**; submit PDF via CMT: https://cmt3.research.microsoft.com/BAYLEARN2026
- Over 2 pages, wrong format, or non-anonymized ⇒ desk reject.

## Build
Easiest: **Overleaf** → new project → "NeurIPS 2023" template → replace the
`.tex`/`.bib`, upload `figs/`. It already has `neurips_2023.sty`.

Local:
```bash
# get the style file (neurips_2023.sty) from the NeurIPS 2023 template, place it here
pdflatex baylearn_abstract && bibtex baylearn_abstract && pdflatex baylearn_abstract && pdflatex baylearn_abstract
```

## Pre-submission checklist
- [ ] **Body ≤ 2 pages** (references may spill to p.3). If it runs over, in order:
      shrink `fig_final_evaluators` (`width=0.85\linewidth`), cut finding F4's
      last sentence, or trim Related Work to 3 lines.
- [ ] No identifying info anywhere (author, funding, repo link, dataset path).
- [ ] Numbers match `analysis/FINAL_master_table.csv`.
- [ ] Model IDs stated as run; note AILuminate grade is a LlamaGuard-2 reproduction
      (not the official MLCommons ensemble) — already flagged in Limitations.
- [ ] `neurips_2023.sty` present; compiles without the `final` option (stays anonymous).

## Optional strengthening before the deadline
- Run the 20 SORRY-Bench mutation styles → an adversarial-robustness / resilience-gap
  result (directly engages H-CoT and the reasoning-safety debate).
- Add a second small model per class (e.g. Llama-3.2-3B, a second reasoning model)
  to show the reasoning≠safety pattern isn't a single-model artifact.
- Human-label ~100 responses to report LLM-judge agreement (κ) — strengthens the
  methodology and preempts the "why trust the judge" question.
