#!/usr/bin/env python3
"""Assemble the self-contained experiment-v2 report page (results/report_v2.html):
narrative + the three figs_v2 PNGs embedded as base64 data URIs. Theme-aware,
no external assets."""
import base64
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "results" / "figs_v2"
OUT = ROOT / "results" / "report_v2.html"


def img(name):
    b64 = base64.b64encode((FIGS / name).read_bytes()).decode()
    return f"data:image/png;base64,{b64}"


FIG1, FIG2, FIG3 = img("fig1_safety_before_after.png"), img("fig2_capability.png"), img("fig3_training_dynamics.png")
FIG4 = img("fig4_train_val.png")

HTML = f"""<title>VibeThinker ES Safety v2</title>
<style>
:root {{
  --bg:#F6F7F9; --surface:#FFFFFF; --ink:#191C20; --muted:#59616B; --line:#E2E6EA;
  --accent:#0072B2; --accent2:#D55E00; --base:#5A5A5A;
  --warn:#8A5D00; --warn-bg:#FBF1D8; --warn-line:#E9D28A;
  --good:#2E7D57; --good-bg:#E7F4EC; --good-line:#B6DDC6;
  --figframe:#FFFFFF;
}}
@media (prefers-color-scheme:dark) {{
  :root:not([data-theme="light"]) {{
    --bg:#121417; --surface:#1A1E22; --ink:#E7EBEF; --muted:#9AA4AF; --line:#2A2F35;
    --accent:#4BA3D8; --accent2:#E8813F; --base:#9098A0;
    --warn:#E0B24A; --warn-bg:#241F12; --warn-line:#4A3D1C;
    --good:#5FBF8C; --good-bg:#132018; --good-line:#274B37;
    --figframe:#FFFFFF;
  }}
}}
:root[data-theme="dark"] {{
  --bg:#121417; --surface:#1A1E22; --ink:#E7EBEF; --muted:#9AA4AF; --line:#2A2F35;
  --accent:#4BA3D8; --accent2:#E8813F; --base:#9098A0;
  --warn:#E0B24A; --warn-bg:#241F12; --warn-line:#4A3D1C;
  --good:#5FBF8C; --good-bg:#132018; --good-line:#274B37;
  --figframe:#FFFFFF;
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0; background:var(--bg); color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  line-height:1.62; font-size:17px; -webkit-font-smoothing:antialiased;
}}
.wrap {{ max-width:760px; margin:0 auto; padding:64px 24px 96px; }}
.eyebrow {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12.5px;
  letter-spacing:.14em; text-transform:uppercase; color:var(--muted); margin:0 0 14px; }}
h1 {{ font-family:ui-serif,Georgia,"Times New Roman",serif; font-weight:650;
  font-size:2.35rem; line-height:1.12; letter-spacing:-.01em; margin:0 0 10px;
  text-wrap:balance; }}
h2 {{ font-family:ui-serif,Georgia,serif; font-weight:640; font-size:1.5rem;
  margin:56px 0 4px; letter-spacing:-.01em; }}
h2 .n {{ color:var(--accent); font-family:ui-monospace,monospace; font-size:1rem;
  margin-right:.6em; vertical-align:.12em; }}
.sub {{ color:var(--muted); font-size:1.06rem; margin:0 0 8px; }}
p {{ margin:14px 0; }}
a {{ color:var(--accent); }}
strong {{ font-weight:660; }}
code {{ font-family:ui-monospace,Menlo,monospace; font-size:.86em;
  background:color-mix(in srgb,var(--ink) 7%,transparent); padding:.12em .38em; border-radius:4px; }}
.verdict {{ background:var(--warn-bg); border:1px solid var(--warn-line);
  border-left:4px solid var(--warn); border-radius:10px; padding:18px 20px; margin:26px 0 8px; }}
.verdict .tag {{ font-family:ui-monospace,monospace; font-size:12px; letter-spacing:.12em;
  text-transform:uppercase; color:var(--warn); font-weight:600; }}
.verdict p {{ margin:6px 0 0; font-size:1.02rem; }}
.verdict.good {{ background:var(--good-bg); border-color:var(--good-line); border-left-color:var(--good); }}
.verdict.good .tag {{ color:var(--good); }}
.metric.best {{ border-color:var(--accent2); box-shadow:inset 0 0 0 1px var(--accent2); }}
tr.win td {{ background:color-mix(in srgb,var(--accent2) 9%,transparent); }}
tr.win td.pool {{ color:var(--accent2); }}
.metrics {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin:28px 0 8px; }}
.metric {{ background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:16px 16px 14px; }}
.metric .lab {{ font-size:12.5px; color:var(--muted); font-weight:600; display:flex; align-items:center; gap:7px; }}
.metric .dot {{ width:10px; height:10px; border-radius:50%; display:inline-block; }}
.metric .big {{ font-family:ui-monospace,monospace; font-variant-numeric:tabular-nums;
  font-size:1.9rem; font-weight:600; margin-top:8px; letter-spacing:-.02em; }}
.metric .cap {{ font-size:12.5px; color:var(--muted); margin-top:2px; font-variant-numeric:tabular-nums; }}
figure {{ margin:26px 0 8px; }}
figure .frame {{ background:var(--figframe); border:1px solid var(--line); border-radius:12px;
  padding:14px; overflow-x:auto; }}
figure img {{ display:block; width:100%; height:auto; max-width:100%; }}
figcaption {{ color:var(--muted); font-size:13.5px; margin-top:10px; line-height:1.5; }}
figcaption b {{ color:var(--ink); font-weight:620; }}
.tablewrap {{ overflow-x:auto; margin:20px 0; }}
table {{ border-collapse:collapse; width:100%; font-size:14.5px;
  font-variant-numeric:tabular-nums; }}
th,td {{ text-align:right; padding:9px 12px; border-bottom:1px solid var(--line); white-space:nowrap; }}
th:first-child,td:first-child {{ text-align:left; }}
thead th {{ color:var(--muted); font-weight:600; font-size:12.5px; letter-spacing:.02em;
  border-bottom:1.5px solid var(--line); }}
tbody tr:last-child td {{ border-bottom:none; }}
td.pool {{ font-weight:660; color:var(--ink); }}
.ci {{ color:var(--muted); font-size:12px; }}
ul {{ padding-left:22px; }} li {{ margin:7px 0; }}
.foot {{ margin-top:60px; padding-top:20px; border-top:1px solid var(--line);
  color:var(--muted); font-size:13px; }}
.pill {{ display:inline-block; font-family:ui-monospace,monospace; font-size:11.5px;
  padding:2px 9px; border-radius:999px; border:1px solid var(--line); color:var(--muted); }}
</style>

<div class="wrap">
  <p class="eyebrow">Evolution-Strategies · VibeThinker-3B · 2026-08-15</p>
  <h1>ES safety fine-tuning v2: Opus reward, balanced data, AWD</h1>
  <p class="sub">Does a granular Claude-Opus reward and a balanced 30-sample set beat the v1
  recipe — and does Anchored Weight Decay let ES improve safety while preserving reasoning?</p>

  <div class="verdict good">
    <span class="tag">Result — AWD makes ES safety fine-tuning work</span>
    <p>At <b>300 iterations</b>, <b>plain ES did nothing</b> to safety (pooled 13.6%→13.0%, p=0.87),
    but <b>ES&nbsp;+&nbsp;AWD roughly halved</b> the plain-chat unsafe rate to <b>6.5%</b>
    (−7.1&nbsp;pp, p=0.039 vs baseline; p≈0.05 vs the matched no-AWD run; anywhere-metric
    p=0.014 vs no-AWD). Same data, same reward, same budget — the <b>only</b> difference is the AWD
    anchor. Reasoning held within noise; over-refusal stayed 0%.</p>
  </div>

  <div class="metrics">
    <div class="metric"><div class="lab"><span class="dot" style="background:var(--base)"></span>Baseline</div>
      <div class="big">13.6%</div><div class="cap">pooled unsafe · MATH 95%</div></div>
    <div class="metric"><div class="lab"><span class="dot" style="background:var(--accent)"></span>ES no-AWD</div>
      <div class="big">13.0%</div><div class="cap">pooled unsafe · MATH 92.5%</div></div>
    <div class="metric best"><div class="lab"><span class="dot" style="background:var(--accent2)"></span>ES +AWD</div>
      <div class="big">6.5%</div><div class="cap">pooled unsafe · MATH 90.5%</div></div>
  </div>

  <h2><span class="n">01</span>Setup</h2>
  <p>Two matched ES runs on VibeThinker-3B, identical except AWD. <strong>Granular Claude-Opus
  reward</strong> (1–5 fulfillment/helpfulness rubric → reward in [0,1], replacing v1's binary
  LlamaGuard); <strong>balanced 30-sample set</strong> — 15 harmful drawn uniformly across
  {{SORRY, AILuminate}} × {{base-safe, base-unsafe}} plus 15 benign folded in as an anti-over-refusal
  signal; <strong><code>--teacher-shaping 0</code></strong>; <strong>300 iterations</strong>, pop 30,
  batch 10. Run A: no AWD. Run B: <strong>Anchored Weight Decay</strong> (<code>kschweig/es-awd</code>,
  l2, λ=10 — decay toward the frozen t=0 model). Checkpoint = best small-val (iteration&nbsp;200 for
  both). Eval: plain chat, untouched test split, Claude-Opus judge, <strong>greedy</strong>, 95% CI.</p>

  <h2><span class="n">02</span>Safety — plain-chat, untouched test split</h2>
  <figure><div class="frame"><img alt="Safety before/after with 95% CIs" src="{FIG1}"></div>
    <figcaption><b>Fig 1.</b> +AWD (vermillion) is lowest on every benchmark; baseline and no-AWD
    are indistinguishable. Answer-level unsafe %, 95% Wilson intervals.</figcaption>
  </figure>
  <div class="tablewrap"><table>
    <thead><tr><th>Model (plain chat)</th><th>SORRY-Bench</th><th>AILuminate</th><th>Pooled (answer)</th><th>Anywhere</th></tr></thead>
    <tbody>
      <tr><td>Baseline</td><td>18.8 <span class="ci">[12–28]</span></td><td>6.5 <span class="ci">[3–15]</span></td><td class="pool">13.6 <span class="ci">[9–20]</span></td><td>9.0</td></tr>
      <tr><td>ES — no AWD</td><td>17.0 <span class="ci">[11–26]</span></td><td>8.1 <span class="ci">[4–17]</span></td><td class="pool">13.0 <span class="ci">[9–19]</span></td><td>12.5</td></tr>
      <tr class="win"><td>ES — +AWD (l2, λ=10)</td><td>9.2 <span class="ci">[5–17]</span></td><td>3.0 <span class="ci">[1–10]</span></td><td class="pool">6.5 <span class="ci">[4–12]</span></td><td>5.5</td></tr>
    </tbody>
  </table></div>
  <div class="tablewrap"><table>
    <thead><tr><th>Comparison</th><th>Answer-level pooled</th><th>Anywhere (n=200)</th></tr></thead>
    <tbody>
      <tr class="win"><td>+AWD vs baseline</td><td>6.5 vs 13.6% · z=−2.06 · <b>p=0.039</b></td><td>5.5 vs 9.0% · p=0.18</td></tr>
      <tr class="win"><td>+AWD vs no-AWD (AWD ablation)</td><td>6.5 vs 13.0% · z=−1.93 · <b>p=0.053</b></td><td>5.5 vs 12.5% · z=−2.45 · <b>p=0.014</b></td></tr>
      <tr><td>no-AWD vs baseline</td><td>13.0 vs 13.6% · p=0.87 (null)</td><td>12.5 vs 9.0% · p=0.26</td></tr>
    </tbody>
  </table></div>
  <p>Plain ES moved nothing and its <code>&lt;think&gt;</code> got <em>worse</em> (9→12.5%). AWD is
  what turned ES into a working safety optimizer.</p>

  <h2><span class="n">03</span>Capability — reasoning &amp; over-refusal</h2>
  <figure><div class="frame"><img alt="MATH accuracy and benign over-refusal" src="{FIG2}"></div>
    <figcaption><b>Fig 2.</b> MATH-200 drifts 95→92.5→90.5% (±~3 pp) — the safety win costs a small,
    roughly-within-noise reasoning decrement, and AWD did not protect MATH better than no-AWD (its
    benefit was safety <em>generalization</em>). Zero benign over-refusal for all three.</figcaption>
  </figure>

  <h2><span class="n">04</span>Training dynamics</h2>
  <figure><div class="frame"><img alt="Reward, response length, and val unsafe over iterations" src="{FIG3}"></div>
    <figcaption><b>Fig 3.</b> Reward climbs ~0.74→0.81 (AWD tracks no-AWD). <b>Length collapse
    recurs</b> at 300 iters even with teacher-shaping 0 — no-AWD falls to ~520 tokens (min 469);
    <b>AWD mitigates it</b> (~600, min 520). Plain ES over-optimizes the tiny reward into short,
    degenerate outputs that don't generalize; AWD's anchor curbs that drift — the likely mechanism
    behind its safety-generalization edge.</figcaption>
  </figure>
  <figure><div class="frame"><img alt="Training reward, length, and validation curves for both runs" src="{FIG4}"></div>
    <figcaption><b>Fig 4.</b> Reward jumps in the first few iters (easy refusals) then does a noisy
    random walk near the optimum — the drift AWD constrains. Held-out val safety is <b>best at
    iteration&nbsp;150</b> for both runs and degrades after (over-optimization of the 30-prompt set,
    tracking the length collapse); the late rise is driven mostly by <b>AILuminate</b> (dashed).</figcaption>
  </figure>

  <h2><span class="n">05</span>Interpretation &amp; limits</h2>
  <ul>
    <li><b>AWD is necessary here, not optional.</b> The only difference between a null result and a
      ~50% unsafe cut is the l2 anchor; it keeps ES near VibeThinker's safety-capable Qwen prior so
      the safety it finds transfers to plain chat.</li>
    <li><b>Budget mattered too</b> — the same recipe at 100 iters (v1) was null for both arms.</li>
    <li><b>Weight-level &amp; prompt-free</b> — trained with the guardrail, evaluated without it.</li>
  </ul>
  <p><strong>Read it as promising, not settled:</strong> one run per condition (no seed replication);
  greedy decoding lowered coverage to 62–88% (the coverage-robust <em>anywhere</em> metric, p=0.014
  vs no-AWD, is the check); MATH-200 has ±~3 pp. <strong>Next:</strong> replicate across 3 seeds,
  ablate λ ∈ {{1,5,10,20}}, and raise max-tokens so greedy eval closes <code>&lt;/think&gt;</code>.</p>

  <div class="foot">
    <span class="pill">pop 30 · batch 10 · 300 iters · σ=1e-3</span>
    <span class="pill">AWD l2 λ=10</span>
    <span class="pill">judge claude-opus-4-8 · greedy eval</span>
    <span class="pill">8× A100</span>
    <p>Method, scripts, and raw summaries in the repo: <code>REPORT_v2.md</code>,
    <code>scripts/run_all_v2.sh</code>, <code>results/</code>. Figures: <code>results/figs_v2/</code>.</p>
  </div>
</div>
"""

OUT.write_text(HTML, encoding="utf-8")
kb = OUT.stat().st_size / 1024
print(f"wrote {OUT}  ({kb:.0f} KB)")
