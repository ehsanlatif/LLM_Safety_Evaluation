#!/usr/bin/env python3
"""
Pre-flight for the OLMo 3 7B safety sweep.

Loads ONLY the tokenizer + config for each of the 12 OLMo variants (no weights, no
GPU) and verifies each repo is reachable, is an Olmo3 checkpoint, and whether it
ships a chat template. This catches gated/missing repos and the base/RL-Zero
"no chat template" case BEFORE we spend GPU-hours generating.

Writes analysis/olmo_preflight.json and exits non-zero if any variant fails to load,
so run_olmo_all.sh stops early.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import run_benchmarks as rb

OUT = Path("analysis"); OUT.mkdir(exist_ok=True)


def main() -> int:
    rb.load_env_file()  # HF_Token from .env (normalized to HF_TOKEN) for gated/rate-limited repos
    import olmo_compat
    olmo_compat.register_olmo2_retrofit()  # alias 'olmo2-retrofit' (RL-Zero-Mix) -> Olmo3
    from transformers import AutoConfig, AutoTokenizer

    report, failures = {}, []
    for name, (hf_id, tmpl, max_new) in rb.OLMO_MODELS.items():
        entry = {"hf_id": hf_id, "configured_template": tmpl, "max_new_tokens": max_new}
        try:
            cfg = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
            tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
            arch = list(getattr(cfg, "architectures", []) or [])
            has_ct = getattr(tok, "chat_template", None) not in (None, "")
            # Effective mode actually used by run_local_vllm.to_prompt():
            #   "raw" -> verbatim prompt; else chat template (with graceful fallback).
            resolved = "raw" if tmpl == "raw" or not has_ct else tmpl
            entry.update(
                architectures=arch,
                has_chat_template=has_ct,
                resolved_template=resolved,
                max_position_embeddings=getattr(cfg, "max_position_embeddings", None),
                is_olmo3=any("Olmo3" in a for a in arch),
                ok=True,
            )
            if tmpl != "raw" and not has_ct:
                entry["warning"] = "no chat template on disk -> will fall back to raw prompt"
        except Exception as e:  # gated, missing, network, etc.
            entry.update(ok=False, error=f"{type(e).__name__}: {e}")
            failures.append(name)
        report[name] = entry
        flag = "OK " if entry["ok"] else "FAIL"
        extra = entry.get("error") or entry.get("warning") or ""
        print(f"[{flag}] {name:26s} {hf_id:34s} "
              f"tmpl={entry.get('resolved_template','?'):5s} {extra}", flush=True)

    (OUT / "olmo_preflight.json").write_text(json.dumps(report, indent=2))
    print(f"\nWrote {OUT/'olmo_preflight.json'}  ({len(report)-len(failures)}/{len(report)} loadable)")
    if failures:
        print("FAILED to load:", ", ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
