#!/usr/bin/env python3
"""
Compatibility shim for allenai's "olmo2-retrofit" checkpoints.

Some OLMo 3 7B checkpoints (e.g. Olmo-3-7B-RL-Zero-Mix) were exported by a
pre-release transformers (4.54.0.dev0) that labelled the architecture
`model_type: "olmo2-retrofit"` / `Olmo2RetrofitForCausalLM`. Their config.json is
otherwise byte-for-byte identical to the released Olmo3 config (same hidden size,
layer_types, YaRN rope, sliding window, vocab), so the weights load correctly under
the Olmo3 implementation. No released transformers registers the pre-release name,
so AutoConfig raises "does not recognize this architecture".

register_olmo2_retrofit() aliases the stale model_type onto Olmo3 so both the
transformers path (preflight, annotators) and vLLM can load these checkpoints.
For vLLM, ALSO pass hf_overrides={"architectures": ["Olmo3ForCausalLM"]} so its
architecture dispatch (which keys off the string) picks the Olmo3 model class.
"""
from __future__ import annotations

RETROFIT_TYPE = "olmo2-retrofit"
_done = False


def register_olmo2_retrofit() -> bool:
    """Idempotently alias 'olmo2-retrofit' -> Olmo3. Returns True on success."""
    global _done
    if _done:
        return True
    try:
        from transformers import AutoConfig, AutoModelForCausalLM, Olmo3Config, Olmo3ForCausalLM
    except Exception:
        return False

    class Olmo2RetrofitConfig(Olmo3Config):
        # Subclass so model_type matches the registered key (AutoConfig.register
        # requires config.model_type == the key); behaviour is pure Olmo3.
        model_type = RETROFIT_TYPE

    try:
        AutoConfig.register(RETROFIT_TYPE, Olmo2RetrofitConfig, exist_ok=True)
    except TypeError:  # older signature without exist_ok
        try:
            AutoConfig.register(RETROFIT_TYPE, Olmo2RetrofitConfig)
        except Exception:
            pass
    try:
        AutoModelForCausalLM.register(Olmo2RetrofitConfig, Olmo3ForCausalLM, exist_ok=True)
    except Exception:
        pass
    _done = True
    return True


def vllm_hf_overrides(config):
    """vLLM hf_overrides callable applied to the HF config before model build.

    Fixes two OLMo-3-in-vLLM issues (vLLM 0.19.x + transformers 5.x):
      1. Force architecture -> Olmo3ForCausalLM. No-op for genuine Olmo3; maps the
         pre-release 'Olmo2RetrofitForCausalLM' (RL-Zero-Mix) onto the Olmo3 impl.
      2. Flatten rope_parameters. transformers 5.x emits Olmo3's rope as a nested
         per-layer dict {'sliding_attention': {...}, 'full_attention': {...}}, but
         vLLM's olmo2.py (which also serves Olmo3) expects a FLAT dict with a
         top-level 'rope_theta' (and applies YaRN only on full-attention layers).
         The 'full_attention' sub-dict IS that flat form, so use it verbatim —
         this reproduces Olmo3's intended rope exactly.
    """
    config.architectures = ["Olmo3ForCausalLM"]
    rp = getattr(config, "rope_parameters", None)
    if isinstance(rp, dict) and "full_attention" in rp:
        flat = dict(rp["full_attention"])
        config.rope_parameters = flat
        # Mirror onto rope_scaling/rope_theta for any code path that reads those.
        config.rope_scaling = flat
        if "rope_theta" in flat:
            config.rope_theta = flat["rope_theta"]
    return config
