"""vLLM worker extension for the GRPO rollout engines.

The ES worker_extension syncs weights by saving/loading a state_dict captured
from the vLLM model itself (fused qkv/gate_up names). GRPO instead trains a
*separate* HuggingFace policy model and must push ITS weights (un-fused HF
names) into the vLLM engines each step. vLLM's own `model.load_weights()` is
the loader that maps HF checkpoint names -> the engine's fused parameters and
handles tied embeddings, so we route through it.
"""
import gc
import torch


class RLWorkerExtension:
    """Mixed into each vLLM worker (see ESNcclLLM launch)."""

    def load_hf_weights(self, path):
        """Load an HF-format state_dict (saved by the Learner) into this engine's model.

        Uses vLLM's native load_weights so HF names (q_proj/k_proj/v_proj,
        gate_proj/up_proj, tied lm_head) are correctly mapped to fused params."""
        sd = torch.load(path, map_location="cpu")
        model = self.model_runner.model
        model.load_weights(list(sd.items()))
        del sd
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return True

    def ping(self):
        return True
