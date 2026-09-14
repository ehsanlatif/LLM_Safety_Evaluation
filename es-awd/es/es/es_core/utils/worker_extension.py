import gc
import time
import torch

def _stateless_init_process_group(master_address, master_port, rank, world_size, device):
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup
    pg = StatelessProcessGroup.create(
        host=master_address, port=master_port, rank=rank, world_size=world_size
    )
    return PyNcclCommunicator(pg, device=device)

class WorkerExtension:
    """
    The class for vLLM's worker to inherit from.
    """

    def save_self_initial_weights(self,):
        """Save a copy of itself in the CPU memory."""
        self.initial_weights = {}
        for name, p in self.model_runner.model.named_parameters():
            self.initial_weights[name] = p.detach().clone().cpu()
        print("Initial weights saved.")

    def get_first_weights_preview(self, n_layers: int = 10, n_params: int = 20):
        """Return the first n_params of each of the first n_layers."""
        out = {}
        count = 0
        for name, param in self.model_runner.model.named_parameters():
            if count >= n_layers:
                break
            arr = param.detach().flatten().float().cpu().numpy()
            out[name] = arr[:n_params].tolist()
            count += 1
        
        return out
    
    def get_weight_rms(self):
        """Return RMS(theta) = sqrt(mean(theta^2)) across all parameters."""
        s2 = 0.0
        n = 0
        for _, p in self.model_runner.model.named_parameters():
            # fp32 accumulation for stability
            x = p.detach().float()
            s2 += (x * x).sum().item()
            n  += x.numel()
        rms = (s2 / max(n, 1)) ** 0.5
        return float(rms)


    def restore_self_weights(self, seed, sign, sigma):
        for _, p in self.model_runner.model.named_parameters():
            gen = torch.Generator(device=p.device)
            gen.manual_seed(int(seed))
            noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
            p.data.add_(float(sigma) * noise * -float(sign))
            del noise
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return True


    def init_inter_engine_group(self, master_address: str, master_port: int, rank: int, world_size: int):
        self.inter_pg = _stateless_init_process_group(
            master_address, master_port, rank, world_size, self.device
        )
        return True
    
    
    def broadcast_all_weights(self, src_rank: int):
        for _, p in self.model_runner.model.named_parameters():
            self.inter_pg.broadcast(p, src=int(src_rank), stream=torch.cuda.current_stream())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True

    def update_weights_from_seeds(self, seeds, coeffs, alpha, population_size):
        """
        Mimics the Original implementation's update loop structure:
        Iterate Param -> Iterate Seeds -> Accumulate -> Single Update.
        """
        # seeds and coeffs should be lists of equal length
        # coeffs[i] should be: (alpha / population_size) * normalized_reward
        
        for _, p in self.model_runner.model.named_parameters():
            # float32
            update_accumulator = torch.zeros_like(p.data, dtype=torch.float32)
            
            for i, (seed, sign) in enumerate(seeds):
                gen = torch.Generator(device=p.device)
                gen.manual_seed(int(seed))
                
                # Generate noise (in native precision, usually float16/bfloat16)
                noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
                
                # FIXED: Convert noise to float32 BEFORE multiplication.
                # Previous code: noise.to(torch.float16) * coeffs[i]
                # This caused the tiny update signal (1e-5) to be truncated by FP16 precision limits.
                term = noise.to(torch.float32) * coeffs[i] * sign
                
                # Accumulate in FP32
                update_accumulator.add_(term)
            
            # div by population_size multiply by alpha (scalar)
            update_accumulator.div_(population_size)
            update_accumulator.mul_(alpha)
            # Apply final update to weight (cast back to model dtype at the very end)
            p.data.add_(update_accumulator.to(p.dtype))
            
            del update_accumulator
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return True

    def perturb_self_weights(self, seed, noise_scale, sign):
        """
        Add noise(seed) scaled by sigma_or_scale * coeff (or subtract when negate=True).
        - For exploration:  perturb_self_weights(seed, SIGMA, 1.0, False)
          and restore with restore_self_weights(seed, SIGMA) as before.
        - For ES update:   perturb_self_weights(seed, 1.0, coeff, False)
          where coeff = ALPHA/POPULATION_SIZE * norm_reward.
        """
        scale = float(noise_scale)

        for _, p in self.model_runner.model.named_parameters():
            gen = torch.Generator(device=p.device)
            gen.manual_seed(int(seed))
            noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
            p.data.add_(sign * scale * noise)
            del noise

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        print(f"Weights changed with: sign={sign}; scale={sign * scale}.")


    def save_self_weights_to_disk(self, filepath):
        """Save the current model weights to disk."""
        state_dict_to_save = {}
        for name, p in self.model_runner.model.named_parameters():
            state_dict_to_save[name] = p.detach().cpu()
        torch.save(state_dict_to_save, filepath)
        print(f"Model weights saved to {filepath}.")

    def load_weights_from_disk(self, filepath):
        state_dict = torch.load(filepath, map_location=self.device)
        model = self.model_runner.model
        # Tied-embedding models (Qwen2.5-3B / VibeThinker-3B: tie_word_embeddings=True)
        # do not expose lm_head.weight in named_parameters() (torch de-dups the shared
        # tensor), so save_self_weights_to_disk never wrote it. Backfill it from the input
        # embedding and load non-strict so the tied head is restored correctly.
        if "lm_head.weight" not in state_dict and "model.embed_tokens.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[load_weights_from_disk] missing={list(missing)} unexpected={list(unexpected)}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(0.1)
        return True