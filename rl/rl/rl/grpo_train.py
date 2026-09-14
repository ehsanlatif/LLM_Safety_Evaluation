#!/usr/bin/env python3
"""
GRPO safety fine-tuning of VibeThinker-3B — the RL counterpart to the ES run.

Same hypothesis test as ES ("does adding a safety guardrail on top of the model's
reasoning improve safety"), same knobs held fixed so the two are comparable:
  * SAME reward           reward/safety_opus.OpusSafetyScorer (granular Opus 1-5 -> [0,1];
                          harmful=fulfillment, benign=helpfulness, penalize-cot blend)
  * SAME guardrail prompt es_core.train.apply_safety_template / SAFETY_SYSTEM
  * SAME in-loop val       datasets/safety_val_small (safe-rate = mean reward)
  * SAME fast rollouts     Ray + vLLM engines (reused from the ES harness)
Only the optimizer differs: GRPO (group-relative policy gradient with a KL-to-ref
penalty) instead of Evolution Strategies.

Architecture (all Ray actors, 1 GPU each):
  Learner       HF policy model (trainable, bf16) + frozen ref model + AdamW; computes
                the GRPO loss and the optimizer step, then dumps its weights.
  RolloutEngine K vLLM engines that generate G samples/prompt and reload the learner's
                weights each step (RL/rl_worker_extension.RLWorkerExtension.load_hf_weights).
  RayOpusGuard  the Opus reward actor (no GPU), imported unchanged from es_core.train.

Outputs mirror the ES layout under RL/experiments/<run>/ (metrics.jsonl, checkpoints/,
final HF model dir) so scripts/gen_and_grade.py can evaluate it with --model <dir>.
"""
import argparse
import gc
import json
import os
import random
import shutil
import sys
import time
from datetime import datetime

import numpy as np
import torch

ROOT = os.environ.get("ES4SAFETY_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RL_DIR = os.path.join(ROOT, "RL")
for p in (ROOT, os.path.join(ROOT, "es_core"), RL_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
# Ray workers must be able to import rl_worker_extension + es_reasoning + reward.*
os.environ["PYTHONPATH"] = os.pathsep.join(
    [RL_DIR, ROOT, os.path.join(ROOT, "es_core"), os.environ.get("PYTHONPATH", "")])

import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk

try:
    import wandb
except Exception:  # keep the trainer runnable even if wandb is absent
    wandb = None

# Reuse the EXACT template, reward actor, and vLLM engine subclass from ES.
from train import apply_safety_template, SAFETY_SYSTEM, RayOpusGuard, ESNcclLLM  # noqa: E402

EXPERIMENTS_DIR = os.path.join(RL_DIR, "experiments")


# ======================================================================== Learner
@ray.remote(num_gpus=1)
class Learner:
    """Holds the trainable policy + frozen reference model and runs GRPO updates."""

    def __init__(self, model_name, lr, beta, eps_clip, grad_clip, micro_bsz,
                 grad_ckpt=True, kl_free=False):
        self.device = "cuda"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16).to(self.device)
        if grad_ckpt:
            self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False
        self.model.train()

        self.kl_free = bool(kl_free)
        if not self.kl_free:
            self.ref = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16).to(self.device)
            self.ref.eval()
            for p in self.ref.parameters():
                p.requires_grad_(False)
        else:
            self.ref = None

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr,
                                     betas=(0.9, 0.95), weight_decay=0.0)
        self.beta = float(beta)
        self.eps_clip = float(eps_clip)
        self.grad_clip = float(grad_clip)
        self.micro_bsz = int(micro_bsz)
        self.pad_id = self.model.config.eos_token_id or 0

    def ping(self):
        return True

    # ---- build padded micro-batches from raw (prompt_ids, comp_ids, adv) samples ----
    def _pack(self, samples):
        packed = []
        for s in range(0, len(samples), self.micro_bsz):
            chunk = samples[s:s + self.micro_bsz]
            seqs = [c["prompt_ids"] + c["comp_ids"] for c in chunk]
            L = max(len(x) for x in seqs)
            B = len(chunk)
            ids = torch.full((B, L), self.pad_id, dtype=torch.long)
            attn = torch.zeros((B, L), dtype=torch.long)
            cmask = torch.zeros((B, L), dtype=torch.float32)  # 1 at completion token positions
            for i, c in enumerate(chunk):
                pl, cl = len(c["prompt_ids"]), len(c["comp_ids"])
                seq = seqs[i]
                ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
                attn[i, :len(seq)] = 1
                cmask[i, pl:pl + cl] = 1.0
            adv = torch.tensor([c["adv"] for c in chunk], dtype=torch.float32)
            packed.append({"ids": ids.to(self.device), "attn": attn.to(self.device),
                           "mask": cmask[:, 1:].to(self.device), "adv": adv.to(self.device)})
        return packed

    def _token_logp(self, model, mb):
        """Per-token logprob of the realized next token, shape [B, L-1] (memory-lean via CE)."""
        out = model(input_ids=mb["ids"], attention_mask=mb["attn"], use_cache=False)
        logits = out.logits[:, :-1, :]
        targets = mb["ids"][:, 1:]
        B, Lm1, V = logits.shape
        nll = torch.nn.functional.cross_entropy(
            logits.reshape(-1, V).float(), targets.reshape(-1), reduction="none")
        return (-nll).reshape(B, Lm1)

    def update(self, samples, mu):
        packed = self._pack(samples)

        # behavior-policy (old) + reference logprobs, computed once with pre-update weights
        with torch.no_grad():
            for mb in packed:
                mb["old"] = self._token_logp(self.model, mb).detach()
                mb["ref"] = (self._token_logp(self.ref, mb).detach()
                             if self.ref is not None else None)

        total_tok = float(sum(mb["mask"].sum().item() for mb in packed)) or 1.0
        stat = {"loss": 0.0, "kl": 0.0, "ratio": 0.0, "clipfrac": 0.0, "tok": total_tok}

        for _epoch in range(max(1, int(mu))):
            self.opt.zero_grad(set_to_none=True)
            for mb in packed:
                new = self._token_logp(self.model, mb)
                old, mask, adv = mb["old"], mb["mask"], mb["adv"].unsqueeze(1)
                ratio = torch.exp(new - old)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * adv
                surr = torch.min(surr1, surr2)
                if self.ref is not None:
                    d = mb["ref"] - new
                    kl = torch.exp(d) - d - 1.0  # k3 estimator, >= 0
                    per_tok = -(surr - self.beta * kl)
                else:
                    kl = torch.zeros_like(new)
                    per_tok = -surr
                loss = (per_tok * mask).sum() / total_tok
                loss.backward()
                with torch.no_grad():
                    m = mask.sum().clamp(min=1)
                    stat["loss"] += float((per_tok * mask).sum().item())
                    stat["kl"] += float((kl * mask).sum().item())
                    stat["ratio"] += float((ratio * mask).sum().item() / m)
                    stat["clipfrac"] += float(((surr1 > surr2).float() * mask).sum().item() / m)
            gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.opt.step()

        nmb = max(1, len(packed))
        stat["loss"] /= total_tok
        stat["kl"] /= total_tok
        stat["ratio"] /= nmb
        stat["clipfrac"] /= nmb
        stat["grad_norm"] = float(gnorm)
        return stat

    def save_state_to(self, path):
        """Dump policy weights (HF names, cpu bf16) for the engines to reload."""
        sd = {k: v.detach().to("cpu", torch.bfloat16) for k, v in self.model.state_dict().items()}
        tmp = path + ".tmp"
        torch.save(sd, tmp)
        os.replace(tmp, path)
        del sd
        gc.collect()
        return True

    def save_pretrained(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        self.model.save_pretrained(out_dir, safe_serialization=True)
        return True


# ======================================================================== engines
def launch_engines(num_engines, model_name, precision="bfloat16", gpu_mem=0.7):
    pgs = [placement_group([{"GPU": 1, "CPU": 0}], lifetime="detached") for _ in range(num_engines)]
    ray.get([pg.ready() for pg in pgs])
    engines = []
    for pg in pgs:
        strat = PlacementGroupSchedulingStrategy(
            placement_group=pg, placement_group_capture_child_tasks=True,
            placement_group_bundle_index=0)
        engines.append(
            ray.remote(num_cpus=0, num_gpus=0, scheduling_strategy=strat)(ESNcclLLM).remote(
                model=model_name, tensor_parallel_size=1, distributed_executor_backend="ray",
                worker_extension_cls="rl_worker_extension.RLWorkerExtension",
                dtype=precision, enable_prefix_caching=False, enforce_eager=False,
                gpu_memory_utilization=gpu_mem))
    return engines, pgs


# ======================================================================== trainer
class GRPOTrainer:
    def __init__(self, args):
        self.args = args
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in range(8)))
        for k in ("RAY_ADDRESS", "RAY_HEAD_IP", "RAY_GCS_SERVER_ADDRESS"):
            os.environ.pop(k, None)
        ray.init(address="local", include_dashboard=False, ignore_reinit_error=True)

        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0

        # run dir (ES-style) + metrics.jsonl for the existing plotters
        tag = f"grpo-safety-G{args.num_generations}-B{args.prompts_per_step}-lr{args.lr}-kl{args.beta}"
        self.run_dir = os.path.join(EXPERIMENTS_DIR, f"{tag}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        os.makedirs(os.path.join(self.run_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "eval-output"), exist_ok=True)
        self.metrics_path = os.path.join(self.run_dir, "metrics.jsonl")
        self.tokenizer.save_pretrained(self.run_dir)  # so checkpoints are loadable HF dirs

        # ---- Weights & Biases (training-curve plots) ----
        self.run_tag = os.path.basename(self.run_dir)
        self.use_wandb = (args.logging == "wandb") and (wandb is not None)
        if args.logging == "wandb" and wandb is None:
            print("[wandb] requested but not importable; falling back to metrics.jsonl only.")
        if self.use_wandb:
            try:
                wandb.login()
            except Exception as e:
                print(f"[wandb] login() failed ({e}); continuing (offline/anonymous may apply).")
            wandb.init(project=args.wandb_project, name=self.run_tag, dir=self.run_dir,
                       group="grpo-safety", config=vars(args),
                       mode=os.environ.get("WANDB_MODE", "online"),
                       settings=wandb.Settings(start_method="thread"))
            wandb.define_metric("global_step")
            wandb.define_metric("train/*", step_metric="global_step")
            wandb.define_metric("eval/*", step_metric="global_step")
            wandb.define_metric("time/*", step_metric="global_step")

        # weight-sync scratch (tmpfs if available)
        shm = "/dev/shm" if os.path.isdir("/dev/shm") else self.run_dir
        self.sync_path = os.path.join(shm, f"rl_policy_sync_{os.getpid()}.pt")

        # data
        ds = load_from_disk(args.train_dataset)
        self.rows = [{"problem": r["problem"],
                      "kind": r.get("kind") or ("benign" if str(r["answer"]).lower() == "benign" else "harmful")}
                     for r in ds]
        print(f"[data] train rows={len(self.rows)} "
              f"(harmful={sum(r['kind']=='harmful' for r in self.rows)}, "
              f"benign={sum(r['kind']=='benign' for r in self.rows)})")

        # in-loop val (same as ES): DatasetDict of harmful splits -> safe-rate
        self.val = {}
        vd = load_from_disk(args.eval_dataset)
        for name, d in (vd.items() if hasattr(vd, "items") else [("val", vd)]):
            self.val[name] = [{"problem": p, "kind": "harmful"} for p in d["problem"]]

        # actors
        self.engines, self.pgs = launch_engines(args.n_engines, args.model, gpu_mem=args.gpu_mem)
        self.guard = RayOpusGuard.remote(args.judge_model, args.penalize_cot, args.judge_workers)
        ray.get(self.guard.ping.remote())
        self.learner = Learner.remote(
            args.model, args.lr, args.beta, args.eps_clip, args.grad_clip,
            args.micro_bsz, grad_ckpt=True, kl_free=(args.beta <= 0))
        ray.get(self.learner.ping.remote())
        # start engines from the learner's exact weights
        self._sync_weights()
        self.best_safe = -1.0

    # ---------------------------------------------------------------- helpers
    def _sync_weights(self):
        ray.get(self.learner.save_state_to.remote(self.sync_path))
        ray.get([e.collective_rpc.remote("load_hf_weights", args=(self.sync_path,))
                 for e in self.engines])

    def _generate(self, prompts, sampling_params):
        """Round-robin the prompts across engines; return RequestOutputs in prompt order."""
        k = len(self.engines)
        buckets = [[] for _ in range(k)]
        idx_map = [[] for _ in range(k)]
        for i, p in enumerate(prompts):
            buckets[i % k].append(p)
            idx_map[i % k].append(i)
        handles = [self.engines[e].generate.remote(buckets[e], sampling_params, use_tqdm=False)
                   if buckets[e] else None for e in range(k)]
        out = [None] * len(prompts)
        for e in range(k):
            if handles[e] is None:
                continue
            res = ray.get(handles[e])
            for j, o in zip(idx_map[e], res):
                out[j] = o
        return out

    def _log(self, payload):
        rec = {k: v for k, v in payload.items() if isinstance(v, (int, float, str, bool))}
        with open(self.metrics_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if self.use_wandb:
            try:
                wandb.log(rec, commit=True)
            except Exception as e:
                print(f"[wandb] log failed: {e}")

    # ---------------------------------------------------------------- eval
    def evaluate(self, step):
        sp = SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=self.args.max_tokens, seed=40)
        per_ds, all_r = {}, []
        for name, rows in self.val.items():
            prompts = [apply_safety_template(r["problem"]) for r in rows]
            outs = self._generate(prompts, sp)
            pairs, kinds = [], []
            for r, o in zip(rows, outs):
                pairs.append((r["problem"], o.outputs[0].text))
                kinds.append(r["kind"])
            scored = ray.get(self.guard.score_pairs.remote(pairs, kinds))
            rs = [float(x[1]) for x in scored]
            per_ds[name] = float(np.mean(rs)) if rs else 0.0
            all_r += rs
        safe = float(np.mean(all_r)) if all_r else 0.0
        payload = {"global_step": step, "eval/avg/safe_rate": safe, "eval/avg/unsafe_rate": 1.0 - safe}
        for name, v in per_ds.items():
            payload[f"eval/{name}/safe_rate"] = v
        self._log(payload)
        print(f"[eval step {step}] safe_rate={safe:.4f}  " + "  ".join(f"{k}={v:.3f}" for k, v in per_ds.items()))
        if safe > self.best_safe:
            self.best_safe = safe
            ckpt = os.path.join(self.run_dir, "checkpoints", f"best-step{step}-safe{safe:.4f}")
            ray.get(self.learner.save_pretrained.remote(ckpt))
            for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                      "special_tokens_map.json", "added_tokens.json", "chat_template.jinja"):
                src = os.path.join(self.run_dir, f)
                if os.path.exists(src):
                    shutil.copy(src, ckpt)
            print(f"[eval] new best safe_rate={safe:.4f} -> {ckpt}")

    # ---------------------------------------------------------------- train
    def fit(self):
        a = self.args
        order = list(range(len(self.rows)))
        random.shuffle(order)
        cur = 0
        sp = SamplingParams(n=a.num_generations, temperature=a.train_temperature,
                            top_p=a.train_top_p, max_tokens=a.max_tokens)

        for step in range(1, a.n_iterations + 1):
            t0 = time.time()
            # sample a batch of prompts (cycle through a reshuffled epoch)
            if cur + a.prompts_per_step > len(order):
                random.shuffle(order)
                cur = 0
            batch = [self.rows[i] for i in order[cur:cur + a.prompts_per_step]]
            cur += a.prompts_per_step

            prompts = [apply_safety_template(r["problem"]) for r in batch]
            outs = self._generate(prompts, sp)

            # reward every completion with the SAME Opus scorer
            pairs, kinds, group_of = [], [], []
            for gi, (r, o) in enumerate(zip(batch, outs)):
                for comp in o.outputs:
                    pairs.append((r["problem"], comp.text))
                    kinds.append(r["kind"])
                    group_of.append(gi)
            scored = ray.get(self.guard.score_pairs.remote(pairs, kinds))
            rewards = np.array([float(x[1]) for x in scored], dtype=np.float64)

            # group-relative advantages (per prompt)
            adv = np.zeros_like(rewards)
            for gi in range(len(batch)):
                m = np.array([j for j in range(len(group_of)) if group_of[j] == gi])
                g = rewards[m]
                adv[m] = (g - g.mean()) / (g.std() + 1e-4)

            # build learner samples (exact vLLM token ids -> no re-tokenization drift)
            samples, comp_lens = [], []
            p = 0
            for gi, o in enumerate(outs):
                pids = list(o.prompt_token_ids)
                for comp in o.outputs:
                    cids = list(comp.token_ids)
                    if cids:
                        samples.append({"prompt_ids": pids, "comp_ids": cids, "adv": float(adv[p])})
                        comp_lens.append(len(cids))
                    p += 1

            stat = ray.get(self.learner.update.remote(samples, a.mu)) if samples else {}
            self._sync_weights()

            payload = {
                "global_step": step,
                "train/reward/mean": float(rewards.mean()), "train/reward/std": float(rewards.std()),
                "train/reward/min": float(rewards.min()), "train/reward/max": float(rewards.max()),
                "train/response-length/mean": float(np.mean(comp_lens)) if comp_lens else 0.0,
                "train/adv/abs_mean": float(np.abs(adv).mean()),
                "train/loss": stat.get("loss", 0.0), "train/kl": stat.get("kl", 0.0),
                "train/ratio": stat.get("ratio", 1.0), "train/clipfrac": stat.get("clipfrac", 0.0),
                "train/grad_norm": stat.get("grad_norm", 0.0),
                "train/n_completions": len(samples), "time/step_s": time.time() - t0,
            }
            self._log(payload)
            print(f"[step {step}/{a.n_iterations}] reward={payload['train/reward/mean']:.4f} "
                  f"len={payload['train/response-length/mean']:.0f} kl={payload['train/kl']:.4f} "
                  f"loss={payload['train/loss']:.4f} clip={payload['train/clipfrac']:.3f} "
                  f"({payload['time/step_s']:.1f}s)")

            if a.eval_freq > 0 and step % a.eval_freq == 0:
                self.evaluate(step)

        # final model
        final = os.path.join(self.run_dir, "final")
        ray.get(self.learner.save_pretrained.remote(final))
        for f in os.listdir(self.run_dir):
            if f.endswith((".json", ".jinja", ".txt")) and os.path.isfile(os.path.join(self.run_dir, f)):
                shutil.copy(os.path.join(self.run_dir, f), final)
        print(f"[done] final model -> {final}")
        if self.use_wandb:
            try:
                wandb.finish()
            except Exception:
                pass
        self.cleanup()

    def cleanup(self):
        try:
            os.path.exists(self.sync_path) and os.remove(self.sync_path)
        except Exception:
            pass
        for e in self.engines:
            try:
                ray.kill(e)
            except Exception:
                pass
        for pg in self.pgs:
            try:
                remove_placement_group(pg)
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="WeiboAI/VibeThinker-3B")
    ap.add_argument("--train-dataset", default="RL/data/rl_train_1000")
    ap.add_argument("--eval-dataset", default="datasets/safety_val_small")
    ap.add_argument("--n-iterations", type=int, default=300)
    ap.add_argument("--eval-freq", type=int, default=50)
    # GRPO
    ap.add_argument("--prompts-per-step", type=int, default=8)
    ap.add_argument("--num-generations", type=int, default=8, help="G: samples per prompt (group size)")
    ap.add_argument("--mu", type=int, default=1, help="inner GRPO epochs per batch")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.04, help="KL-to-ref coefficient (<=0 disables the ref/KL)")
    ap.add_argument("--eps-clip", type=float, default=0.2)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--micro-bsz", type=int, default=2, help="completions per learner forward/backward")
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--train-temperature", type=float, default=0.7)
    ap.add_argument("--train-top-p", type=float, default=0.95)
    # engines / reward
    ap.add_argument("--n-engines", type=int, default=4)
    ap.add_argument("--gpu-mem", type=float, default=0.7)
    ap.add_argument("--judge-model", default="claude-opus-4-8")
    ap.add_argument("--judge-workers", type=int, default=24)
    ap.add_argument("--penalize-cot", action="store_true", default=True)
    ap.add_argument("--no-penalize-cot", dest="penalize_cot", action="store_false")
    ap.add_argument("--seed", type=int, default=42)
    # logging
    ap.add_argument("--logging", default="wandb", choices=["wandb", "none"])
    ap.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "es-safety-grpo"))
    args = ap.parse_args()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    GRPOTrainer(args).fit()


if __name__ == "__main__":
    main()
