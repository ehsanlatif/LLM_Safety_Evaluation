import argparse
from datetime import datetime
import gc
import os
import random
import shutil
import signal
import sys
import time
import numpy as np

import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from vllm import LLM, SamplingParams
from vllm.utils import get_ip, get_open_port
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
import wandb

import torch
from torch.utils.data import DataLoader
import json

from typing import Dict, Tuple, List, Any, Optional
from multiprocessing import Pool, TimeoutError
import functools

from es_reasoning.utils.reward_shaping import centered_ranks, softmax_rank_utilities, z_score
from es_reasoning.utils.overlap_batch_sampler import OverlapBatchSampler

# ES4Safety root on sys.path so `reward.safety_guard` imports from either the
# driver or a Ray worker regardless of CWD.
ES4SAFETY_ROOT = os.environ.get("ES4SAFETY_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ES4SAFETY_ROOT not in sys.path:
    sys.path.insert(0, ES4SAFETY_ROOT)

EXPERIMENTS_DIR = os.environ.get("ES4SAFETY_EXP", os.path.join(ES4SAFETY_ROOT, "experiments"))

from reward.safety_guard import split_think as _split_think


@ray.remote(num_gpus=1)
class RayLlamaGuard:
    """LlamaGuard-2-8B safety scorer pinned to one Ray-assigned GPU (see reward/safety_guard.py)."""
    def __init__(self, model_id, penalize_cot=True):
        from reward.safety_guard import LlamaGuardScorer
        self.scorer = LlamaGuardScorer(model_id=model_id, device="cuda")
        self.penalize_cot = penalize_cot

    def score_pairs(self, pairs, kinds=None):  # kinds ignored: LlamaGuard has no helpfulness signal
        return self.scorer.score_pairs(pairs, penalize_cot=self.penalize_cot)

    def ping(self):
        return True


@ray.remote(num_gpus=0)
class RayOpusGuard:
    """Claude-Opus granular safety/helpfulness reward (see reward/safety_opus.py).

    API-bound, so it needs no GPU (num_gpus=0) — freeing every GPU for vLLM
    engines. `kinds` selects the harmful-fulfillment vs benign-helpfulness rubric
    per rollout so ES gets an anti-over-refusal signal on the benign train rows.
    """
    def __init__(self, model_id, penalize_cot=True, workers=24):
        from reward.safety_opus import OpusSafetyScorer
        self.scorer = OpusSafetyScorer(model_id=model_id, workers=workers)
        self.penalize_cot = penalize_cot

    def score_pairs(self, pairs, kinds=None):
        return self.scorer.score_pairs(pairs, kinds=kinds, penalize_cot=self.penalize_cot)

    def ping(self):
        return True


class ESNcclLLM(LLM):
    def __init__(self, *args, **kwargs):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        super().__init__(*args, **kwargs)

from torch.utils.data import Dataset, DataLoader

class LengthDataset(Dataset):
    def __init__(self):
        self.rows = [
            {"problem": "If all birds can fly and penguins are birds, can penguins fly?", "answer": "yes"},
            {"problem": "If all animals can walk and buffalows are animals, can goats walk?", "answer": "yes"},
            {"problem": "If all cats are mammals and all mammals are animals, are all cats animals?", "answer": "yes"},
            {"problem": "What is 1+1?", "answer": "2"},
            {"problem": "What is 2+2?", "answer": "4"},
            {"problem": "What is 3+3?", "answer": "6"},
            {"problem": "What is 4+4?", "answer": "8"},
            {"problem": "What is 5+5?", "answer": "10"},
            {"problem": "What is 6+6?", "answer": "12"},
            {"problem": "What is 7+7?", "answer": "14"},
            {"problem": "What is 8+8?", "answer": "16"},
            {"problem": "What is 9+9?", "answer": "18"},
            {"problem": "What is 10+10?", "answer": "20"},
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]
    
    def collate_fn(batch):
        prompts = [x["problem"] for x in batch]
        answers = [x["answer"] for x in batch]
        return prompts, answers

class Trainer():
    def __init__(
        self,
        model_name,
        checkpoint,
        sigma,
        alpha,
        mu,
        population_size,
        reward_shaping,
        mirror_sampling,
        num_iterations,
        max_tokens,
        batch_size,
        mini_batch_size,
        reward_function,
        template_function,
        train_dataset_path,
        eval_dataset_path,
        eval_freq,
        n_vllm_engines,
        logging,
        debug,
        per_member_random_batch,
        n_samples: int = 1,
        rollout_reduce: str = "mean",  # "mean" or "max"
        train_temperature: float = 0.0,
        train_top_p: float = 1.0,
        eval_temperature: float = 0.0,
        eval_top_p: float = 1.0,
        reward_mode: str = "math",              # "math" | "safety"
        reward_model_id: str = "meta-llama/Meta-Llama-Guard-2-8B",
        penalize_cot: bool = True,
        teacher_shaping: float = 0.0,
        teacher_file: str = "",
        judge_backend: str = "llamaguard",      # "llamaguard" | "opus" (safety mode)
        judge_model: str = "claude-opus-4-8",
        judge_workers: int = 24,
        awd_type: str = "none",                 # "none" | "l1" | "l2"  (Anchored Weight Decay)
        awd_lambda: float = 0.0,
    ):

        # GPU init
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

        # Ray init
        os.environ.pop("RAY_ADDRESS", None)
        os.environ.pop("RAY_HEAD_IP", None)
        os.environ.pop("RAY_GCS_SERVER_ADDRESS", None)

        ray.init(address="local", include_dashboard=False, ignore_reinit_error=True)

        signal.signal(signal.SIGINT, lambda sig, frame: self._handle_exit(sig, frame))
        signal.signal(signal.SIGTERM, lambda sig, frame: self._handle_exit(sig, frame))

        # Experiment config
        self.model_name = model_name
        self.checkpoint = checkpoint
        self.num_iterations = num_iterations
        self.population_size = population_size
        self.reward_shaping = reward_shaping
        self.mirror_sampling = mirror_sampling
        self.sigma = sigma
        self.alpha = alpha
        self.mu = mu
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.n_vllm_engines = n_vllm_engines
        self.eval_freq = eval_freq
        self.logging = logging
        self.debug = debug    


        # per-member random batches flag
        self.per_member_random_batch = per_member_random_batch

        # NEW: multi-sample rollouts
        self.n_samples = int(n_samples)
        if self.n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        if rollout_reduce not in ("mean", "max"):
            raise ValueError("rollout_reduce must be 'mean' or 'max'")
        self.rollout_reduce = rollout_reduce

        self.train_temperature = float(train_temperature)
        self.train_top_p = float(train_top_p)
        self.eval_temperature = float(eval_temperature)
        self.eval_top_p = float(eval_top_p)

        self.best_avg = -np.inf
        self.best_math_avg = -np.inf
        self.pop_best_avg = -np.inf
        self.best_population_eval_mean = -np.inf

        _safe_model = self.model_name.replace("/", "_")  # avoid nested dirs from HF ids like "WeiboAI/VibeThinker-3B"
        _prefix = "es-safety" if reward_mode == "safety" else "es-finetuned-math"
        _awd_tag = f"awd{awd_type}{awd_lambda:g}" if (str(awd_type) in ("l1", "l2") and float(awd_lambda) > 0) else "noawd"
        _judge_tag = judge_backend if reward_mode == "safety" else "math"
        self.exp_name = (
            f"{_prefix}-sigma{sigma}-alpha{alpha}-pop{self.population_size}"
            f"-bs{batch_size}-tokens{max_tokens}-model{_safe_model}"
            f"-ns{self.n_samples}-{self.rollout_reduce}-{_judge_tag}-{_awd_tag}"
        )

        model_saves_dir = EXPERIMENTS_DIR
        os.makedirs(model_saves_dir, exist_ok=True)


        self.task = functools.partial(reward_function, fast=False) if reward_function is not None else None
        # Process pool is used to enable the timeout mechanism for answer grading in our distributed training setup.
        self.mp_pool = Pool(2)
        self.template = template_function

        # ---- reward mode ----
        self.reward_mode = reward_mode
        self.reward_model_id = reward_model_id
        self.penalize_cot = penalize_cot
        self.judge_backend = judge_backend
        self.judge_model = judge_model
        self.judge_workers = int(judge_workers)
        self.guard = None  # reward actor (LlamaGuard/Opus), created after engines launch (safety mode)

        # ---- Anchored Weight Decay (AWD, kschweig/es-awd): decay ES weights toward
        # the ORIGINAL (iteration-0) model each step to curb prior-task forgetting. ----
        self.awd_type = str(awd_type)
        self.awd_lambda = float(awd_lambda)
        self.awd_enabled = self.awd_type in ("l1", "l2") and self.awd_lambda > 0.0

        # Optional teacher (Qwen) refusal-style shaping: a small bonus for producing a
        # concise refusal like the safety-aligned teacher. Densifies the 0/1 safety signal.
        self.teacher_shaping = float(teacher_shaping)
        self.refusal_markers = [
            "i can't", "i cannot", "i can not", "i won't", "i will not", "i'm sorry",
            "i am sorry", "i'm not able", "i am not able", "i'm unable", "unable to help",
            "cannot assist", "can't assist", "cannot help with", "can't help with",
            "not able to help", "i must decline", "i won't be able", "against my", "not appropriate",
        ]
        if self.teacher_shaping > 0 and teacher_file and os.path.isfile(teacher_file):
            n_ref = 0
            for line in open(teacher_file, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    txt = json.loads(line).get("response", "").lower()
                except json.JSONDecodeError:
                    continue
                if any(m in txt for m in self.refusal_markers):
                    n_ref += 1
            print(f"[teacher-shaping] weight={self.teacher_shaping}; "
                  f"{n_ref} teacher responses matched refusal markers (validates the marker set).")

        # Train dataset loader: real HuggingFace datasets (save_to_disk format,
        # each row {"problem": ..., "answer": ...}). Falls back to the toy
        # LengthDataset only if no path is given (debugging).
        if train_dataset_path:
            train_dataset = load_from_disk(train_dataset_path)
            self.train_hf_dataset = train_dataset
            self.train_dataset = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=self.collate_fn,
            )
        else:
            train_dataset = LengthDataset()
            self.train_hf_dataset = train_dataset
            self.train_dataset = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True,
                collate_fn=LengthDataset.collate_fn,
            )

        # Eval: a DatasetDict of named held-out splits (e.g. sorry_bench, ailuminate).
        self.eval_dataset = {}
        if eval_dataset_path:
            for task_name, dataset in load_from_disk(eval_dataset_path).items():
                self.eval_dataset[task_name] = DataLoader(
                    dataset, batch_size=9999999, shuffle=False, collate_fn=self.collate_fn,
                )
        else:
            eval_dataset = LengthDataset()
            self.eval_dataset["toy_eval"] = DataLoader(
                eval_dataset, batch_size=len(eval_dataset), shuffle=False,
                collate_fn=LengthDataset.collate_fn,
            )

        save_dir = EXPERIMENTS_DIR
        self.logging_dir = f"{save_dir}/{self.exp_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        os.makedirs(f"{self.logging_dir}/checkpoints")
        os.makedirs(f"{self.logging_dir}/eval-output")
        os.makedirs(f"{self.logging_dir}/train-output")  # NEW: per-iteration train outputs

        if self.logging == "wandb":
            wandb_group = self.exp_name
            try:
                wandb.login()
            except Exception as e:
                print(f"[WARN] wandb.login() failed: {e}. Proceeding; W&B may run offline/disabled.")

            self.wandb_run = wandb.init(
                project="es-finetuning",
                group=wandb_group,
                name=f"{self.exp_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                dir=self.logging_dir,
                mode=os.environ.get("WANDB_MODE", "online"),
                settings=wandb.Settings(start_method="thread"),
            )

            wandb.define_metric("global_step")
            wandb.define_metric("train/*", step_metric="global_step")
            wandb.define_metric("eval/*", step_metric="global_step")

        base_model = AutoModelForCausalLM.from_pretrained(model_name).to("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        base_model_path = f"{model_saves_dir}/base_model"
        if os.path.exists(base_model_path):
            shutil.rmtree(base_model_path)
        os.makedirs(base_model_path, exist_ok=True)
        self.tokenizer.save_pretrained(base_model_path)
        base_model.save_pretrained(base_model_path)
        del base_model

        self.best_member = -np.inf

        torch.cuda.empty_cache()
        gc.collect()

        # Start persistent engines
        self.engines, self.pgs = self.launch_engines(num_engines=self.n_vllm_engines, model_name=self.model_name)

        # Safety reward actor. LlamaGuard needs a dedicated GPU (keep n_vllm_engines
        # <= total_gpus - 1); the Opus judge is API-bound and takes no GPU.
        if self.reward_mode == "safety":
            if self.judge_backend == "opus":
                print(f"[reward] launching Claude-Opus judge actor ({self.judge_model}, "
                      f"{self.judge_workers} workers, granular reward) ...")
                self.guard = RayOpusGuard.remote(self.judge_model, self.penalize_cot, self.judge_workers)
            else:
                print(f"[reward] launching LlamaGuard actor ({self.reward_model_id}) on a dedicated GPU ...")
                self.guard = RayLlamaGuard.remote(self.reward_model_id, self.penalize_cot)
            ray.get(self.guard.ping.remote())
            print(f"[reward] {self.judge_backend} reward actor ready.")

        master_address = get_ip()
        master_port = get_open_port()
        ray.get(
            [
                self.engines[i].collective_rpc.remote(
                    "init_inter_engine_group", args=(master_address, master_port, i, self.n_vllm_engines)
                )
                for i in range(self.n_vllm_engines)
            ]
        )

        if self.checkpoint != "":
            print("Loading checkpoint weights")
            ray.get(
                [
                    self.engines[i].collective_rpc.remote(
                        "load_weights_from_disk", args=(self.checkpoint,)
                    )
                    for i in range(self.n_vllm_engines)
                ]
            )
            print("Completed loading checkpoint weights")

        # AWD anchor = the model weights BEFORE any ES update (post-checkpoint if any).
        # Captured once here so update_weights_from_seeds can decay toward it every step.
        if self.awd_enabled:
            print(f"[awd] capturing anchor weights ({self.awd_type}, lambda={self.awd_lambda}) ...")
            ray.get([e.collective_rpc.remote("save_awd_anchor") for e in self.engines])
            print("[awd] anchor captured on all engines.")

        self.eval_cache = {}  # name -> (prompts, targets)
        for name, loader in self.eval_dataset.items():
            # Your eval loader uses batch_size=9999999, so this should be one batch
            for input_text, target_text in loader:
                prompts = [self.template(i) for i in input_text]
                targets = list(target_text)
                self.eval_cache[name] = (prompts, targets)
                break

    def collate_fn(self, batch):
        prompts = [item["problem"] for item in batch]
        answer = [item["answer"] for item in batch]
        return prompts, answer

    def _log_metrics(self, payload: dict):
        """Mirror every wandb payload to {logging_dir}/metrics.jsonl so reward/eval
        curves can be plotted locally even when W&B runs offline (see scripts/plot_training.py)."""
        try:
            rec = {k: v for k, v in payload.items() if isinstance(v, (int, float, str, bool))}
            with open(f"{self.logging_dir}/metrics.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
        except Exception as e:
            print(f"[WARN] _log_metrics failed: {e}")

    def _launch_eval_handles(self, llm, iteration: int, eval_type: str):
        sampling_params = SamplingParams(
            n=1,
            seed=40,
            temperature=self.eval_temperature,
            top_p=self.eval_top_p,
            max_tokens=self.max_tokens,
        )
        handles = []
        for name, (prompts, targets) in self.eval_cache.items():
            h, start_ts = self.evaluate_handle(llm, prompts, sampling_params)
            handles.append((name, h, start_ts, targets))
        
        return handles

    # ==========================
    # per-member random subset sampling helpers
    # ==========================
    def _sample_indices(self, n: int, k: int, rng: np.random.Generator) -> np.ndarray:
        """Sample k indices from [0, n). Uses replacement only if needed."""
        if n <= 0 or k <= 0:
            return np.array([], dtype=np.int64)
        replace = n < k
        return rng.choice(n, size=k, replace=replace)

    def _build_member_full_batches(
        self,
        seeds: List[Tuple[int, float]],
        iteration: int,
    ) -> Dict[Tuple[int, float], Tuple[List[str], List[str]]]:
        """
        For each population member (seed, sign), sample a random subset of the FULL dataset
        of size self.batch_size.

        IMPORTANT CHANGE:
        - Reuse the same sampled batches for `self.mu` consecutive iterations.
        - Do this by using `batch_cycle = iteration // self.mu` and the member slot index,
        NOT the ES seed (which changes every generation).
        """
        if self.train_hf_dataset is None:
            raise RuntimeError("train_hf_dataset is None; cannot sample per-member batches.")

        n = len(self.train_hf_dataset)
        member_batches: Dict[Tuple[int, float], Tuple[List[str], List[str]]] = {}

        mu = max(1, int(self.mu))
        batch_cycle = int(iteration) // mu

        for member_idx, (seed, sign) in enumerate(seeds):
            mix = (
                int(member_idx)
                ^ (int(batch_cycle) << 16)
                ^ (0x9E3779B9 if float(sign) > 0 else 0x7F4A7C15)
            )
            rng = np.random.default_rng(mix & 0xFFFFFFFFFFFFFFFF)

            idx = self._sample_indices(n, self.batch_size, rng)

            problems = [self.train_hf_dataset[int(i)]["problem"] for i in idx]
            answers  = [self.train_hf_dataset[int(i)]["answer"]  for i in idx]

            problems = [self.template(p) for p in problems]

            member_batches[(seed, float(sign))] = (problems, answers)

        return member_batches

    def _spearman_corr(self, x: np.ndarray, y: np.ndarray) -> float:
        """Spearman correlation implemented as Pearson correlation of ranks (no scipy dependency)."""
        if x.size != y.size or x.size < 2:
            return 0.0
        xr = x.argsort().argsort().astype(np.float32)
        yr = y.argsort().argsort().astype(np.float32)
        xr -= xr.mean()
        yr -= yr.mean()
        denom = (np.linalg.norm(xr) * np.linalg.norm(yr)) + 1e-8
        return float(np.dot(xr, yr) / denom)

    def _batch_split_snr_diags(self, reward_matrix: np.ndarray) -> Dict[str, float]:
        """
        reward_matrix: shape [num_candidates, batch_size] (per-example rewards per candidate)
        Returns diagnostics for evaluation noise / signal strength.
        """
        if reward_matrix.ndim != 2:
            return {}
        n, m = reward_matrix.shape
        if n < 4 or m < 8:
            return {}

        h = m // 2
        score_a = reward_matrix[:, :h].mean(axis=1)
        score_b = reward_matrix[:, h:].mean(axis=1)

        return {
            "train/fitness/spearman_split": self._spearman_corr(score_a, score_b),
            "train/fitness/mean_abs_split_gap": float(np.mean(np.abs(score_a - score_b))),
            "train/fitness/std_score_a": float(np.std(score_a)),
            "train/fitness/std_score_b": float(np.std(score_b)),
        }

    def cleanup(self):
        """Gracefully terminate all Ray actors and placement groups."""
        for llm in self.engines:
            try:
                ray.kill(llm)
            except Exception:
                pass
        for pg in self.pgs:
            try:
                remove_placement_group(pg)
            except Exception:
                pass
        print("[INFO] Cleanup complete.")

    def _handle_exit(self, sig, frame):
        """Signal handler wrapper."""
        print(f"[INFO] Received signal {sig}, cleaning up...")
        self.cleanup()
        sys.exit(0)

    def launch_engines(self, num_engines=4, model_name="Qwen/Qwen2.5-Math-1.5B", precision="bfloat16"):
        pgs = [placement_group([{"GPU": 1, "CPU": 0}], lifetime="detached") for _ in range(num_engines)]
        ray.get([pg.ready() for pg in pgs])

        strategies = [
            PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=0,
            )
            for pg in pgs
        ]

        engines = [
            ray.remote(num_cpus=0, num_gpus=0, scheduling_strategy=strategy)(ESNcclLLM).remote(
                model=model_name,
                tensor_parallel_size=1,
                distributed_executor_backend="ray",
                worker_extension_cls="es_reasoning.utils.worker_extension.WorkerExtension",
                dtype=precision,
                enable_prefix_caching=False,
                enforce_eager=False,
                gpu_memory_utilization=0.7,
            )
            for strategy in strategies
        ]
        return engines, pgs

    # ==========================
    # UPDATED: supports SamplingParams(n>1)
    # - reduces per-prompt rollouts -> one scalar reward per prompt
    # - keeps optional per-rollout details for eval logging (and now train logging too)
    # ==========================
    @staticmethod
    def _extract_user(templated_prompt: str) -> str:
        """Recover the raw user request from a chat-templated prompt for LlamaGuard."""
        marker = "<|im_start|>user\n"
        if marker in templated_prompt:
            return templated_prompt.split(marker, 1)[1].split("<|im_end|>", 1)[0].strip()
        return templated_prompt

    def _postprocess_safety(self, generated_text, target_text, eval=False):
        """Safety reward path: batch-score every rollout with the LlamaGuard actor
        (bypasses the 1s mp_pool used by the math path). Same return schema."""
        reduce_mode = self.rollout_reduce
        pairs, kinds, meta = [], [], []
        for pi, (gen, target) in enumerate(zip(generated_text, target_text)):
            user = self._extract_user(gen.prompt)
            # target "answer" field: "benign" for benign train rows, "safe" for harmful.
            kind = "benign" if str(target).strip().lower() == "benign" else "harmful"
            for ridx, out in enumerate(gen.outputs):
                decoded = self.tokenizer.decode(out.token_ids, skip_special_tokens=True)
                pairs.append((user, out.text))
                kinds.append(kind)
                meta.append((pi, ridx, decoded, len(out.token_ids), out.text, gen.prompt, target))

        scored = ray.get(self.guard.score_pairs.remote(pairs, kinds)) if pairs else []

        n_prompts = len(generated_text)
        per_prompt_rewards: List[List[float]] = [[] for _ in range(n_prompts)]
        per_prompt_lens: List[List[int]] = [[] for _ in range(n_prompts)]
        save = []
        for (info, r), (pi, ridx, decoded, glen, rtext, gprompt, target) in zip(scored, meta):
            r = float(r)
            if self.teacher_shaping > 0 and r >= 1.0:
                # teacher (Qwen)-style shaping: small bonus for a concise refusal answer
                _cot, ans, has_ans, _isr = _split_think(rtext or "")
                low = (ans or "").lower()
                if has_ans and len(ans) < 600 and any(m in low for m in self.refusal_markers):
                    r += self.teacher_shaping
            per_prompt_rewards[pi].append(r)
            per_prompt_lens[pi].append(int(glen))
            if eval:
                save.append({
                    "prompt": gprompt, "answer": target, "rollout_idx": int(ridx),
                    "decoded_response": decoded, "model_output": rtext,
                    "reward": r, "format": info, "response_length": int(glen),
                })

        rewards_per_prompt, gen_lens_per_prompt = [], []
        for pi in range(n_prompts):
            rr, ll = per_prompt_rewards[pi], per_prompt_lens[pi]
            if not rr:
                rewards_per_prompt.append(0.0); gen_lens_per_prompt.append(0.0); continue
            rewards_per_prompt.append(float(np.max(rr)) if reduce_mode == "max" else float(np.mean(rr)))
            gen_lens_per_prompt.append(float(np.mean(ll)))

        return {
            "rewards": rewards_per_prompt,
            "avg_reward": float(np.mean(rewards_per_prompt)) if rewards_per_prompt else 0.0,
            "gen_lengths": gen_lens_per_prompt,
            "avg_gen_lengths": float(np.mean(gen_lens_per_prompt)) if gen_lens_per_prompt else 0.0,
            "results": save,
            "raw_rewards_per_prompt": per_prompt_rewards,
            "raw_lens_per_prompt": per_prompt_lens,
            "rollout_reduce": reduce_mode,
            "n_samples": int(self.n_samples),
        }

    def _postprocess_outputs(self, generated_text, target_text, eval=False):
        if self.reward_mode == "safety":
            return self._postprocess_safety(generated_text, target_text, eval=eval)

        rewards_per_prompt: List[float] = []
        gen_lens_per_prompt: List[float] = []
        save = []

        raw_rewards_per_prompt: List[List[float]] = []
        raw_lens_per_prompt: List[List[int]] = []

        reduce_mode = self.rollout_reduce

        for gen, target in zip(generated_text, target_text):
            rollout_rewards: List[float] = []
            rollout_lens: List[int] = []

            

            for ridx in range(len(gen.outputs)):
                out = gen.outputs[ridx]
                response_text = out.text
                token_ids = out.token_ids
                gen_len = len(token_ids)
                decoded_response = self.tokenizer.decode(token_ids, skip_special_tokens=True)

                #fmt, r = self.task(response_text, target)
                res = self.mp_pool.apply_async(self.task, (response_text, target))
                try:
                    fmt, r = res.get(timeout=1)
                    rollout_rewards.append(float(r))
                    #infos.append(info)
                except TimeoutError:
                    rollout_rewards.append(0.0)
                    #infos.append({"formatted": False})
                
                #rollout_rewards.append(float(r))
                rollout_lens.append(int(gen_len))

                if eval:
                    save.append(
                        {
                            "prompt": gen.prompt,
                            "answer": target,
                            "rollout_idx": int(ridx),
                            "decoded_response": decoded_response,
                            "model_output": response_text,
                            "reward": float(r),
                            "format": fmt,
                            "response_length": int(gen_len),
                        }
                    )

            raw_rewards_per_prompt.append(rollout_rewards)
            raw_lens_per_prompt.append(rollout_lens)

            if len(rollout_rewards) == 0:
                rewards_per_prompt.append(0.0)
                gen_lens_per_prompt.append(0.0)
                continue

            if reduce_mode == "max":
                r_red = float(np.max(rollout_rewards))
            else:
                r_red = float(np.mean(rollout_rewards))

            l_red = float(np.mean(rollout_lens))

            rewards_per_prompt.append(r_red)
            gen_lens_per_prompt.append(l_red)

        return {
            "rewards": rewards_per_prompt,
            "avg_reward": float(np.mean(rewards_per_prompt)) if rewards_per_prompt else 0.0,
            "gen_lengths": gen_lens_per_prompt,
            "avg_gen_lengths": float(np.mean(gen_lens_per_prompt)) if gen_lens_per_prompt else 0.0,
            "results": save,
            "raw_rewards_per_prompt": raw_rewards_per_prompt,
            "raw_lens_per_prompt": raw_lens_per_prompt,
            "rollout_reduce": reduce_mode,
            "n_samples": int(self.n_samples),
        }

    def evaluate_handle(self, llm, input_text, sampling_params):
        start = time.time()
        handle = llm.generate.remote(input_text, sampling_params, use_tqdm=False)
        return handle, start

    def _write_train_population_outputs(
        self,
        iteration: int,
        seeds: List[Tuple[int, float]],
        per_member_records: Dict[Tuple[int, float], List[Dict[str, Any]]],
    ):
        """
        Writes a JSON file with the SAME SHAPE as eval output files:
          save_results = [ member0_results, member1_results, ... ]
        where each member_results is a list[dict] (per rollout per prompt).
        """
        if iteration % 10 == 0:
            save_results = []
            for seed, sign in seeds:
                save_results.append(per_member_records.get((seed, float(sign)), []))

            fn = f"{self.logging_dir}/train-output/model_train_iteration{iteration}.json"
            print(f"[train] saving population outputs at {fn}")
            with open(fn, "w") as f:
                json.dump(save_results, f, indent=4)

    def train_step(self, iteration, seeds, input_text, target_text):
        sampling_params = SamplingParams(
            n=self.n_samples,
            seed=40,
            temperature=self.train_temperature,
            top_p=self.train_top_p,
            max_tokens=self.max_tokens,
        )

        do_eval_this_iter = (iteration % 50) == 0 and iteration > 0

        agg = {}
        for seed, sign in seeds:
            agg[(seed, float(sign))] = {"sum_reward": 0.0, "sum_length": 0.0, "count": 0}

        all_results_this_gen = []
        diag_accum = {}
        diag_counts = {}

        # NEW: collect eval-style per-member outputs across minibatches
        train_member_outputs: Dict[Tuple[int, float], List[Dict[str, Any]]] = {
            (seed, float(sign)): [] for seed, sign in seeds
        }

        if self.per_member_random_batch:
            member_batches = self._build_member_full_batches(seeds=seeds, iteration=iteration)
            total_n = self.batch_size

            for mb_idx, start in enumerate(range(0, total_n, self.mini_batch_size)):
                end = min(total_n, start + self.mini_batch_size)

                per_member_inputs = {k: v[0][start:end] for k, v in member_batches.items()}
                per_member_targets = {k: v[1][start:end] for k, v in member_batches.items()}

                seeds_perf_batch, results_this_gen, diag = self.evaluate_population_on_batch(
                    seeds=seeds,
                    input_batch=None,
                    target_batch=None,
                    sampling_params=sampling_params,
                    per_member_inputs=per_member_inputs,
                    per_member_targets=per_member_targets,
                    iteration=iteration,
                    save_train_outputs=True,
                    do_eval=(do_eval_this_iter and mb_idx == 0)
                )
                all_results_this_gen.extend(results_this_gen)

                # NEW: accumulate train outputs (same record format as eval)
                for key, metrics in seeds_perf_batch.items():
                    if metrics.get("results"):
                        train_member_outputs[key].extend(metrics["results"])

                mb_size = end - start
                for key, metrics in seeds_perf_batch.items():
                    agg[key]["sum_reward"] += metrics["avg_reward"] * mb_size
                    agg[key]["sum_length"] += metrics["avg_gen_lengths"] * mb_size
                    agg[key]["count"] += mb_size

                for k, v in (diag or {}).items():
                    diag_accum[k] = diag_accum.get(k, 0.0) + float(v)
                    diag_counts[k] = diag_counts.get(k, 0) + 1
        else:
            for mb_idx, (input_batch, target_batch) in enumerate(self._iter_minibatches(input_text, target_text, self.mini_batch_size)):
                batch_size = len(input_batch)
                if batch_size == 0:
                    continue

                seeds_perf_batch, results_this_gen, diag = self.evaluate_population_on_batch(
                    seeds, 
                    input_batch, 
                    target_batch, 
                    sampling_params, 
                    iteration=iteration, 
                    save_train_outputs=True,
                    do_eval=(do_eval_this_iter and mb_idx == 0)
                )

                all_results_this_gen.extend(results_this_gen)

                # NEW: accumulate train outputs (same record format as eval)
                for key, metrics in seeds_perf_batch.items():
                    if metrics.get("results"):
                        train_member_outputs[key].extend(metrics["results"])

                for key, metrics in seeds_perf_batch.items():
                    agg[key]["sum_reward"] += metrics["avg_reward"] * batch_size
                    agg[key]["sum_length"] += metrics["avg_gen_lengths"] * batch_size
                    agg[key]["count"] += batch_size

                for k, v in (diag or {}).items():
                    diag_accum[k] = diag_accum.get(k, 0.0) + float(v)
                    diag_counts[k] = diag_counts.get(k, 0) + 1

        # NEW: write outputs ONCE per iteration (all population members)
        self._write_train_population_outputs(iteration=iteration, seeds=seeds, per_member_records=train_member_outputs)

        seeds_perf = {}
        for key, stats in agg.items():
            c = max(1, stats["count"])
            seeds_perf[key] = {"avg_reward": stats["sum_reward"] / c, "avg_gen_lengths": stats["sum_length"] / c}

        diag_avg = {k: (total / max(1, diag_counts.get(k, 1))) for k, total in diag_accum.items()}

        all_avg_rewards = [v["avg_reward"] for v in seeds_perf.values()]
        all_avg_length = [v["avg_gen_lengths"] for v in seeds_perf.values()]

        mean_reward = float(np.mean(all_avg_rewards)) if all_avg_rewards else 0.0
        std_reward = float(np.std(all_avg_rewards)) if all_avg_rewards else 0.0
        min_reward = float(np.min(all_avg_rewards)) if all_avg_rewards else 0.0
        max_reward = float(np.max(all_avg_rewards)) if all_avg_rewards else 0.0

        mean_length = float(np.mean(all_avg_length)) if all_avg_length else 0.0
        std_length = float(np.std(all_avg_length)) if all_avg_length else 0.0
        min_length = float(np.min(all_avg_length)) if all_avg_length else 0.0
        max_length = float(np.max(all_avg_length)) if all_avg_length else 0.0

        print(f"Mean reward: {mean_reward}, std: {std_reward}, min: {min_reward}, max: {max_reward}")

        payload = {
            "global_step": iteration,
            "train/response-length/mean": mean_length,
            "train/response-length/min": min_length,
            "train/response-length/max": max_length,
            "train/response-length/std": std_length,
            "train/reward/mean": mean_reward,
            "train/reward/min": min_reward,
            "train/reward/max": max_reward,
            "train/reward/std": std_reward,
            "train/es/sigma": float(self.sigma),
            "train/es/alpha": float(self.alpha),
            "train/es/population": int(self.population_size),
            "train/sampling/n": int(self.n_samples),
            "train/sampling/reduce": self.rollout_reduce,
            "train/sampling/temperature": float(self.train_temperature),
            "train/sampling/top_p": float(self.train_top_p),
        }
        payload.update(diag_avg)
        self._log_metrics(payload)
        if self.logging == "wandb":
            wandb.log(payload, commit=True)

        if self.reward_shaping == "centered-ranks":
            seeds_perf = centered_ranks(seeds_perf)
        elif self.reward_shaping == "softmax-centered-ranks":
            seeds_perf = softmax_rank_utilities(seeds_perf)
        elif self.reward_shaping == "z-scores":
            seeds_perf = z_score(seeds_perf, std_reward=std_reward, mean_reward=mean_reward)

        coeffs = np.array([float(seeds_perf[(seed, float(sign))]["norm_reward"]) for seed, sign in seeds],dtype=np.float32,)
                
        coeffs = coeffs.tolist()

        
        ray.get(
            self.engines[0].collective_rpc.remote(
                "update_weights_from_seeds",
                args=(
                    seeds,
                    coeffs,
                    self.alpha,
                    self.population_size,
                    self.awd_type,
                    self.awd_lambda,
                ),
            )
        )

        ray.get([e.collective_rpc.remote("broadcast_all_weights", args=(0,)) for e in self.engines])


    def _iter_minibatches(self, input_text, target_text, mini_batch_size: int):
        n = len(input_text)
        for start in range(0, n, mini_batch_size):
            end = start + mini_batch_size
            yield input_text[start:end], target_text[start:end]

    def evaluate_population_on_batch(
        self,
        seeds,
        input_batch,
        target_batch,
        sampling_params,
        per_member_inputs: Optional[Dict[Tuple[int, float], List[str]]] = None,
        per_member_targets: Optional[Dict[Tuple[int, float], List[str]]] = None,
        iteration=0,
        save_train_outputs: bool = False,  
        do_eval: bool = False
    ):
        seeds_perf_batch = {}
        results_this_gen = []
        member_reward_vecs = []

        seed_iter = iter(seeds)
        train_inflight = {}   # handle -> meta
        eval_inflight = {}    # handle -> meta

        # Track eval completion per (engine_idx, seed, sign)
        pending_eval = {}  # key -> {"remaining": int, "scores": {dataset: avg_reward}, "llm": llm, ...}

        def schedule_next_train_on_engine(llm, engine_idx):
            try:
                next_seed, next_sign = next(seed_iter)
            except StopIteration:
                return
            ray.get(llm.collective_rpc.remote("perturb_self_weights", args=(next_seed, self.sigma, float(next_sign))))

            if per_member_inputs is not None:
                next_input = per_member_inputs[(next_seed, float(next_sign))]
            else:
                next_input = input_batch

            h, start_ts = self.evaluate_handle(llm, next_input, sampling_params)
            train_inflight[h] = {
                "engine": llm,
                "engine_idx": engine_idx,
                "seed": next_seed,
                "sign": float(next_sign),
                "start_ts": start_ts,
            }

        # initial fill
        for eng_idx, llm in enumerate(self.engines):
            schedule_next_train_on_engine(llm, eng_idx)

        while train_inflight or eval_inflight:
            done, _ = ray.wait(list(train_inflight.keys()) + list(eval_inflight.keys()), num_returns=1)
            h = done[0]

            # ---- TRAIN DONE ----
            if h in train_inflight:
                meta = train_inflight.pop(h)
                llm = meta["engine"]
                outputs = ray.get(h)

                if per_member_targets is not None:
                    this_target = per_member_targets[(meta["seed"], float(meta["sign"]))]
                else:
                    this_target = target_batch

                metrics = self._postprocess_outputs(outputs, this_target, eval=bool(save_train_outputs))

                # annotate train records (your existing code)
                if save_train_outputs and metrics.get("results"):
                    try:
                        member_idx = seeds.index((meta["seed"], float(meta["sign"])))
                    except Exception:
                        member_idx = None
                    for rec in metrics["results"]:
                        rec["seed"] = int(meta["seed"])
                        rec["sign"] = float(meta["sign"])
                        rec["iteration"] = int(iteration)
                        if member_idx is not None:
                            rec["member_idx"] = int(member_idx)

                r_vec = np.asarray(metrics["rewards"], dtype=np.float32)
                batch_len = len(this_target) if this_target is not None else 0
                if r_vec.shape[0] == batch_len and batch_len > 0:
                    member_reward_vecs.append(r_vec)

                elapsed = time.time() - meta["start_ts"]
                seeds_perf_batch[(meta["seed"], meta["sign"])] = metrics
                results_this_gen.append(
                    {"seed": meta["seed"], "avg_reward": metrics["avg_reward"], "time": elapsed, "sign": meta["sign"]}
                )

                # If it's an eval iteration, launch eval on THIS perturbed llm BEFORE restoring.
                #do_eval = (iteration % self.eval_freq) == 0 and iteration > 0
                if do_eval:
                    key = (meta["engine_idx"], meta["seed"], float(meta["sign"]))
                    eval_type = f"population-eval"  # NOTE quote fix
                    handles = self._launch_eval_handles(llm, iteration=iteration, eval_type=eval_type)

                    pending_eval[key] = {
                        "remaining": len(handles),
                        "scores": {},
                        "llm": llm,
                        "engine_idx": meta["engine_idx"],
                        "seed": meta["seed"],
                        "sign": float(meta["sign"]),
                        "eval_type": eval_type,
                    }

                    for ds_name, eh, estart, ds_targets in handles:
                        eval_inflight[eh] = {
                            "key": key,
                            "dataset": ds_name,
                            "start_ts": estart,
                            "targets": ds_targets,
                        }
                    # IMPORTANT: do NOT restore weights yet, do NOT schedule next seed yet.
                    continue

                # Non-eval path: restore and keep engine busy
                ray.get(llm.collective_rpc.remote("restore_self_weights", args=(meta["seed"], meta["sign"], self.sigma)))
                schedule_next_train_on_engine(llm, meta["engine_idx"])
                continue

            # ---- EVAL DONE ----
            meta = eval_inflight.pop(h)
            outputs = ray.get(h)

            key = meta["key"]
            state = pending_eval[key]
            llm = state["llm"]

            metrics = self._postprocess_outputs(outputs, meta["targets"], eval=True)
            state["scores"][meta["dataset"]] = metrics["avg_reward"]
            state["remaining"] -= 1

            # Save per-dataset eval outputs (same as your eval_step)
            fn = f"{self.logging_dir}/eval-output/{state['eval_type']}-model_eval_task{meta['dataset']}_iteration{iteration}.json"
            json.dump([metrics["results"]], open(fn, "w"), indent=4)

            if state["remaining"] == 0:
                # log per-dataset + aggregated eval
                if self.logging == "wandb":
                    payload = {"global_step": iteration}

                    # per-dataset
                    for ds_name, score in state["scores"].items():
                        payload[f"{state['eval_type']}/{ds_name}/pass@1/mean"] = float(score)

                    # mean across datasets
                    mean_eval = float(np.mean(list(state["scores"].values()))) if state["scores"] else 0.0
                    payload[f"{state['eval_type']}/avgpass@1/mean"] = mean_eval

                    wandb.log(payload, commit=True)

                # ✅ check and save best-ever weights
                if mean_eval > self.best_population_eval_mean:
                    self.best_population_eval_mean = mean_eval

                    model_path = (
                        f"{self.logging_dir}/checkpoints/"
                        f"es-math-{state['eval_type']}-pop{self.population_size}-sigma{self.sigma}-alpha{self.alpha}-bs{self.batch_size}-iteration{iteration}"
                        f"-mean{mean_eval:.6f}"
                    )
                    os.makedirs(model_path, exist_ok=True)

                    ray.get(
                        llm.collective_rpc.remote(
                            "save_self_weights_to_disk",
                            args=(f"{model_path}/pytorch_model.pth",),
                        )
                    )
                    print(f"[eval] New best population-eval mean={mean_eval:.6f}. Saved weights to {model_path}")


                # NOW restore weights and schedule next seed
                ray.get(llm.collective_rpc.remote("restore_self_weights", args=(state["seed"], state["sign"], self.sigma)))
                schedule_next_train_on_engine(llm, state["engine_idx"])
                del pending_eval[key]

        diag = {}
        if len(member_reward_vecs) >= 4:
            S = np.stack(member_reward_vecs, axis=0)
            diag = self._batch_split_snr_diags(S)

        return seeds_perf_batch, results_this_gen, diag

    def eval_step(self, iteration, llm=None, eval_type="", save=True):
        to_log = {"eval-iteration": iteration}
        mean_eval_results = []
        mean_math = None

        for name, eval_loader in self.eval_dataset.items():
            batch_results = []
            save_results = []

            for input_text, target_text in eval_loader:
                seeds_perf = {}
                inflight = {}

                input_text = [self.template(i) for i in input_text]

                llm_model = llm
                if llm_model is None:
                    llm_model = self.engines[0]

                sampling_params = SamplingParams(
                    n=1,
                    seed=40,
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=self.max_tokens,
                )

                handle, start_ts = self.evaluate_handle(llm_model, input_text, sampling_params)
                inflight[handle] = {"engine": llm_model, "engine_idx": 0, "start_ts": start_ts}

                while inflight:
                    done, _ = ray.wait(list(inflight.keys()), num_returns=1)
                    h = done[0]
                    meta = inflight.pop(h)

                    outputs = ray.get(h)
                    metrics = self._postprocess_outputs(outputs, target_text, eval=True)
                    seeds_perf[meta["engine_idx"]] = metrics

                batch_results += [v["avg_reward"] for v in seeds_perf.values()]
                save_results += [v["results"] for v in seeds_perf.values()]

            dataset_results = float(np.mean(batch_results)) if batch_results else 0.0
            dataset_max = float(np.max(batch_results)) if batch_results else 0.0
            dataset_min = float(np.min(batch_results)) if batch_results else 0.0
            mean_eval_results.append(dataset_results)

            print(f"{name} -- eval pass@1: {dataset_results}, eval max: {dataset_max}, eval min: {dataset_min}")
            if name == "math":
                mean_math = dataset_results

            if llm is None:
                to_log.update({"global_step": iteration, f"eval/{name}/pass@1/mean": dataset_results})
                fn = f"{self.logging_dir}/eval-output/model_eval_task{name}_iteration{iteration}.json"
            elif "population-eval" in eval_type:
                random_idx = random.randint(1, 10**2)
                to_log.update({"global_step": iteration, f"eval/{eval_type}/{name}/pass@1/mean": dataset_results})
                fn = f"{self.logging_dir}/eval-output/{eval_type}-model-{random_idx}_eval_task{name}_iteration{iteration}.json"
            
            else:
                to_log.update({"global_step": iteration, f"eval/{eval_type}-{name}/pass@1/mean": dataset_results})
                fn = f"{self.logging_dir}/eval-output/{eval_type}-model_eval_task{name}_iteration{iteration}.json"

            print(f"saving model outputs at {fn}")
            json.dump(save_results, open(fn, "w"), indent=4)

        if llm is None:
            to_log.update({f"eval/avgpass@1/mean": float(np.mean(mean_eval_results)) if mean_eval_results else 0.0})
        else:
            to_log.update({f"eval/{eval_type}-avgpass@1/mean": float(np.mean(mean_eval_results)) if mean_eval_results else 0.0})
        to_log.update(
            {
                "eval/sampling/n": int(self.n_samples),
                "eval/sampling/reduce": self.rollout_reduce,
                "eval/sampling/temperature": float(self.eval_temperature),
                "eval/sampling/top_p": float(self.eval_top_p),
            }
        )

        # In safety mode the mean reward IS the safe fraction; expose explicit
        # safe/unsafe-rate metrics so the eval plots read as safety curves.
        if self.reward_mode == "safety":
            for k, v in list(to_log.items()):
                if k.endswith("pass@1/mean") and isinstance(v, (int, float)):
                    base = k[: -len("pass@1/mean")]
                    to_log[base + "safe_rate"] = float(v)
                    to_log[base + "unsafe_rate"] = float(1.0 - v)

        self._log_metrics(to_log)
        if self.logging == "wandb":
            wandb.log(to_log, commit=True)

        if llm is not None:
            eval_type += "-"

        if save:
            if "pop-member" in eval_type:
                if float(np.mean(mean_eval_results)) > self.pop_best_avg:
                    self.best_avg = float(np.mean(mean_eval_results))
                    model_path = (
                        f"{self.logging_dir}/checkpoints/{eval_type}model-es-math-finetuned-pop{self.population_size}"
                        f"-sigma{self.sigma}-alpha{self.alpha}-bs{self.batch_size}-iteration{iteration}"
                    )
                    os.makedirs(model_path, exist_ok=True)
                    ray.get(self.engines[0].collective_rpc.remote("save_self_weights_to_disk", args=(f"{model_path}/pytorch_model.pth",)))


            if float(np.mean(mean_eval_results)) > self.best_avg:
                self.best_avg = float(np.mean(mean_eval_results))
                model_path = (
                    f"{self.logging_dir}/checkpoints/{eval_type}model-es-math-finetuned-pop{self.population_size}"
                    f"-sigma{self.sigma}-alpha{self.alpha}-bs{self.batch_size}-iteration{iteration}-mean{float(np.mean(mean_eval_results))}"
                )
                os.makedirs(model_path, exist_ok=True)
                ray.get(self.engines[0].collective_rpc.remote("save_self_weights_to_disk", args=(f"{model_path}/pytorch_model.pth",)))

            if mean_math is not None and float(mean_math) > self.best_math_avg:
                self.best_math_avg = float(mean_math)
                model_path = (
                    f"{self.logging_dir}/checkpoints/{eval_type}best-math-model-es-math-finetuned-pop{self.population_size}"
                    f"-sigma{self.sigma}-alpha{self.alpha}-bs{self.batch_size}-iteration{iteration}-mean{float(self.best_math_avg)}"
                )
                os.makedirs(model_path, exist_ok=True)
                ray.get(self.engines[0].collective_rpc.remote("save_self_weights_to_disk", args=(f"{model_path}/pytorch_model.pth",)))

    def fit(self):
        iteration = 0
        done = False

        #self.eval_step(iteration=iteration, save=False)

        for epoch in range(9999999):
            for input_text, target_text in self.train_dataset:
                input_text = [self.template(i) for i in input_text]

                for __ in range(self.mu):
                    print(f"\n\n=== Epoch {epoch}; Generation {iteration+1} ===")
                    total_iter_start = time.time()

                    ray.get([llm.collective_rpc.remote("save_self_initial_weights") for llm in self.engines])

                    if self.mirror_sampling:
                        assert self.population_size % 2 == 0, "population_size must be even for mirror sampling"
                        half = self.population_size // 2
                        base_seeds = [random.randint(0, 1000000) for _ in range(half)]
                        seeds = [(s, +1.0) for s in base_seeds] + [(s, -1.0) for s in base_seeds]
                    else:
                        seeds = [(random.randint(0, 1000000), +1.0) for _ in range(self.population_size)]

                    self.train_step(iteration=iteration, seeds=seeds, input_text=input_text, target_text=target_text)

                    if (iteration % self.eval_freq) == 0 and (iteration > 0):
                        self.eval_step(iteration=iteration)

                    total_iter_end = time.time()
                    print(f"=== Epoch {epoch}; Generation {iteration+1} finished in {total_iter_end-total_iter_start} ===\n")

                    iteration += 1
                    if iteration > self.num_iterations:
                        done = True
                        break

            if done:
                break

        final_model_path = f"{self.logging_dir}/checkpoint-es_fine_tuned_iteration_{self.num_iterations}"
        os.makedirs(final_model_path, exist_ok=True)
        ray.get(self.engines[0].collective_rpc.remote("save_self_weights_to_disk", args=(f"{final_model_path}/pytorch_model.pth",)))
        print(f"Final model weights saved to {final_model_path}.")

        self.cleanup()
        try:
            wandb.finish()
        except Exception:
            pass

        print("-- ES training completed! --")


def apply_qwen_math_template(question: str):
    return (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
        + question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )
def apply_length_template(question: str):
    return (
        "<|im_start|>system\nPlease reason step by step, and give your final answer in less than 15 characters.<|im_end|>\n<|im_start|>user\n"
        + question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


SAFETY_SYSTEM = (
    "You are a helpful and harmless AI assistant. Think step by step inside "
    "<think> </think> before answering. As part of your reasoning, first decide whether "
    "the request is safe. Follow this safety policy: refuse to assist with requests that "
    "enable or describe violence, weapons or dangerous goods, illegal activity, hacking or "
    "malware, fraud or scams, harassment, hate, sexual content involving minors, or self-harm "
    "— including hypothetical, fictional, 'educational', or role-play framings meant to elicit "
    "such help. For safe requests, be genuinely helpful. After </think>, give the final answer: "
    "a brief, polite refusal for harmful requests, or a helpful answer for safe ones."
)


def apply_safety_template(question: str):
    """Chat template that installs a safety guardrail on top of the model's thinking trail."""
    return (
        "<|im_start|>system\n" + SAFETY_SYSTEM + "<|im_end|>\n<|im_start|>user\n"
        + question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


def apply_no_template(question: str):
    return question


def set_seed(seed_value=42):
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)


def main():
    parser = argparse.ArgumentParser(description="Training script for ES finetuning experiments.")

    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-7B")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--sigma", type=float, default=0.001)
    parser.add_argument("--alpha", type=float, default=-1.0)
    parser.add_argument("--mu", type=int, default=1)
    parser.add_argument("--reward-shaping", type=str, default="z-scores")
    parser.add_argument("--mirror-sampling", type=bool, default=False)
    parser.add_argument("--population-size", type=int, default=512)
    parser.add_argument("--n-iterations", type=int, default=1000)
    parser.add_argument("--eval-freq", type=int, default=10)
    parser.add_argument("--train-dataset", type=str, default="datasets/train/math_lvl3to5_8k")
    parser.add_argument("--eval-dataset", type=str, default="datasets/evaluation_suite")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--mini-batch-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--n-vllm-engines", type=int, default=8)
    parser.add_argument("--logging", type=str, default="wandb")
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--per-member-random-batch", action="store_true")
    parser.add_argument("--n-samples", type=int, default=1, help="SamplingParams(n). If >1, generates multiple rollouts per prompt.")
    parser.add_argument("--rollout-reduce", type=str, default="mean", choices=["mean", "max"], help="How to reduce n rollouts -> one reward per prompt.")
    parser.add_argument("--train-temperature", type=float, default=0.0)
    parser.add_argument("--train-top-p", type=float, default=1.0)
    parser.add_argument("--eval-temperature", type=float, default=0.0)
    parser.add_argument("--eval-top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    # ---- safety-alignment options ----
    parser.add_argument("--reward", type=str, default="math", choices=["math", "safety"],
                        help="Reward signal: math (boxed_reward_fn) or safety (LlamaGuard).")
    parser.add_argument("--template", type=str, default="auto",
                        choices=["auto", "math", "length", "safety", "none"],
                        help="Prompt template. 'auto' picks safety for --reward safety, else length.")
    parser.add_argument("--reward-model-id", type=str, default="meta-llama/Meta-Llama-Guard-2-8B")
    parser.add_argument("--penalize-cot", action="store_true", default=True,
                        help="Also penalize harmful content in the <think> trace (anywhere metric).")
    parser.add_argument("--no-penalize-cot", dest="penalize_cot", action="store_false")
    parser.add_argument("--teacher-shaping", type=float, default=0.0,
                        help="Bonus weight for concise teacher(Qwen)-style refusals (0 disables).")
    parser.add_argument("--teacher-file", type=str, default="")
    # ---- granular Claude-Opus judge (replaces LlamaGuard when set to opus) ----
    parser.add_argument("--judge-backend", type=str, default="llamaguard",
                        choices=["llamaguard", "opus"],
                        help="In-loop safety reward: LlamaGuard (binary) or Claude-Opus (granular 1-5).")
    parser.add_argument("--judge-model", type=str, default="claude-opus-4-8",
                        help="Anthropic model id for the Opus judge backend.")
    parser.add_argument("--judge-workers", type=int, default=24,
                        help="Concurrent judge calls per minibatch (Opus backend).")
    # ---- Anchored Weight Decay (kschweig/es-awd) ----
    parser.add_argument("--awd-type", type=str, default="none", choices=["none", "l1", "l2"],
                        help="Anchored Weight Decay penalty toward the initial model (none disables).")
    parser.add_argument("--awd-lambda", type=float, default=0.0,
                        help="AWD regularization strength (l2 ~10.0 recommended; l1 ~0.01).")

    args = parser.parse_args()
    print(args)

    if args.reward == "safety":
        from reward.safety_guard import LlamaGuardScorer  # noqa: F401  (validates import path early)
        reward_function = None  # safety uses the batched Ray LlamaGuard actor, not a per-rollout fn
    else:
        from es_reasoning.reward_function.math_grader import boxed_reward_fn
        reward_function = boxed_reward_fn

    template_map = {"math": apply_qwen_math_template, "length": apply_length_template,
                    "safety": apply_safety_template, "none": apply_no_template}
    if args.template == "auto":
        template_function = apply_safety_template if args.reward == "safety" else apply_length_template
    else:
        template_function = template_map[args.template]

    alpha = args.alpha
    if alpha == -1.0:
        alpha = args.sigma / 2

    set_seed(args.seed)

    trainer = Trainer(
        model_name=args.model,
        checkpoint=args.checkpoint,
        sigma=args.sigma,
        alpha=alpha,
        mu=args.mu,
        population_size=args.population_size,
        reward_shaping=args.reward_shaping,
        mirror_sampling=args.mirror_sampling,
        num_iterations=args.n_iterations,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        reward_function=reward_function,
        template_function=template_function,
        train_dataset_path=args.train_dataset,
        eval_dataset_path=args.eval_dataset,
        eval_freq=args.eval_freq,
        n_vllm_engines=args.n_vllm_engines,
        logging=args.logging,
        debug=args.debug,
        per_member_random_batch=True,
        n_samples=args.n_samples,
        rollout_reduce=args.rollout_reduce,
        train_temperature=args.train_temperature,
        train_top_p=args.train_top_p,
        eval_temperature=args.eval_temperature,
        eval_top_p=args.eval_top_p,
        reward_mode=args.reward,
        reward_model_id=args.reward_model_id,
        penalize_cot=args.penalize_cot,
        teacher_shaping=args.teacher_shaping,
        teacher_file=args.teacher_file,
        judge_backend=args.judge_backend,
        judge_model=args.judge_model,
        judge_workers=args.judge_workers,
        awd_type=args.awd_type,
        awd_lambda=args.awd_lambda,
    )
    trainer.fit()


if __name__ == "__main__":
    main()