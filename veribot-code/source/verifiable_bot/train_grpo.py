"""Active-GRPO stage (design spec section 6).

pi_ref is the cold-start SFT checkpoint (train_sft.py's "merged" output, base weights + cold-start
LoRA folded in). GRPO then trains a *fresh* LoRA adapter on top of that merged model, and a
confidence head (confidence_head.py) is attached alongside it and trained jointly (shares the same
optimizer instance, so both are updated over the course of the same run).

Group-relative advantage A_j = (R_j - mean(R_group)) / (std(R_group) + eps) and the clipped
surrogate + KL-to-reference objective are both handled natively by TRL's GRPOTrainer
(scale_rewards="group", beta=KL coefficient, epsilon=clip range) -- no custom PPO-style loop is
implemented for that part. When a group's reward std is ~0 (spec 6.4's "discard and resample"
case), TRL's epsilon in the denominator drives that group's advantages to ~0, which already yields
zero gradient contribution from that group; that is relied on instead of an explicit resample loop.

Active-GRPO's own contribution -- section 6.1's pre-rollout skip filter -- is implemented OUTSIDE
TRL's internals, as a design choice: TRL's `_generate_and_score_completions` builds fixed-shape
batches (per_device_train_batch_size x steps_per_generation) before reward functions ever run, so
"skip before generating" cannot be done inside a reward function (generation already happened) and
patching TRL's internal batch-shape machinery to drop rows mid-batch is fragile and version-sensitive.

TRL 1.8's GRPOTrainer additionally requires a materialized `datasets.Dataset` -- it explicitly
raises NotImplementedError for `datasets.IterableDataset` -- so the filter cannot even be a lazy
generator handed straight to the trainer. Training is therefore organized into short *rounds*:
before each round, `ActiveAccountStream` draws candidate accounts, runs a cheap no_grad forward
pass through the SAME live policy (base + LoRA + confidence head) to get the prefill hidden state,
evaluates the confidence head, and keeps drawing until it has collected `--round-size` accounts that
were NOT skipped; that materialized batch becomes one round's `train_dataset`. A fresh GRPOTrainer
is built for each round on top of the *same* `policy` object (weights carry over automatically;
only the optimizer's Adam moment estimates reset between rounds -- a deliberate, documented
trade-off against the far more fragile alternative of hand-rolling checkpoint resumption across
GRPOTrainer instances). Because each round re-runs the filter against the just-updated policy, the
skip decision does stay "live" across the run, just at round granularity instead of per-step.
Skipped accounts never reach `_generate_and_score_completions`, so they cost no rollout and no LLM
judge calls -- exactly the saving spec section 6.0 is after.

The confidence head's calibration loss (spec 6.1's L_calib) needs the *empirical* accuracy of a
completed rollout group, which is only known after `_generate_and_score_completions` + the reward
function have already run for that account. Fusing that into TRL's own `compute_loss` backward
would require intercepting internal state that isn't part of TRL's public surface. Instead,
`make_reward_fn` buffers (account, empirical group accuracy) pairs as a side effect while scoring
rewards, and `CalibrationCallback` periodically drains that buffer and takes an explicit, separate
forward + backward + optimizer.step() on just the confidence head's cross-entropy loss, using the
trainer's own optimizer (so it is the same optimizer state driving both the LoRA weights and the
confidence head across the run, even though the two backward passes are not the same graph).
"""
from __future__ import annotations

import argparse
import json
import random
import threading
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from .config import apply_config_defaults, load_config
from .confidence_head import ConfidenceHead, bucketize_accuracy, compute_prefill_hidden, skip_vote
from .llm_client import DeepSeekJudge
from .prompting import inference_messages
from .rewards import score_completion
from .schema import AccountRecord

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

CONFIG_MAPPING = {
    "lora_r": "lora_rank",
    "lora_alpha": "lora_alpha",
    "max_completion_length": "max_completion_length",
    "num_generations": "grpo.group_size",
    "lr": "grpo.learning_rate",
    "beta": "grpo.beta_kl",
    "epsilon": "grpo.epsilon_clip",
    "batch_size": "grpo.batch_size",
    "grad_accum": "grpo.gradient_accumulation_steps",
    "rounds": "grpo.rounds",
    "round_size": "grpo.round_size",
    "steps_per_round": "grpo.steps_per_round",
    "confidence_k": "confidence_head.k",
    "delta_p": "confidence_head.delta_p",
    "delta_m": "confidence_head.delta_m",
    "v_min": "confidence_head.v_min",
    "lambda_calib": "confidence_head.lambda_calib",
    "calib_every": "confidence_head.calib_every",
    "skip_audit_rate": "confidence_head.skip_audit_rate",
    "warmup_rounds": "confidence_head.warmup_rounds",
    "judge_k": "reward.judge_k",
    "judge_model": "reward.judge_model",
    "judge_workers": "reward.judge_workers",
    "reward_workers": "reward.reward_workers",
}


def build_policy(sft_merged: str, lora_r: int, lora_alpha: int, confidence_k: int,
                 resume_adapter: str | None = None, resume_confidence_head: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(sft_merged, local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        sft_merged, torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True
    )
    if resume_adapter:
        policy = PeftModel.from_pretrained(base, resume_adapter, is_trainable=True)
        print(f"resumed trainable GRPO LoRA from {resume_adapter}")
    else:
        policy = get_peft_model(base, LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
                                                 task_type="CAUSAL_LM", target_modules=LORA_TARGET_MODULES))
    policy.enable_input_require_grads()
    device = next(policy.parameters()).device
    confidence_head = ConfidenceHead(policy.config.hidden_size, k=confidence_k).to(device=device, dtype=torch.float32)
    if resume_confidence_head:
        confidence_head.load_state_dict(torch.load(resume_confidence_head, map_location=device, weights_only=True))
        print(f"resumed confidence head from {resume_confidence_head}")
    policy.confidence_head = confidence_head
    return policy, tokenizer, device


class ActiveAccountStream:
    """Generates AccountRecord rows from `path`, dropping any the confidence head votes to skip.

    Runs the confidence-head forward pass in eval/no_grad mode: this is purely a filtering
    decision, not a place where we want gradients. See module docstring for why calibration
    training happens separately, in CalibrationCallback, and why the caller (main()) materializes
    this generator into a fixed-size list per round rather than handing it to TRL directly.
    """

    def __init__(self, path: Path, policy, tokenizer, device, confidence_k: int, delta_p: float,
                 delta_m: float, v_min: int, skip_audit_rate: float, max_prompt_tokens: int, stats: dict):
        self.path = path
        self.policy = policy
        self.tokenizer = tokenizer
        self.device = device
        self.confidence_k = confidence_k
        self.delta_p = delta_p
        self.delta_m = delta_m
        self.v_min = v_min
        self.skip_audit_rate = skip_audit_rate
        self.max_prompt_tokens = max_prompt_tokens
        self.stats = stats

    def __call__(self, force_keep: bool = False):
        """force_keep=True bypasses the confidence-head vote entirely (still computes it, for
        logging, but always yields). Used for the head's warm-up rounds -- before it has seen any
        calibration examples it is an untrained random 5-way classifier, and an untrained head can
        pathologically vote to skip every single candidate (observed in testing: a fresh MLP can by
        chance put a confident, high-margin mass on one extreme bucket for near-duplicate inputs).
        Also used as an automatic one-shot fallback if a real filtering pass ever empties out an
        entire round (see main()), so a temporarily miscalibrated head can never hard-stop training.
        """
        with open(self.path, encoding="utf-8") as f:
            lines = f.readlines()
        random.shuffle(lines)
        was_training = self.policy.training
        self.policy.eval()
        try:
            for line in lines:
                record = AccountRecord.model_validate_json(line)
                if record.label is None:
                    continue
                prompt_messages = inference_messages(record)
                prompt_text = self.tokenizer.apply_chat_template(prompt_messages, tokenize=False,
                                                                  add_generation_prompt=True)
                enc = self.tokenizer(prompt_text, return_tensors="pt", truncation=True,
                                     max_length=self.max_prompt_tokens).to(self.device)
                with torch.no_grad():
                    hidden = compute_prefill_hidden(self.policy, enc["input_ids"], enc["attention_mask"])
                    logits = self.policy.confidence_head(hidden)
                    probs = torch.softmax(logits.float(), dim=-1)[0].tolist()
                skip = skip_vote(probs, self.delta_p, self.delta_m, self.v_min, self.confidence_k)
                self.stats["seen"] = self.stats.get("seen", 0) + 1
                if skip and random.random() < self.skip_audit_rate:
                    # Spec section 10, item 2: periodically force a would-be-skipped account through
                    # the full pipeline so the head's blind spot on skipped accounts can be audited.
                    skip = False
                    self.stats["audited"] = self.stats.get("audited", 0) + 1
                if skip and not force_keep:
                    self.stats["skipped"] = self.stats.get("skipped", 0) + 1
                    continue
                self.stats["kept"] = self.stats.get("kept", 0) + 1
                yield {"prompt": prompt_messages, "record_json": record.model_dump_json()}
        finally:
            if was_training:
                self.policy.train()


# Broadened per user direction: the original design only pooled base_correct/strong/perfect for
# comparison, which is exactly the sparse tail of the tier distribution (empirically ~1 in 10-30
# completions in early rounds) -- there were almost never 2+ such completions in the same ~36-item
# batch, so the check had no real opportunity to fire. Eligibility is now "format valid AND correct
# label" (r_cls == 1.0), i.e. every one of {correct_insufficient, correct_unsupported,
# correct_unnecessary, correct_inconsistent, base_correct, strong, perfect} -- all of these have
# real claim+pattern text worth comparing, even the ones whose r_hybrid is hard-gated to 0.
NOVELTY_SIM_THRESHOLD = 0.75   # cosine similarity above this, against a DIFFERENT account, counts
                               # as templated language reuse
NOVELTY_MAX_PENALTY = 0.5      # subtracted from `total` at similarity == 1.0; scales linearly down
                               # to 0 at NOVELTY_SIM_THRESHOLD
NOVELTY_BUFFER_CAPACITY = 300  # bounded FIFO history of (account_id, text) pairs spanning multiple
                               # batches/rounds, so new eligible completions have a much larger,
                               # cross-round reference pool to be compared against, not just the
                               # ~36 completions in their own single reward() call


def _claim_text_for_novelty(raw_text: str) -> str:
    """Best-effort extraction of claim+pattern text for the cross-account novelty check.

    Deliberately NOT format_check.check_format(): that function hard-rejects any evidence_id not
    found in a supplied valid-id set, which would require rebuilding each account's full unit map
    here just to answer a much narrower question ("what text did the claims/pattern fields say").
    A malformed/unparseable completion (already penalized at -2 elsewhere in rewards.py) just
    yields an empty string here and is skipped (nothing to compare).
    """
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        obj = json.loads(text)
        claims = obj.get("claims") or []
        return " ".join(f"{c.get('claim', '')} {c.get('pattern', '')}" for c in claims if isinstance(c, dict))
    except (json.JSONDecodeError, AttributeError, TypeError):
        return ""


def make_novelty_checker(buffer_capacity: int = NOVELTY_BUFFER_CAPACITY):
    """Returns a closure holding a bounded FIFO buffer of (account_id, claim+pattern text) pairs
    that persists ACROSS reward() calls (i.e. across training steps and rounds), analogous to
    make_reward_fn's calib_buffer. This is what actually gives the novelty check a realistic chance
    to fire: a single reward() call only sees ~36 completions, but the buffer accumulates eligible
    completions across the whole run (bounded so it doesn't grow unboundedly or go stale forever).

    Returns (apply_fn, stats_fn). apply_fn mutates `results` in place. stats_fn returns the last
    call's observability counters (eligible/pairs_compared/trigger_count/max_similarity) for logging.
    """
    buffer: deque[tuple[str, str]] = deque(maxlen=buffer_capacity)
    last_stats = {"eligible": 0, "pairs_compared": 0, "trigger_count": 0, "max_similarity": 0.0}

    def apply_fn(records: list[AccountRecord], texts: list[str], results: list) -> None:
        eligible_idx = [i for i, r in enumerate(results) if r.format_ok and r.r_cls == 1.0]
        stats = {"eligible": len(eligible_idx), "pairs_compared": 0, "trigger_count": 0, "max_similarity": 0.0}

        new_entries: list[tuple[int, str, str]] = []  # (result_idx, account_id, text)
        for i in eligible_idx:
            text = _claim_text_for_novelty(texts[i])
            if text.strip():
                new_entries.append((i, records[i].account_id, text))

        if new_entries and buffer:
            # Compare each NEW completion against the accumulated cross-round buffer (excluding
            # same-account entries), plus against the other new entries in this same batch.
            reference_texts = [t for _, t in buffer]
            reference_accounts = [a for a, _ in buffer]
            new_texts = [t for _, _, t in new_entries]
            matrix = TfidfVectorizer(min_df=1).fit_transform(new_texts + reference_texts)
            n_new = len(new_texts)
            sims = cosine_similarity(matrix[:n_new], matrix[n_new:]) if reference_texts else None
            new_vs_new = cosine_similarity(matrix[:n_new], matrix[:n_new])

            for a, (result_idx, account_id, _) in enumerate(new_entries):
                max_sim = 0.0
                if sims is not None:
                    for b, ref_account in enumerate(reference_accounts):
                        if ref_account == account_id:
                            continue
                        stats["pairs_compared"] += 1
                        max_sim = max(max_sim, sims[a, b])
                for b, (_, other_account, _) in enumerate(new_entries):
                    if b == a or other_account == account_id:
                        continue
                    stats["pairs_compared"] += 1
                    max_sim = max(max_sim, new_vs_new[a, b])

                stats["max_similarity"] = max(stats["max_similarity"], max_sim)
                if max_sim > NOVELTY_SIM_THRESHOLD:
                    frac = (max_sim - NOVELTY_SIM_THRESHOLD) / (1 - NOVELTY_SIM_THRESHOLD)
                    penalty = NOVELTY_MAX_PENALTY * frac
                    r = results[result_idx]
                    r.total = max(0.0, r.total - penalty)
                    r.detail["novelty_penalty"] = penalty
                    r.detail["novelty_max_cross_sim"] = float(max_sim)
                    stats["trigger_count"] += 1
        elif new_entries and len(new_entries) >= 2:
            # No buffer history yet (very first calls), but still compare within this batch.
            new_texts = [t for _, _, t in new_entries]
            matrix = TfidfVectorizer(min_df=1).fit_transform(new_texts)
            new_vs_new = cosine_similarity(matrix)
            for a, (result_idx, account_id, _) in enumerate(new_entries):
                max_sim = 0.0
                for b, (_, other_account, _) in enumerate(new_entries):
                    if b == a or other_account == account_id:
                        continue
                    stats["pairs_compared"] += 1
                    max_sim = max(max_sim, new_vs_new[a, b])
                stats["max_similarity"] = max(stats["max_similarity"], max_sim)
                if max_sim > NOVELTY_SIM_THRESHOLD:
                    frac = (max_sim - NOVELTY_SIM_THRESHOLD) / (1 - NOVELTY_SIM_THRESHOLD)
                    penalty = NOVELTY_MAX_PENALTY * frac
                    r = results[result_idx]
                    r.total = max(0.0, r.total - penalty)
                    r.detail["novelty_penalty"] = penalty
                    r.detail["novelty_max_cross_sim"] = float(max_sim)
                    stats["trigger_count"] += 1

        for _, account_id, text in new_entries:
            buffer.append((account_id, text))

        last_stats.clear()
        last_stats.update(stats)

    def stats_fn() -> dict:
        return dict(last_stats)

    return apply_fn, stats_fn


def make_reward_fn(judge: DeepSeekJudge, group_size: int, calib_buffer: list, calib_lock: threading.Lock,
                   suspect_path: Path, max_workers: int = 8):
    pool = ThreadPoolExecutor(max_workers=max_workers)
    novelty_apply, novelty_stats = make_novelty_checker()
    suspect_lock = threading.Lock()
    suspect_path.parent.mkdir(parents=True, exist_ok=True)

    def reward(completions, record_json, **_kwargs) -> list[float]:
        records = [AccountRecord.model_validate_json(r) for r in record_json]
        texts = [c[0]["content"] if isinstance(c, list) else str(c) for c in completions]
        futures = [pool.submit(score_completion, rec, text, judge) for rec, text in zip(records, texts)]
        results = [f.result() for f in futures]

        novelty_apply(records, texts, results)

        # Keep reward-v2 behavior observable in the live log without storing prompts or raw account
        # content. This makes it possible to verify that hard gates and 3/4-point tiers are actually
        # firing after a checkpoint-boundary rollout-policy change.
        tier_counts = Counter(
            "invalid_format" if not result.format_ok else result.detail.get("branch", "unknown")
            for result in results
        )
        print("reward tier counts:", dict(sorted(tier_counts.items())), flush=True)
        for label in ("bot", "human"):
            indices = [i for i, record in enumerate(records) if record.label == label]
            suf = [results[i].s_suf for i in indices if results[i].s_suf is not None]
            nec = [results[i].s_nec for i in indices if results[i].s_nec is not None]
            disagree = sum(results[i].detail.get("agreement") == "disagree" or
                           str(results[i].detail.get("branch", "")).startswith("disagree_")
                           for i in indices)
            print(
                f"reward audit label={label} n={len(indices)} "
                f"mean_s_suf={sum(suf)/len(suf) if suf else float('nan'):.4f} "
                f"mean_s_nec={sum(nec)/len(nec) if nec else float('nan'):.4f} "
                f"disagree_rate={disagree/len(indices) if indices else 0.0:.4f}", flush=True,
            )
        suspects = [
            {"source": record.source, "account_id": record.account_id,
             "trusted_label": record.label, "upstream_verdict": record.upstream_verdict,
             "s_suf": result.s_suf, "s_nec": result.s_nec}
            for record, result in zip(records, results) if result.detail.get("record_hgt_suspect")
        ]
        if suspects:
            with suspect_lock, suspect_path.open("a", encoding="utf-8") as handle:
                for suspect in suspects:
                    handle.write(json.dumps(suspect, ensure_ascii=False) + "\n")
        stats = novelty_stats()
        print(f"novelty check: eligible={stats['eligible']} pairs_compared={stats['pairs_compared']} "
              f"trigger_count={stats['trigger_count']} max_similarity={stats['max_similarity']:.3f}", flush=True)

        # TRL lays out completions as consecutive blocks of `group_size` per unique prompt.
        with calib_lock:
            for i in range(0, len(records), group_size):
                block_records = records[i:i + group_size]
                block_results = results[i:i + group_size]
                if not block_records:
                    continue
                accuracy = sum(
                    1.0 for r in block_results if r.detail.get("derived_prediction_correct", False)
                ) / len(block_results)
                calib_buffer.append({"record_json": block_records[0].model_dump_json(), "accuracy": accuracy})

        return [r.total for r in results]

    return reward


class CalibrationCallback(TrainerCallback):
    def __init__(self, policy, tokenizer, device, calib_buffer: list, calib_lock: threading.Lock,
                 lambda_calib: float, calib_every: int, confidence_k: int, max_prompt_tokens: int,
                 max_batch: int = 32):
        self.policy = policy
        self.tokenizer = tokenizer
        self.device = device
        self.buffer = calib_buffer
        self.lock = calib_lock
        self.lambda_calib = lambda_calib
        self.calib_every = calib_every
        self.confidence_k = confidence_k
        self.max_prompt_tokens = max_prompt_tokens
        self.max_batch = max_batch
        self.trainer = None  # set externally after GRPOTrainer construction

    def on_step_end(self, args, state, control, **kwargs):
        if self.trainer is None or state.global_step == 0 or state.global_step % self.calib_every != 0:
            return control
        with self.lock:
            if not self.buffer:
                return control
            batch = self.buffer[-self.max_batch:]
            self.buffer.clear()

        prompts, targets = [], []
        for item in batch:
            record = AccountRecord.model_validate_json(item["record_json"])
            prompts.append(self.tokenizer.apply_chat_template(inference_messages(record), tokenize=False,
                                                               add_generation_prompt=True))
            targets.append(bucketize_accuracy(item["accuracy"], self.confidence_k))

        enc = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                             max_length=self.max_prompt_tokens).to(self.device)
        self.policy.train()
        hidden = compute_prefill_hidden(self.policy, enc["input_ids"], enc["attention_mask"])
        logits = self.policy.confidence_head(hidden)
        target_tensor = torch.tensor(targets, device=self.device, dtype=torch.long)
        loss = self.lambda_calib * F.cross_entropy(logits.float(), target_tensor)

        optimizer = self.trainer.optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.trainer.log({"calibration_loss": loss.item(), "calibration_batch_size": len(batch)})
        return control


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="unified AccountRecord jsonl, one per line")
    ap.add_argument("--sft-merged", type=Path, required=True, help="train_sft.py --output/merged directory")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--judge-cache", type=Path, required=True)
    ap.add_argument("--resume-adapter", type=Path, default=None,
                    help="completed prior round/checkpoint containing a GRPO LoRA adapter")
    ap.add_argument("--resume-confidence-head", type=Path, default=None,
                    help="confidence_head.pt saved after a completed prior round")
    ap.add_argument("--start-round", type=int, default=0,
                    help="first round index to run when continuing an interrupted multi-round run")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--judge-k", type=int, default=5)
    ap.add_argument("--judge-workers", type=int, default=32,
                    help="maximum concurrent Judge API requests; each K-vote group submits in parallel")
    ap.add_argument("--reward-workers", type=int, default=12,
                    help="parallel completion-level reward evaluations per rollout batch")
    ap.add_argument("--num-generations", type=int, default=6, help="G, group size, spec section 6.2")
    ap.add_argument("--beta", type=float, default=0.04, help="KL coefficient toward pi_ref")
    ap.add_argument("--epsilon", type=float, default=0.2, help="PPO-style clip range, epsilon_c")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-prompt-tokens", type=int, default=4096)
    ap.add_argument("--max-completion-length", type=int, default=640)
    ap.add_argument("--rounds", type=int, default=10,
                    help="each round re-filters the pool with the current policy, then materializes "
                         "--round-size kept accounts into one GRPOTrainer.train() call")
    ap.add_argument("--round-size", type=int, default=48, help="kept accounts (prompts) per round")
    ap.add_argument("--steps-per-round", type=int, default=-1,
                    help="-1 = one full epoch over the round's --round-size accounts")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    # Active-GRPO confidence head / skip filter (spec section 6.1)
    ap.add_argument("--confidence-k", type=int, default=5)
    ap.add_argument("--delta-p", type=float, default=0.6)
    ap.add_argument("--delta-m", type=float, default=0.3)
    ap.add_argument("--v-min", type=int, default=2)
    ap.add_argument("--skip-audit-rate", type=float, default=0.05)
    ap.add_argument("--lambda-calib", type=float, default=0.1)
    ap.add_argument("--calib-every", type=int, default=1, help="run a calibration step every N GRPO steps")
    ap.add_argument("--warmup-rounds", type=int, default=2,
                    help="rounds where the filter is bypassed (force_keep) so the head sees real "
                         "calibration data before it is trusted to skip anything")
    ap.add_argument("--config", type=Path, default=None,
                    help="YAML config (default configs/veriroute_lite.yaml); explicit CLI flags override it")
    args, _ = ap.parse_known_args()
    apply_config_defaults(ap, load_config(args.config), CONFIG_MAPPING)
    args = ap.parse_args()

    total_accounts = args.rounds * args.round_size
    if args.round_size % 2:
        raise ValueError("--round-size must be even for strict upstream-verdict balancing")
    print(f"NOTE: this run will touch at most {total_accounts} account-slots "
          f"({args.rounds} rounds x {args.round_size} accounts/round, with replacement across rounds) "
          "-- NOT one pass over the full input file. Raise --rounds/--round-size for more coverage.")

    policy, tokenizer, device = build_policy(
        str(args.sft_merged), args.lora_r, args.lora_alpha, args.confidence_k,
        str(args.resume_adapter) if args.resume_adapter else None,
        str(args.resume_confidence_head) if args.resume_confidence_head else None,
    )
    judge = DeepSeekJudge(cache_dir=args.judge_cache, model=args.judge_model, k=args.judge_k,
                          max_workers=args.judge_workers)

    skip_stats: dict = {}
    calib_buffer: list = []
    calib_lock = threading.Lock()
    reward_fn = make_reward_fn(
        judge, args.num_generations, calib_buffer, calib_lock,
        args.output / "hgt_suspect_candidates.jsonl", max_workers=args.reward_workers,
    )

    trainer = None
    for round_idx in range(args.start_round, args.rounds):
        stream = ActiveAccountStream(args.data, policy, tokenizer, device, args.confidence_k, args.delta_p,
                                     args.delta_m, args.v_min, args.skip_audit_rate, args.max_prompt_tokens,
                                     skip_stats)
        force_keep = round_idx < args.warmup_rounds
        target_per_verdict = args.round_size // 2
        rows_by_verdict = {"bot": [], "human": []}
        for row in stream(force_keep=force_keep):
            verdict = AccountRecord.model_validate_json(row["record_json"]).upstream_verdict
            if verdict in rows_by_verdict and len(rows_by_verdict[verdict]) < target_per_verdict:
                rows_by_verdict[verdict].append(row)
            if all(len(group) >= target_per_verdict for group in rows_by_verdict.values()):
                break
        rows = rows_by_verdict["bot"] + rows_by_verdict["human"]
        random.shuffle(rows)
        if not rows and not force_keep:
            # Automatic fallback (see ActiveAccountStream docstring): don't let a temporarily
            # miscalibrated head hard-stop a multi-hour run.
            print(f"round {round_idx}: filter rejected the entire pool, forcing this round through unfiltered")
            rows_by_verdict = {"bot": [], "human": []}
            for row in stream(force_keep=True):
                verdict = AccountRecord.model_validate_json(row["record_json"]).upstream_verdict
                if verdict in rows_by_verdict and len(rows_by_verdict[verdict]) < target_per_verdict:
                    rows_by_verdict[verdict].append(row)
                if all(len(group) >= target_per_verdict for group in rows_by_verdict.values()):
                    break
            rows = rows_by_verdict["bot"] + rows_by_verdict["human"]
            random.shuffle(rows)
        if not rows:
            print(f"round {round_idx}: dataset file produced zero usable accounts, stopping")
            break
        if any(len(rows_by_verdict[label]) != target_per_verdict for label in ("bot", "human")):
            raise RuntimeError(
                f"round {round_idx} could not reach strict verdict balance: "
                f"bot={len(rows_by_verdict['bot'])} human={len(rows_by_verdict['human'])}"
            )
        print(
            f"round {round_idx}: upstream verdict balance bot={len(rows_by_verdict['bot'])} "
            f"human={len(rows_by_verdict['human'])}", flush=True,
        )
        dataset = Dataset.from_list(rows)

        config = GRPOConfig(
            output_dir=str(args.output / f"round_{round_idx:04d}"),
            learning_rate=args.lr,
            bf16=True,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_generations=args.num_generations,
            max_completion_length=args.max_completion_length,
            num_train_epochs=1.0,
            max_steps=args.steps_per_round,
            beta=args.beta,
            epsilon=args.epsilon,
            loss_type="grpo",
            scale_rewards="group",
            num_iterations=1,
            temperature=1.0,
            lr_scheduler_type="constant",  # keeps LR meaningful across independently-constructed rounds
            save_total_limit=1,
            logging_steps=1,
            log_completions=True,
            report_to="none",
            remove_unused_columns=False,
        )

        trainer = GRPOTrainer(model=policy, reward_funcs=reward_fn, args=config, train_dataset=dataset)
        calib_cb = CalibrationCallback(policy, tokenizer, device, calib_buffer, calib_lock, args.lambda_calib,
                                       args.calib_every, args.confidence_k, args.max_prompt_tokens)
        calib_cb.trainer = trainer
        trainer.add_callback(calib_cb)
        trainer.train()
        # PEFT is saved by Trainer; persist the non-PEFT confidence head alongside every completed
        # round so a later continuation restores the entire Active-GRPO policy state.
        trainer.save_model(str(args.output / f"round_{round_idx:04d}"))
        torch.save(policy.confidence_head.state_dict(),
                   args.output / f"round_{round_idx:04d}" / "confidence_head.pt")
        print(f"round {round_idx}: trained on {len(rows)} accounts; skip stats so far: {skip_stats}")

    if trainer is None:
        raise RuntimeError("no round ever produced a trainer -- the skip filter rejected every candidate account")
    trainer.save_model(str(args.output / "final"))
    torch.save(policy.confidence_head.state_dict(), args.output / "final" / "confidence_head.pt")
    print("final skip filter stats:", skip_stats)


if __name__ == "__main__":
    main()
