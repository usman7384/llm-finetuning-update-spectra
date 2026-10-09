"""DPO, run on top of the SFT checkpoint rather than the base model.

Tight on 6GB: DPOTrainer keeps a frozen reference copy of the model around for
the KL term, so there are two 0.5B models resident plus optimizer state.

Second round (v2 configs): fp32 master weights with bf16 compute, as for SFT,
frozen embeddings, and the reference log-probs computed once before training
(`precompute_ref_log_probs`) so no reference copy has to fit next to the fp32
model. After training we check how many weights never moved, as train_sft
does. DPO_MAX_LENGTH in the environment overrides the sequence limit, for a
retry when 6GB is not enough.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import torch
import transformers
import trl
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from align_geometry.analysis.svd_delta import LazyWeights, is_analyzable_matrix
from align_geometry.data.dpo_dataset import load_dpo_dataset
from align_geometry.hub import weights_dir
from align_geometry.train.common import (
    compute_warmup_and_save_steps,
    file_hash,
    find_latest_checkpoint,
    freeze_input_embeddings,
    git_commit_hash,
    unchanged_fraction,
)


def train_dpo(config_path: str | Path, force: bool = False) -> Path:
    """Train from `train.init_from` (the SFT checkpoint). Resumable and
    idempotent like train_sft. Returns the checkpoint dir."""
    config_path = Path(config_path)
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_config_path = Path(cfg["train"]["model_config"])
    with open(model_config_path, encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    save_dir = Path(cfg["train"]["save_dir"])
    manifest_path = save_dir / "manifest.json"
    if manifest_path.exists() and not force:
        print(f"{manifest_path} exists, run already finished. force=True to redo.")
        return save_dir

    torch.manual_seed(cfg["seed"])

    init_from = cfg["train"]["init_from"]
    dtype = getattr(torch, model_cfg["model"]["dtype"])

    tokenizer = AutoTokenizer.from_pretrained(init_from)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        init_from,
        dtype=dtype,
        attn_implementation=model_cfg["model"]["attn_implementation"],
        trust_remote_code=model_cfg["model"]["trust_remote_code"],
    )
    if dtype != torch.float32:
        print(
            f"WARNING: training with {dtype} weights. Small updates will round away; "
            "use dtype: float32 with compute_dtype: bfloat16."
        )
    if model_cfg["memory"].get("freeze_embeddings"):
        n = freeze_input_embeddings(model)
        print(f"froze {n:,} embedding parameters")
        if model_cfg["model"].get("compute_dtype") == "bfloat16":
            model.get_input_embeddings().to(torch.bfloat16)
            if model.get_output_embeddings() is not None:
                model.get_output_embeddings().to(torch.bfloat16)
    compute_dtype = model_cfg["model"].get("compute_dtype", model_cfg["model"]["dtype"])
    max_length = int(os.environ.get("DPO_MAX_LENGTH", model_cfg["model"]["max_seq_len"]))

    log_dir = Path(cfg["train"]["log_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    resume_checkpoint = None if force else find_latest_checkpoint(save_dir)
    if resume_checkpoint is not None:
        print(f"Resuming from {resume_checkpoint}")

    indices_path = Path("data/processed") / f"{config_path.stem}_indices.json"
    train_ds, val_ds = load_dpo_dataset(
        dataset_name=cfg["data"]["dataset"],
        split=cfg["data"]["split"],
        n_examples=cfg["data"]["n_examples"],
        subsample_seed=cfg["data"]["subsample_seed"],
        val_fraction=cfg["data"]["val_fraction"],
        indices_out=indices_path,
        swap_fraction=cfg["data"].get("swap_fraction", 0.0),
    )

    memory_cfg = model_cfg["memory"]

    warmup_steps, save_steps, _ = compute_warmup_and_save_steps(
        train_len=len(train_ds),
        per_device_train_batch_size=memory_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=memory_cfg["gradient_accumulation_steps"],
        num_epochs=cfg["train"]["num_epochs"],
        warmup_ratio=cfg["train"]["warmup_ratio"],
    )

    dpo_config = DPOConfig(
        output_dir=str(save_dir),
        seed=cfg["seed"],
        beta=cfg["train"]["beta"],
        learning_rate=cfg["train"]["learning_rate"],
        lr_scheduler_type=cfg["train"]["lr_scheduler"],
        warmup_steps=warmup_steps,
        num_train_epochs=cfg["train"]["num_epochs"],
        max_grad_norm=cfg["train"]["max_grad_norm"],
        per_device_train_batch_size=memory_cfg["per_device_train_batch_size"],
        # Has to be set explicitly. The eval path builds full-vocab logits
        # (151936 wide) for an entropy stat with both models resident, and the
        # default eval batch is bigger than the training one. OOMs otherwise.
        per_device_eval_batch_size=memory_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=memory_cfg["gradient_accumulation_steps"],
        gradient_checkpointing=memory_cfg["gradient_checkpointing"],
        optim=memory_cfg["optimizer"],
        # with fp32 weights this turns on bf16 autocast and keeps fp32 masters
        bf16=(compute_dtype == "bfloat16"),
        max_length=max_length,
        precompute_ref_log_probs=cfg["train"].get("precompute_ref_log_probs", False),
        precompute_ref_batch_size=memory_cfg["per_device_train_batch_size"],
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        logging_steps=10,
        report_to=["tensorboard"],
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # trainer makes its own frozen copy
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    train_result = trainer.train(
        resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint is not None else None
    )
    eval_metrics = trainer.evaluate()

    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))

    # did the weights actually move? (see train_sft)
    before = LazyWeights(weights_dir(init_from))
    after = trainer.model.state_dict()
    keys = [k for k in before if is_analyzable_matrix(k) and k in after]
    unchanged = unchanged_fraction(after, before, keys)
    print(f"{unchanged:.2%} of attention/MLP weights unchanged")

    manifest = {
        "phase": "dpo",
        "init_from": init_from,
        "config_path": str(config_path),
        "config_hash": file_hash(config_path),
        "model_config_path": str(model_config_path),
        "model_config_hash": file_hash(model_config_path),
        "git_commit": git_commit_hash(),
        "seed": cfg["seed"],
        "beta": cfg["train"]["beta"],
        "resumed_from_checkpoint": str(resume_checkpoint)
        if resume_checkpoint is not None
        else None,
        "data_subsample_indices_file": str(indices_path),
        "swap_fraction": cfg["data"].get("swap_fraction", 0.0),
        "max_length": max_length,
        "weights_dtype": model_cfg["model"]["dtype"],
        "unchanged_weight_fraction": unchanged,
        "observed_train_loss": train_result.metrics.get("train_loss"),
        "observed_eval_loss": eval_metrics.get("eval_loss"),
        # share of held-out (real, unswapped) pairs where the chosen answer
        # gets the higher implicit reward: ~0.5 means no preference learned
        "observed_eval_reward_accuracy": eval_metrics.get("eval_rewards/accuracies"),
        "observed_eval_reward_margin": eval_metrics.get("eval_rewards/margins"),
        "observed_train_runtime_sec": train_result.metrics.get("train_runtime"),
        "observed_train_samples_per_second": train_result.metrics.get("train_samples_per_second"),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "package_versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "trl": trl.__version__,
        },
    }
    (save_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    limit = model_cfg["memory"].get("max_unchanged_fraction", 0.01)
    if unchanged > limit:
        raise RuntimeError(
            f"{unchanged:.1%} of weights never changed (limit {limit:.0%}). "
            "This update isn't a real fine-tuning update; check the weight dtype."
        )
    return save_dir
