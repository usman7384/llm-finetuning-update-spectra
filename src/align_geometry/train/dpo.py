"""DPO, run on top of the SFT checkpoint rather than the base model.

Tight on 6GB: DPOTrainer keeps a frozen reference copy of the model around for
the KL term, so there are two 0.5B models resident plus optimizer state.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import torch
import transformers
import trl
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from align_geometry.data.dpo_dataset import load_dpo_dataset
from align_geometry.train.common import (
    compute_warmup_and_save_steps,
    file_hash,
    find_latest_checkpoint,
    git_commit_hash,
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
        bf16=(model_cfg["model"]["dtype"] == "bfloat16"),
        max_length=model_cfg["model"]["max_seq_len"],
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
        "observed_train_loss": train_result.metrics.get("train_loss"),
        "observed_eval_loss": eval_metrics.get("eval_loss"),
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

    return save_dir
