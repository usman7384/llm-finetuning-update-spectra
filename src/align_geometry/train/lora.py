"""One point of the LoRA rank sweep.

Same data and token budget as the full SFT run so the comparison is matched.
Each run saves two things: the raw adapter under adapter/ (the subspace
analysis needs A and B separately) and a merged model at the root (so the eval
harness and SVD code can treat it like any other checkpoint).

scripts/lora_sweep.py drives this across ranks and learning rates. The LR
gets swept per rank because Biderman et al. found the LoRA-vs-full-FT gap is
sensitive to it.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import torch
import transformers
import trl
import yaml
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from align_geometry.data.sft_dataset import load_sft_dataset
from align_geometry.train.common import (
    compute_warmup_and_save_steps,
    count_prompt_completion_tokens,
    file_hash,
    find_latest_checkpoint,
    git_commit_hash,
)


def run_name(rank: int, learning_rate: float) -> str:
    return f"r{rank}_lr{learning_rate:.0e}"


def train_lora(
    config_path: str | Path, rank: int, learning_rate: float, force: bool = False
) -> Path:
    """Train one (rank, lr) point. Resumable and idempotent like train_sft.

    Returns the run directory: merged model at the root, adapter under adapter/.
    """
    config_path = Path(config_path)
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_config_path = Path(cfg["train"]["model_config"])
    with open(model_config_path, encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    sweep_cfg = cfg["sweep"]
    lora_alpha = int(sweep_cfg["alpha_to_rank_ratio"] * rank)

    name = run_name(rank, learning_rate)
    save_dir = Path(cfg["train"]["save_dir"]) / name
    manifest_path = save_dir / "manifest.json"
    if manifest_path.exists() and not force:
        print(f"{manifest_path} exists, run already finished. force=True to redo.")
        return save_dir

    torch.manual_seed(cfg["seed"])

    model_id = model_cfg["model"]["name_or_path"]
    revision = model_cfg["model"]["revision"]
    dtype = getattr(torch, model_cfg["model"]["dtype"])

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=dtype,
        attn_implementation=model_cfg["model"]["attn_implementation"],
        trust_remote_code=model_cfg["model"]["trust_remote_code"],
    )

    log_dir = Path(cfg["train"]["log_dir"]) / name
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    resume_checkpoint = None if force else find_latest_checkpoint(save_dir)
    if resume_checkpoint is not None:
        print(f"Resuming from {resume_checkpoint}")

    # Same seed and sizes as configs/sft.yaml, so this picks the same rows.
    indices_path = Path("data/processed") / f"{config_path.stem}_indices.json"
    train_ds, val_ds = load_sft_dataset(
        dataset_name=cfg["data"]["dataset"],
        n_examples=cfg["data"]["n_examples"],
        subsample_seed=cfg["data"]["subsample_seed"],
        val_fraction=cfg["data"]["val_fraction"],
        indices_out=indices_path,
    )

    expected_train_tokens = count_prompt_completion_tokens(train_ds, tokenizer)

    peft_config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        target_modules=sweep_cfg["target_modules"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Only the adapters get optimizer state, so none of the full-FT memory
    # tricks are needed here. Plain AdamW and a bigger batch fit fine.
    per_device_train_batch_size = 4
    gradient_accumulation_steps = 4

    warmup_steps, save_steps, _ = compute_warmup_and_save_steps(
        train_len=len(train_ds),
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_epochs=cfg["train"]["num_epochs"],
        warmup_ratio=cfg["train"]["warmup_ratio"],
    )

    sft_config = SFTConfig(
        output_dir=str(save_dir),
        seed=cfg["seed"],
        learning_rate=learning_rate,
        lr_scheduler_type=cfg["train"]["lr_scheduler"],
        warmup_steps=warmup_steps,
        num_train_epochs=cfg["train"]["num_epochs"],
        max_grad_norm=cfg["train"]["max_grad_norm"],
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=False,
        optim="adamw_torch",
        bf16=(model_cfg["model"]["dtype"] == "bfloat16"),
        max_length=model_cfg["model"]["max_seq_len"],
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        logging_steps=10,
        report_to=["tensorboard"],
        completion_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    train_result = trainer.train(
        resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint is not None else None
    )
    eval_metrics = trainer.evaluate()

    # Keep the unmerged adapter around; the subspace analysis needs A and B.
    trainer.model.save_pretrained(str(save_dir / "adapter"))

    # merge_and_unload() mutates the model, so this has to come after the
    # adapter save above.
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))

    manifest = {
        "phase": "lora",
        "rank": rank,
        "learning_rate": learning_rate,
        "lora_alpha": lora_alpha,
        "config_path": str(config_path),
        "config_hash": file_hash(config_path),
        "model_config_path": str(model_config_path),
        "model_config_hash": file_hash(model_config_path),
        "git_commit": git_commit_hash(),
        "seed": cfg["seed"],
        "resumed_from_checkpoint": str(resume_checkpoint)
        if resume_checkpoint is not None
        else None,
        "data_subsample_indices_file": str(indices_path),
        "expected_train_tokens": expected_train_tokens,
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
