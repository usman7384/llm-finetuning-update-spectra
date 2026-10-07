"""Full-parameter SFT, and the plain-LM control runs.

Also drives configs/control_lm.yaml and configs/control_code.yaml, which set
data.format: text to switch off the chat template and prompt masking.

The gradient checkpointing / 8-bit AdamW / frozen-embedding settings in the
model config are what make full fine-tuning fit in 6GB. They aren't tuning
knobs.

Precision matters more than it looks. With `dtype: bfloat16` the weights
themselves are bf16 and any step smaller than half a bf16 rounding unit is
lost, so most weights never move (our first runs left 78-96% of entries
unchanged). Configs now load fp32 weights (`dtype: float32`) and compute in
bf16 (`compute_dtype: bfloat16`), and every run checks afterwards how many
weights stayed exactly the same.
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
from trl import SFTConfig, SFTTrainer

from align_geometry.analysis.svd_delta import LazyWeights, is_analyzable_matrix
from align_geometry.data.lm_dataset import load_lm_dataset
from align_geometry.data.sft_dataset import load_sft_dataset
from align_geometry.hub import weights_dir
from align_geometry.train.common import (
    compute_warmup_and_save_steps,
    count_prompt_completion_tokens,
    count_text_tokens,
    ensure_chat_template,
    file_hash,
    find_latest_checkpoint,
    freeze_input_embeddings,
    git_commit_hash,
    resolve_target_tokens,
    unchanged_fraction,
)


def train_sft(config_path: str | Path, force: bool = False) -> Path:
    """Train per `config_path`, return the checkpoint dir.

    Safe to interrupt. Re-running the same config resumes from the last
    step-level checkpoint, or does nothing if the run already finished
    (pass force=True to redo it).
    """
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

    model_id = model_cfg["model"]["name_or_path"]
    revision = model_cfg["model"]["revision"]
    dtype = getattr(torch, model_cfg["model"]["dtype"])

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    ensure_chat_template(tokenizer, model_cfg["model"].get("chat_template_from"))

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
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

    log_dir = Path(cfg["train"]["log_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    resume_checkpoint = None if force else find_latest_checkpoint(save_dir)
    if resume_checkpoint is not None:
        print(f"Resuming from {resume_checkpoint}")

    indices_path = Path("data/processed") / f"{config_path.stem}_indices.json"

    # "text" = the plain-LM control path. Default keeps the SFT config as-is.
    data_format = cfg["data"].get("format", "conversational")
    if data_format == "text":
        train_ds, val_ds = load_lm_dataset(
            dataset_name=cfg["data"]["dataset"],
            config_name=cfg["data"].get("config_name"),
            split=cfg["data"].get("split", "train"),
            target_tokens=resolve_target_tokens(cfg["data"]),
            tokenizer=tokenizer,
            subsample_seed=cfg["data"]["subsample_seed"],
            val_fraction=cfg["data"]["val_fraction"],
            text_field=cfg["data"].get("text_field", "text"),
            max_tokens_per_example=model_cfg["model"]["max_seq_len"],
            indices_out=indices_path,
        )
        expected_train_tokens = count_text_tokens(
            train_ds, tokenizer, max_tokens_per_example=model_cfg["model"]["max_seq_len"]
        )
        # No prompt to mask when it's just raw text.
        completion_only_loss = False
    else:
        train_ds, val_ds = load_sft_dataset(
            dataset_name=cfg["data"]["dataset"],
            n_examples=cfg["data"]["n_examples"],
            subsample_seed=cfg["data"]["subsample_seed"],
            val_fraction=cfg["data"]["val_fraction"],
            indices_out=indices_path,
        )
        expected_train_tokens = count_prompt_completion_tokens(
            train_ds, tokenizer, max_tokens_per_example=model_cfg["model"]["max_seq_len"]
        )
        completion_only_loss = True

    memory_cfg = model_cfg["memory"]

    warmup_steps, save_steps, _ = compute_warmup_and_save_steps(
        train_len=len(train_ds),
        per_device_train_batch_size=memory_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=memory_cfg["gradient_accumulation_steps"],
        num_epochs=cfg["train"]["num_epochs"],
        warmup_ratio=cfg["train"]["warmup_ratio"],
    )

    sft_config = SFTConfig(
        output_dir=str(save_dir),
        # No logging_dir arg in this transformers version; TB logs land under
        # output_dir/runs/ on their own.
        seed=cfg["seed"],
        learning_rate=cfg["train"]["learning_rate"],
        lr_scheduler_type=cfg["train"]["lr_scheduler"],
        warmup_steps=warmup_steps,
        num_train_epochs=cfg["train"]["num_epochs"],
        weight_decay=cfg["train"]["weight_decay"],
        max_grad_norm=cfg["train"]["max_grad_norm"],
        per_device_train_batch_size=memory_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=memory_cfg["gradient_accumulation_steps"],
        gradient_checkpointing=memory_cfg["gradient_checkpointing"],
        optim=memory_cfg["optimizer"],
        # with fp32 weights this turns on bf16 autocast and keeps fp32 masters
        bf16=(model_cfg["model"].get("compute_dtype", model_cfg["model"]["dtype"]) == "bfloat16"),
        max_length=model_cfg["model"]["max_seq_len"],
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        logging_steps=10,
        report_to=["tensorboard"],
        completion_only_loss=completion_only_loss,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    train_result = trainer.train(
        resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint is not None else None
    )

    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))

    # did the weights actually move? (see the module docstring)
    base = LazyWeights(weights_dir(model_id))
    after = trainer.model.state_dict()
    keys = [k for k in base if is_analyzable_matrix(k) and k in after]
    unchanged = unchanged_fraction(after, base, keys)
    print(f"{unchanged:.2%} of attention/MLP weights unchanged")

    manifest = {
        "phase": cfg.get("phase", "sft"),
        "data_format": data_format,
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
        "weights_dtype": model_cfg["model"]["dtype"],
        "unchanged_weight_fraction": unchanged,
        "observed_train_loss": train_result.metrics.get("train_loss"),
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
