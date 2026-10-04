"""Shared bits between the sft/lora/dpo trainers: manifests, token counting,
resume handling."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path


def git_commit_hash() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def count_prompt_completion_tokens(dataset, tokenizer) -> int:
    """Token count via the same chat template SFTTrainer applies, so the LoRA
    sweep can be matched on tokens rather than steps."""
    total = 0
    for ex in dataset:
        text = tokenizer.apply_chat_template(ex["prompt"] + ex["completion"], tokenize=False)
        total += len(tokenizer(text, add_special_tokens=False)["input_ids"])
    return total


def count_text_tokens(
    dataset, tokenizer, text_field: str = "text", max_tokens_per_example: int | None = None
) -> int:
    """Same idea for the plain-LM controls. Pass max_tokens_per_example to mirror
    the trainer's truncation, otherwise you count tokens it never sees."""
    total = 0
    for ex in dataset:
        n = len(tokenizer(ex[text_field], add_special_tokens=False)["input_ids"])
        total += min(n, max_tokens_per_example) if max_tokens_per_example else n
    return total


def is_complete_checkpoint(path: Path) -> bool:
    # trainer_state.json is written last, so a job killed mid-save leaves the
    # 1GB model file but not this. Resuming from one of those crashes.
    return (path / "trainer_state.json").exists()


def find_latest_checkpoint(save_dir: Path) -> Path | None:
    """Newest checkpoint that's actually loadable. Deletes any half-written ones
    it finds on the way (that's what save_total_limit=2 is for)."""
    if not save_dir.exists():
        return None
    checkpoints = sorted(
        (p for p in save_dir.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.split("-")[-1]),
        reverse=True,
    )
    for ckpt in checkpoints:
        if is_complete_checkpoint(ckpt):
            return ckpt
        print(f"Removing incomplete checkpoint {ckpt} (no trainer_state.json)")
        shutil.rmtree(ckpt)
    return None


def compute_warmup_and_save_steps(
    train_len: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_epochs: int,
    warmup_ratio: float,
) -> tuple[int, int, int]:
    """Returns (warmup_steps, save_steps, total_optimizer_steps).

    This TRL version dropped warmup_ratio and only takes warmup_steps, hence the
    manual conversion. Saving every ~1/15th of the run keeps an interruption
    cheap without thrashing the disk.
    """
    effective_batch = per_device_train_batch_size * gradient_accumulation_steps
    steps_per_epoch = max(1, -(-train_len // effective_batch))  # ceil div
    total_optimizer_steps = steps_per_epoch * num_epochs
    warmup_steps = round(warmup_ratio * total_optimizer_steps)
    save_steps = max(1, min(100, total_optimizer_steps // 15))
    return warmup_steps, save_steps, total_optimizer_steps
