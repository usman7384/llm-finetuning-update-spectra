import json

import pytest

from align_geometry.train.common import ensure_chat_template, resolve_target_tokens


def test_fixed_target_tokens():
    assert resolve_target_tokens({"target_tokens": 1750543}) == 1750543


def test_target_tokens_read_from_manifest(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"expected_train_tokens": 1234567}))
    assert resolve_target_tokens({"target_tokens_from": str(manifest)}) == 1234567


def test_missing_manifest_says_what_to_do(tmp_path):
    with pytest.raises(FileNotFoundError, match="Train the run"):
        resolve_target_tokens({"target_tokens_from": str(tmp_path / "nope.json")})


def test_existing_chat_template_is_kept():
    class Tok:
        chat_template = "keep me"

    tok = Tok()
    ensure_chat_template(tok, "some/other-repo")  # would need a download if it tried
    assert tok.chat_template == "keep me"


def test_weights_dir_keeps_local_paths(tmp_path):
    from align_geometry.hub import weights_dir

    # an existing directory must never trigger a download
    assert weights_dir(tmp_path) == tmp_path
    assert weights_dir(str(tmp_path)) == tmp_path


def test_unchanged_fraction_counts_untouched_entries():
    import torch

    from align_geometry.train.common import unchanged_fraction

    before = {"w": torch.zeros(4, 5), "v": torch.ones(2, 2)}
    after = {"w": before["w"].clone(), "v": before["v"].clone()}
    after["w"][0] += 1e-3  # 5 of the 24 entries move
    assert abs(unchanged_fraction(after, before, ["w", "v"]) - 19 / 24) < 1e-12


def test_bf16_weights_lose_small_updates():
    # the bug this check exists for: a 2e-5 step on a weight of 0.02 vanishes in
    # bf16 but survives in fp32
    import torch

    w = torch.tensor([0.02])
    assert (w.bfloat16() + 2e-5).float() == w.bfloat16().float()
    assert (w + 2e-5) != w


def test_freeze_input_embeddings_freezes_tied_head():
    import torch

    from align_geometry.train.common import freeze_input_embeddings

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(10, 4)
            self.head = torch.nn.Linear(4, 10, bias=False)
            self.head.weight = self.emb.weight  # tied, like Qwen2.5 and SmolLM2
            self.called = False

        def get_input_embeddings(self):
            return self.emb

        def get_output_embeddings(self):
            return self.head

        def enable_input_require_grads(self):
            self.called = True

    m = Tiny()
    assert freeze_input_embeddings(m) == 40  # tied, so counted once
    assert not m.emb.weight.requires_grad and not m.head.weight.requires_grad
    assert m.called


def test_shuffle_words_keeps_words_and_breaks_order(monkeypatch):
    from datasets import Dataset

    import align_geometry.data.lm_dataset as lm

    doc = " ".join(f"w{i}" for i in range(60))
    monkeypatch.setattr(lm, "load_dataset", lambda *a, **k: Dataset.from_dict({"text": [doc] * 4}))

    class Tok:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": text.split()}

    train, _ = lm.load_lm_dataset(
        "x", None, "train", target_tokens=200, tokenizer=Tok(), subsample_seed=0,
        val_fraction=0.25, min_chars=10, shuffle_words=True,
    )  # fmt: skip
    out = train[0]["text"].split()
    assert sorted(out) == sorted(doc.split())
    assert out != doc.split()
