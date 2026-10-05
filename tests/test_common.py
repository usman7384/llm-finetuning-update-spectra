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
