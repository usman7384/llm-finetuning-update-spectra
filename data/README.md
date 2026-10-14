# data/

Datasets are pulled from the Hugging Face Hub when the training scripts run, so
nothing needs downloading by hand.

`processed/` holds one `*_indices.json` per training run: the exact source rows
that run trained and validated on. They're committed so any run's data can be
rebuilt exactly, without relying on a seed giving the same shuffle in some
future version of `datasets`.

| Used for | Dataset |
|---|---|
| SFT | `yahma/alpaca-cleaned`, 10k sample |
| DPO | `HuggingFaceH4/ultrafeedback_binarized`, `train_prefs`, 10k pairs |
| wikitext control | `Salesforce/wikitext`, `wikitext-103-raw-v1` |
| shuffled-words control | the same wikitext rows with the words of each document shuffled (`shuffle_words: true`) |
| code control | `codeparrot/codeparrot-clean-valid` |
| Timber case study | `openai/gsm8k` (test), `Idavidrein/gpqa` (`gpqa_diamond`, gated: accept the terms and log in) |
| style evals (first round only, not in the paper) | IFEval, `natolambert/xstest-v2-copy` |
| capability evals (first round only, not in the paper) | GSM8K, ARC-Challenge (through lm-eval) |

The `v2_*_indices.json` files belong to the runs in the paper; the files without
the prefix are the first (bf16) round.
