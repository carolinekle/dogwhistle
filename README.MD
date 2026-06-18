# dogwhistle

Measuring how badly production content moderation APIs fail on coded extremist language, and training a classifier to do better.

Built on [SALT-NLP/silent_signals](https://huggingface.co/datasets/SALT-NLP/silent_signals) (Kruk et al., ACL 2024) — 16,199 labeled dogwhistle examples from Reddit and the Congressional Record.

---

## The finding

On the held-out test set, Llama Guard 3 catches **18%** of dogwhistle content. OpenAI Moderation catches **45.6%**. Together they miss **52.2%** of the test set entirely.

LG3 is at 0% recall on `liberal`, `climate change denier`, and `anti-GMO`. Politically coded content is essentially invisible to it.

| System | Recall |
|---|---|
| Llama Guard 3 8B | 18.0% |
| OpenAI Moderation | 45.6% |
| RoBERTa (fine-tuned) | *in progress* |

Full per-ingroup breakdown in `results/baseline_results.json`.

---

## What's here

```
explore.py           call OAI + LG3 on all 16K rows, store in SQLite
prepare_data.py      stratified 70/15/15 split, write train/val/test CSVs
analyze_baselines.py per-ingroup recall for both APIs on test set
train.py             fine-tune roberta-base on ingroup attribution (WIP)
```

---

## Setup

```bash
conda create -n dogwhistle python=3.11
conda activate dogwhistle
conda install scikit-learn pandas
pip install openai datasets tqdm python-dotenv transformers torch
```

`.env` needs `OPENAI_API_KEY` and `FEATHERLESS_API_KEY`.

---

## Notes

The dataset is all positive examples (every row is a dogwhistle), so recall is the only valid metric for the API baselines. Adding a negative class to compute full F1 is a planned extension.

`ingroup` in silent_signals labels the community that *uses* the dogwhistle, not the community targeted by it.