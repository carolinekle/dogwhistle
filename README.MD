# dogwhistle
 
Production content moderation APIs are optimizing for low false positives at scale — which means they make a deliberate tradeoff against catching coded extremist language. This project measures that tradeoff and shows what closes the gap.
 
Built on [SALT-NLP/silent_signals](https://huggingface.co/datasets/SALT-NLP/silent_signals) (Kruk et al., ACL 2024) — 16,199 labeled dogwhistle examples from Reddit and the Congressional Record.
 
---
 
## Results
 
| System | Metric | Score |
|---|---|---|
| Llama Guard 3 8B | recall | 18.0% |
| OpenAI Moderation | recall | 45.6% |
| RoBERTa (fine-tuned) | macro-F1 | 93.6% |
 
LG3 and OAI report recall only — the dataset is all positive examples so precision isn't computable without adding a negative class. RoBERTa reports macro-F1, which is stricter. The comparison is directional, not apples-to-apples.
 
Per-ingroup breakdown on the test set:
 
| ingroup | LG3 recall | OAI recall | RoBERTa F1 |
|---|---|---|---|
| Islamophobic | 37.3% | 62.7% | 0.9448 |
| anti-Asian | 29.4% | 76.5% | 0.9189 |
| anti-GMO | 0.0% | 0.0% | 0.9565 |
| anti-LGBTQ | 42.1% | 79.0% | 0.9744 |
| anti-Latino | 6.5% | 58.3% | 0.9422 |
| anti-liberal | 12.2% | 55.9% | 0.9708 |
| anti-vax | 9.5% | 23.8% | 0.9767 |
| antisemitic | 29.7% | 56.4% | 0.8902 |
| climate change denier | 0.0% | 29.0% | 0.9688 |
| conservative | 1.8% | 9.7% | 0.9700 |
| homophobic | 12.2% | 31.7% | 0.8750 |
| liberal | 0.0% | 45.5% | 0.8800 |
| racist | 15.7% | 33.3% | 0.9324 |
| transphobic | 15.1% | 59.8% | 0.9591 |
| white supremacist | 19.8% | 47.9% | 0.8738 |
| xenophobic | 27.3% | 81.8% | 0.8696 |
 
LG3 is at 0% recall on `liberal`, `anti-GMO`, and `climate change denier` — politically coded content is essentially invisible to it. Of the 2,430 test examples, 52.2% escaped both APIs entirely.
 
The model also generalizes across sources. Trained mostly on Reddit, it scores 0.8879 macro-F1 on the Stanford Congressional Record — a different register entirely.
 
---
 
## What's here
 
```
explore.py            call OAI Moderation + LG3 on all 16K rows, store in SQLite
prepare_data.py       stratified 70/15/15 split by ingroup
analyze_baselines.py  per-ingroup recall for both APIs on test set
train.py              fine-tune roberta-base, manual PyTorch loop, early stopping on val macro-F1
evaluate_model.py     load best checkpoint, run on test set, confusion matrix + source breakdown
```
 
---
 
## Setup
 
```bash
conda create -n dogwhistle python=3.11
conda activate dogwhistle
conda install scikit-learn pandas matplotlib
pip install openai datasets tqdm python-dotenv transformers torch
```
 
`.env` needs `OPENAI_API_KEY` and `FEATHERLESS_API_KEY`.
 
Run in order:
 
```bash
python explore.py
python prepare_data.py
python analyze_baselines.py
python train.py
python evaluate_model.py
```
 
---
 
## Notes
 
The gap between RoBERTa and the production APIs isn't purely a capability difference — it's a product tradeoff. Flagging dogwhistles at scale means flagging a lot of politically adjacent speech, which has real false positive costs. This project demonstrates what that tradeoff looks like in practice.
 
Dataset is all positive examples (every row is a dogwhistle), so the API baselines report recall only. Adding a negative class to enable full F1 comparison is a planned extension.
 
`ingroup` labels the community that uses the dogwhistle, not the community being targeted — standard usage in dogwhistle research.
 