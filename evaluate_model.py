from transformers import AutoModelForSequenceClassification, RobertaTokenizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from dogwhistle_dataset import DogwhistleDataset
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — saves to file instead of opening a window
import pandas as pd
import numpy as np
import torch
import json
import os

# ── LOAD ARTIFACTS FROM TRAINING ─────────────────────────────────────────────

# label_map.json was saved during train.py — maps integer predictions back to ingroup strings
# e.g. {0: 'Islamophobic', 1: 'anti-Asian', ...}
with open('data/label_map.json') as f:
    label_map = json.load(f)

# json keys are always strings — convert to int so label_map[0] works, not label_map['0']
label_map = {int(k): v for k, v in label_map.items()}
num_labels = len(label_map)
label_names = [label_map[i] for i in range(num_labels)]  # ordered list for sklearn reports


# ── LOAD TEST SET ─────────────────────────────────────────────────────────────

# this is the first time we open test.csv — it's been locked since prepare_data.py
# we need the ingroup labels as integers, so we rebuild them from label_map
df = pd.read_csv("data/test.csv")

# invert label_map to get string → int mapping for encoding test labels
label_to_int = {v: k for k, v in label_map.items()}
test_labels = df['ingroup'].map(label_to_int).values


# ── TOKENIZE TEST SET ─────────────────────────────────────────────────────────

# same tokenizer settings as train.py — must be identical or predictions are meaningless
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

test_encodings = tokenizer(
    list(df['content']),
    padding='max_length',
    truncation=True,
    max_length=128,
    return_tensors='pt'
)

test_dataset = DogwhistleDataset(test_encodings, test_labels)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# ── LOAD MODEL CHECKPOINT ─────────────────────────────────────────────────────

# load the same architecture used in train.py
model = AutoModelForSequenceClassification.from_pretrained(
    'roberta-base',
    num_labels=num_labels
)

# load_state_dict restores the saved weights into the model
# these are the epoch-4 weights — the best val macro-F1 checkpoint
model.load_state_dict(torch.load('checkpoints/best_model.pt', map_location='cpu'))

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
model.eval()  # eval mode — disables dropout for deterministic output


# ── RUN INFERENCE ON TEST SET ─────────────────────────────────────────────────

all_preds = []
all_labels = []

with torch.inference_mode():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # argmax picks the highest-scoring class as the prediction
        preds = outputs.logits.argmax(dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)


# ── CLASSIFICATION REPORT ─────────────────────────────────────────────────────

# classification_report gives precision, recall, and F1 per class
# precision: of everything the model called 'racist', how many actually were?
# recall:    of all actual 'racist' examples, how many did the model catch?
# F1:        harmonic mean of precision and recall
print("\n── Per-class classification report ─────────────────────────────────")
report = classification_report(all_labels, all_preds, target_names=label_names)
print(report)

overall_f1 = f1_score(all_labels, all_preds, average='macro')
print(f"Overall macro-F1 on test set: {overall_f1:.4f}")


# ── CONFUSION MATRIX ──────────────────────────────────────────────────────────

# confusion matrix shows where the model gets confused — which classes it mixes up.
# rows are true labels, columns are predicted labels.
# diagonal = correct predictions. off-diagonal = mistakes.
os.makedirs("results", exist_ok=True)

cm = confusion_matrix(all_labels, all_preds)

# normalize by row so each cell shows the fraction of true-class examples predicted as each class
# makes it easier to compare across classes of very different sizes
cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(cm_normalized, cmap='Blues')

ax.set_xticks(range(num_labels))
ax.set_yticks(range(num_labels))
ax.set_xticklabels(label_names, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(label_names, fontsize=8)

ax.set_xlabel('Predicted', fontsize=11)
ax.set_ylabel('True', fontsize=11)
ax.set_title('RoBERTa confusion matrix (normalized by true class)', fontsize=12)

plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=150)
plt.close()
print("\nConfusion matrix saved → results/confusion_matrix.png")


# ── SOURCE BREAKDOWN (OOD ANALYSIS) ──────────────────────────────────────────

# the dataset has two main sources: Reddit and the Congressional Record.
# Reddit is the majority of training data (~12K rows).
# Congressional Record is a different register entirely — more formal, longer sentences.
# this tests whether the model generalizes across writing styles (out-of-distribution).
print("\n── Macro-F1 by source ───────────────────────────────────────────────")
source_results = {}

for source in df['source'].unique():
    mask = df['source'].values == source
    if mask.sum() < 10:
        continue  # skip tiny source groups
    source_f1 = f1_score(all_labels[mask], all_preds[mask], average='macro')
    source_results[source] = {
        "n": int(mask.sum()),
        "macro_f1": round(source_f1, 4)
    }
    print(f"  {source:<35} n={mask.sum():>4}   macro-F1={source_f1:.4f}")


# ── THREE-WAY COMPARISON TABLE ────────────────────────────────────────────────

# load baseline recall numbers from analyze_baselines.py
# note the metric mismatch: LG3/OAI are recall-only (positive-class dataset, no negatives)
# RoBERTa reports full macro-F1 (a stricter metric that includes precision)
# the comparison is valid as a directional story but not apples-to-apples
with open('results/baseline_results.json') as f:
    baseline = json.load(f)

lg3_by_ingroup = baseline['per_ingroup_recall']['lg3']
oai_by_ingroup = baseline['per_ingroup_recall']['oai']

# compute per-ingroup F1 for RoBERTa
per_class_f1 = {}
for i, name in label_map.items():
    mask = all_labels == i
    if mask.sum() == 0:
        continue
    per_class_f1[name] = round(f1_score(
        all_labels[mask],
        all_preds[mask],
        average='binary',
        pos_label=i,
        labels=[i]
    ), 4) if False else round(  # use per-class from report dict instead
        f1_score(all_labels, all_preds, labels=[i], average='macro', zero_division=0), 4
    )

print("\n── Three-way comparison (test set) ─────────────────────────────────")
print(f"  {'ingroup':<25} {'LG3 recall':>12} {'OAI recall':>12} {'RoBERTa F1':>12}")
print(f"  {'─'*25} {'─'*12} {'─'*12} {'─'*12}")

comparison = {}
for name in sorted(label_map.values()):
    lg3 = lg3_by_ingroup.get(name, 0.0)
    oai = oai_by_ingroup.get(name, 0.0)
    rob = per_class_f1.get(name, 0.0)
    comparison[name] = {"lg3_recall": lg3, "oai_recall": oai, "roberta_f1": rob}
    print(f"  {name:<25} {lg3:>11.1%} {oai:>11.1%} {rob:>11.4f}")

print(f"\n  {'OVERALL':<25} {'18.0%':>12} {'45.6%':>12} {overall_f1:>12.4f}")
print(f"\n  Note: LG3/OAI report recall only (positive-class dataset).")
print(f"        RoBERTa reports macro-F1 (stricter — includes precision).")


# ── SAVE RESULTS ──────────────────────────────────────────────────────────────

results = {
    "overall_macro_f1": round(overall_f1, 4),
    "source_breakdown": source_results,
    "per_ingroup_comparison": comparison,
}

with open('results/evaluation_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nFull results saved → results/evaluation_results.json")
print("Done.")