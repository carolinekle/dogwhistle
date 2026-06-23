from transformers import AutoModelForSequenceClassification, RobertaTokenizer, get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torch.optim import AdamW
from dogwhistle_dataset import DogwhistleDataset
import pandas as pd
import numpy as np
import torch
import json
import os

# DATA LOADING 

# load the three splits produced by prepare_data.py into pandas DataFrames
# each CSV has: content (text), ingroup (label), dog_whistle, source, llama_response, flagged
df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/val.csv")
test_df = pd.read_csv("data/test.csv")


#LABEL ENCODING 
"""
LabelEncoder turns ingroup strings into integers that CrossEntropyLoss can consume.
it works by sorting all unique labels alphabetically, then assigning position numbers:
['Islamophobic', 'anti-Asian', 'anti-GMO', ...] → [0, 1, 2, ...]

rule: fit ONCE on train, then transform everywhere else.
fitting on val or test would be leakage — the encoder would use information
from data the model is supposed to have never seen.
"""

le = LabelEncoder()
le.fit(df['ingroup'])

train_labels = le.transform(df['ingroup'])
val_labels = le.transform(val_df['ingroup'])
"""
test labels stay untouched here — they're only used in evaluate_model.py

save the integer → ingroup string mapping to disk.
we need this later in evaluate_model.py to convert predictions like [13, 0, 7]
back into readable strings like ['racist', 'Islamophobic', 'antisemitic'].
enumerate(le.classes_) produces (0, 'Islamophobic'), (1, 'anti-Asian'), etc.
the dict comprehension builds {0: 'Islamophobic', 1: 'anti-Asian', ...}
"""

label_map = {int(i): label for i, label in enumerate(le.classes_)}
with open('data/label_map.json', 'w') as f:
    json.dump(label_map, f, indent=2)


# TOKENIZATION 
"""
RoBERTa doesn't read raw text — it reads sequences of integer token IDs
from a fixed vocabulary of ~50k subword units.
the tokenizer handles the conversion and produces two tensors per example:

  input_ids:      each word/subword in the text mapped to its vocabulary ID
  attention_mask: 1 for real tokens, 0 for padding tokens
                  tells the model which positions to attend to and which to ignore

padding='max_length' pads shorter texts to exactly 128 tokens with zeros
truncation=True cuts texts longer than 128 tokens (our p95 is ~73 words, so rare)
return_tensors='pt' returns PyTorch tensors instead of plain Python lists
"""
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

train_encodings = tokenizer(
    list(df['content']),
    padding='max_length',
    truncation=True,
    max_length=128,
    return_tensors='pt'
)

val_encodings = tokenizer(
    list(val_df['content']),
    padding='max_length',
    truncation=True,
    max_length=128,
    return_tensors='pt'
)


# DATASET AND DATALOADER
"""
DogwhistleDataset wraps the encodings and labels into a PyTorch Dataset object.
the DataLoader needs a Dataset because it only knows how to batch and shuffle —
it doesn't know what our data looks like or how to access individual rows
Dataset provides the interface: __len__ (how many rows?) and __getitem__ (give me row N)
"""

train_dataset = DogwhistleDataset(train_encodings, train_labels)
val_dataset = DogwhistleDataset(val_encodings, val_labels)

# DataLoader handles batching and shuffling automatically
# shuffle=True on train: prevents the model from memorizing the order of examples
# shuffle=False on val: order doesn't matter, we just want consistent results
# batch_size=16: each training step processes 16 examples at once
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


#MODEL
"""
AutoModelForSequenceClassification loads roberta-base (the pretrained language model)
and attaches a classification head on top — a single linear layer that takes
RoBERTa's output and produces one score per class.
those scores (logits) are what CrossEntropyLoss and argmax operate on.
the classification head starts with random weights; roberta-base weights are pretrained
fine-tuning updates both, but the head changes much more dramatically.
"""

num_labels = len(le.classes_)  # derived from encoder so it stays correct if classes change
model = AutoModelForSequenceClassification.from_pretrained(
    'roberta-base',
    num_labels=num_labels
)
"""
# move the model to MPS (Metal Performance Shaders — Apple Silicon's GPU).
# by default PyTorch creates everything in CPU RAM.
# the GPU can do thousands of matrix multiplications in parallel, making training
# dramatically faster than CPU. model and data must always be on the same device —
# you cannot do operations across CPU and GPU memory
"""

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
print(f"Training on: {device}")


# CLASS WEIGHTS 
"""
our dataset has severe class imbalance: racist=5097 rows, other=28 rows
without correction, the model learns to just predict 'racist' for everything —
that still produces decent loss because racist is 31% of training data.

class weights fix this by scaling the loss: a mistake on a rare class (high weight)
produces a larger loss signal than a mistake on a common class (low weight),
forcing the model to pay attention to small classes

'balanced' computes weights as: total_rows / (n_classes * class_count)
so rare classes get proportionally higher weights automatically
"""

classes = np.unique(train_labels)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_labels)

# weights must be a float tensor on the same device as the model
class_weights = torch.tensor(weights, dtype=torch.float).to(device)

"""
CrossEntropyLoss: standard loss for multi-class classification.
internally converts logits to probabilities (softmax), then measures
how low the probability was assigned to the correct class.
the weight argument scales each class's loss contribution.
"""

loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)


# OPTIMIZER AND SCHEDULER 
"""
AdamW is the standard optimizer for transformer fine-tuning.
after loss.backward() computes gradients (which direction to nudge each weight),
the optimizer reads those gradients and actually updates the model's weights.
lr=2e-5 is the standard starting learning rate for RoBERTa fine-tuning —
small enough not to destroy the pretrained representations.
"""

optimizer = AdamW(model.parameters(), lr=2e-5)
"""
the scheduler gradually adjusts the learning rate during training.
warmup phase (first 10% of steps): lr ramps from ~0 up to 2e-5.
this prevents the randomly initialized classification head from making
huge destabilizing updates before the model finds its footing.
after warmup: lr decays linearly back toward 0 by the end of training.
"""
total_steps = len(train_loader) * 5  # total batches across all epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * 0.1),
    num_training_steps=total_steps
)


# TRAINING LOOP 

def training_loop(model, train_loader, loss_fn, device, optimizer, scheduler):
    # train mode enables dropout — randomly zeroes out some neurons each forward pass
    # this prevents overfitting by forcing the model not to rely on any single neuron
    model.train()
    running_loss = 0.0

    for batch in train_loader:
        # each batch is a dict with keys: input_ids, attention_mask, labels
        # move everything to the same device as the model —
        # PyTorch will throw an error if model and data are on different devices
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # PyTorch accumulates gradients by default — they add up across batches
        # we must zero them out before each new batch or we'd be mixing gradients
        # from the current batch with leftover gradients from the last one
        # set_to_none=True is slightly faster than zeroing (deallocates instead of filling with 0)
        optimizer.zero_grad(set_to_none=True)

        # forward pass: feed the batch through the model.
        # outputs.logits is a tensor of shape (batch_size, num_labels) —
        # 16 examples × 17 scores each.
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # compute loss: how wrong were the predictions vs the true labels?
        # CrossEntropyLoss with class weights scales rare-class mistakes higher
        loss = loss_fn(outputs.logits, labels)

        # backward pass: PyTorch walks backwards through every operation in the forward pass
        # and computes gradients — how much each weight contributed to the loss,
        # and in which direction changing it would reduce the loss.
        loss.backward()

        # optimizer reads the gradients and nudges each weight by lr in the right direction
        optimizer.step()

        # scheduler updates the learning rate for the next step
        scheduler.step()

        running_loss += loss.item()

    # return average loss across all batches in this epoch
    return running_loss / len(train_loader)


# VALIDATION LOOP 

def validate(model, val_loader, device):
    # eval mode disables dropout — we want deterministic output during validation,
    # not random neuron zeroing messing with our F1 calculation.
    model.eval()
    all_preds = []
    all_labels = []

    # inference_mode disables gradient tracking entirely —
    # we don't need gradients during validation (no backward pass),
    # so this saves memory and speeds up the forward pass.
    with torch.inference_mode():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # forward pass only — no loss, no backward, no weight updates
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # argmax picks the class with the highest logit score as the prediction.
            # dim=-1 means take the argmax across the class dimension (the 17 scores),
            # not across the batch dimension.
            preds = outputs.logits.argmax(dim=-1)

            # move back to CPU and convert to numpy so sklearn can read them
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # macro-F1: compute F1 for each class separately, then average
    # this treats all classes equally regardless of size —
    # getting xenophobic right matters as much as getting racist right
    # that's the right metric when you care about rare classes
    return f1_score(all_labels, all_preds, average='macro')


# TRAINING WITH EARLY STOPPING

# early stopping: if val macro-F1 doesn't improve for PATIENCE consecutive epochs
# stop training. prevents overfitting and saves time
EPOCHS = 5
PATIENCE = 2

best_val_f1 = 0.0
epochs_no_improve = 0

os.makedirs("checkpoints", exist_ok=True)

for epoch in range(EPOCHS):
    train_loss = training_loop(model, train_loader, loss_fn, device, optimizer, scheduler)
    val_f1 = validate(model, val_loader, device)

    print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Macro-F1: {val_f1:.4f}")

    # save checkpoint whenever val F1 improves — this is the model we'll use for evaluation,
    # not necessarily the final epoch (which may have started overfitting)
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'checkpoints/best_model.pt')
        print(f"  saved best model (val F1: {val_f1:.4f})")
    else:
        epochs_no_improve += 1
        print(f"  no improvement ({epochs_no_improve}/{PATIENCE})")
        if epochs_no_improve >= PATIENCE:
            print("Early stopping.")
            break

print(f"\nDone. Best val macro-F1: {best_val_f1:.4f}")