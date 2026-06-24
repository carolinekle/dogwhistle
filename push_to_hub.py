from transformers import AutoModelForSequenceClassification, RobertaTokenizer
import torch
import json

# load label map to get num_labels
with open('data/label_map.json') as f:
    label_map = json.load(f)
num_labels = len(label_map)

# load model and restore weights
model = AutoModelForSequenceClassification.from_pretrained(
    'roberta-base',
    num_labels=num_labels
)
model.load_state_dict(torch.load('checkpoints/best_model.pt', map_location='cpu'))

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# push both to hub
model.push_to_hub("carolineklew/dogwhistle-roberta")
tokenizer.push_to_hub("carolineklew/dogwhistle-roberta")

print("done — https://huggingface.co/carolineklew/dogwhistle-roberta")