from transformers import pipeline
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
pipeline = pipeline(
    task="fill-mask",
    model="FacebookAI/roberta-base",
    device=0
)
pipeline("Plants create <mask> through a process known as photosynthesis.")