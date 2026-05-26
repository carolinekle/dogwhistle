from datasets import load_dataset_builder
from datasets import load_dataset
import sys
from openai import OpenAI

ds_builder = load_dataset_builder("SALT-NLP/silent_signals")

print("Description:", ds_builder.info.description)
print("Features:", ds_builder.info.features)
print("Splits:", ds_builder.info.splits)
print("Size:", ds_builder.info.dataset_size)

client = OpenAI()


dataset = load_dataset("SALT-NLP/silent_signals", split="train[:10]")
for row in dataset:
    input_content = row["content"]
    response = client.moderations.create(
    model="",
    input=input_content
)
    