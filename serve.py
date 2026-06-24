from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, RobertaTokenizer
import torch
import torch.nn.functional as F
import json

# global state — loaded once at startup, shared across all requests
model = None
tokenizer = None
label_map = None
device = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, label_map, device

    with open('data/label_map.json') as f:
        raw = json.load(f)
    label_map = {int(k): v for k, v in raw.items()}
    num_labels = len(label_map)

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        'roberta-base',
        num_labels=num_labels
    )
    model.load_state_dict(torch.load('checkpoints/best_model.pt', map_location='cpu'))
    model = model.to(device)
    model.eval()

    print(f"Model loaded on {device}. Ready.")
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    text: str


@app.post("/predict")
def predict(request: PredictRequest):
    inputs = tokenizer(
        request.text,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # softmax converts raw logits into probabilities that sum to 1
    probs = F.softmax(outputs.logits, dim=-1)[0]

    # top 3 predictions
    top3_vals, top3_idx = torch.topk(probs, k=3)
    top3 = [
        {"ingroup": label_map[idx.item()], "confidence": round(val.item(), 4)}
        for val, idx in zip(top3_vals, top3_idx)
    ]

    return {
        "top3": top3,
        "confident": top3[0]["confidence"] >= 0.30,
    }


# serve static/ at root — visit localhost:8000 in browser
app.mount("/", StaticFiles(directory="static", html=True), name="static")