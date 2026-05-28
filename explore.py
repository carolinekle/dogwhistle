from datasets import load_dataset_builder
from datasets import load_dataset
import sys
from openai import OpenAI
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv
load_dotenv()
import sqlite3
import time

from db import DB_PATH, get_conn, init_db

ds_builder = load_dataset_builder("SALT-NLP/silent_signals")

client = OpenAI(max_retries=5)

Hclient = InferenceClient(
    api_key=os.environ["HF_TOKEN"],
)

def call_OAI(content):
    response = client.moderations.create(
    model="omni-moderation-latest",
    input=content

    )
    return response

def call_llama(content, retries=3):
    prompt = f"""[INST] Task: Check if there is unsafe content in the user message.

    <BEGIN USER MESSAGE>
    {content}
    <END USER MESSAGE>

    Provide your safety assessment: [/INST]"""

    for attempt in range(retries):
        try:
            completion = Hclient.chat.completions.create(
                model="meta-llama/Llama-Guard-3-8B:featherless-ai",
                messages=[{"role": "user", "content": prompt}],
            )
            return completion.choices[0].message.content
        except Exception as e:
            if attempt < retries - 1:
                print(f"Llama timeout, retrying ({attempt + 1}/{retries})...")
                time.sleep(5)
            else:
                print(f"Llama failed after {retries} attempts: {e}")
                return None

    completion = Hclient.chat.completions.create(
        model="meta-llama/Llama-Guard-3-8B:featherless-ai",
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content,

def write_row(conn: sqlite3.Connection, row: dict):
    conn.execute("""
        INSERT OR IGNORE INTO results (
            content,
            dog_whistle,
            ingroup,
            llama_response,
            openai_response,
            c_model_response,   
            flagged,
            categories,
            category_scores,
            source
        ) VALUES (
            :content,
            :dog_whistle,
            :ingroup,
            :llama_response,
            :openai_response,
            :c_model_response,  
            :flagged,
            :categories,
            :category_scores,
            :source
        )
    """, row)
    conn.commit()

def already_processed(conn, row_id):
    cur = conn.execute("SELECT 1 FROM results WHERE id = ?", (row_id,))
    return cur.fetchone() is not None

def prompter(conn):
    dataset = load_dataset("SALT-NLP/silent_signals", split="train[:10]")
    for idx, row in enumerate(dataset):
        if already_processed(conn, idx):
            print(f"Skipping row {idx}, already processed")
            continue

        o_response = call_OAI(row["content"])
        l_response = call_llama(row["content"])

        write_row(conn,{
            "id": idx,
            "content": row["content"],
            "dog_whistle": row["dog_whistle"],
            "ingroup": row["ingroup"],
            "llama_response": str(l_response),
            "openai_response": str(o_response),
            "c_model_response": None,
            "flagged": o_response.results[0].flagged,
            "categories": str(dict(o_response.results[0].categories)),
            "category_scores": str(dict(o_response.results[0].category_scores)),
            "source": row["source"]
        })


if __name__ == "__main__":
    init_db()  
    conn = get_conn()
    prompter(conn)