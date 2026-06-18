from datasets import load_dataset_builder
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
import os
from dotenv import load_dotenv
load_dotenv()
import sqlite3
import time

from db import DB_PATH, get_conn, init_db

ds_builder = load_dataset_builder("SALT-NLP/silent_signals")

client = OpenAI(max_retries=5)

llama_client = OpenAI(
    base_url="https://api.featherless.ai/v1",
    api_key=os.environ["FEATHERLESS_API_KEY"],
    max_retries=0,  # retries handled manually below
)


def call_OAI(content, retries=3):
    for attempt in range(retries):
        try:
            response = client.moderations.create(
                model="omni-moderation-latest",
                input=content
            )
            return response
        except Exception as e:
            if attempt < retries - 1:
                delay = 5 * (2 ** attempt)
                print(f"OAI error, retrying in {delay}s ({attempt + 1}/{retries}): {e}")
                time.sleep(delay)
            else:
                print(f"OAI failed after {retries} attempts: {e}")
                return None


def call_llama(content, retries=5):
    for attempt in range(retries):
        try:
            completion = llama_client.chat.completions.create(
                model="meta-llama/Llama-Guard-3-8B",
                messages=[{"role": "user", "content": content}],
                max_tokens=20,
            )
            return completion.choices[0].message.content
        except Exception as e:
            if attempt < retries - 1:
                delay = 5 * (2 ** attempt)
                print(f"Llama error, retrying in {delay}s ({attempt + 1}/{retries}): {e}")
                time.sleep(delay)
            else:
                print(f"Llama failed after {retries} attempts: {e}")
                return None


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


def already_processed(conn, content):
    cur = conn.execute("SELECT 1 FROM results WHERE content = ?", (content,))
    return cur.fetchone() is not None


def prompter(conn):
    dataset = load_dataset("SALT-NLP/silent_signals", split="train")

    for row in tqdm(dataset, desc="Processing"):
        if already_processed(conn, row["content"]):
            continue

        o_response = call_OAI(row["content"])
        time.sleep(0.1)

        if o_response is None:
            tqdm.write(f"OAI failed — skipping: {row['content'][:40]}")
            continue

        l_response = call_llama(row["content"])

        if l_response is None:
            tqdm.write(f"Llama failed — skipping: {row['content'][:40]}")
            continue

        write_row(conn, {
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