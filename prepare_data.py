import json
import os
import sqlite3

import pandas as pd
from sklearn.model_selection import train_test_split

DB_PATH = "data/whistle_results.db"
MIN_CLASS_SIZE = 50
RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT content, dog_whistle, ingroup, llama_response, flagged, source
        FROM results
        """,
        conn,
    )
    conn.close()
    return df


def handle_small_classes(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["ingroup"].value_counts()
    small = counts[counts < MIN_CLASS_SIZE].index.tolist()
    if small:
        print(f"Merging into 'other' (< {MIN_CLASS_SIZE} rows): {small}")
        df["ingroup"] = df["ingroup"].apply(
            lambda x: "other" if x in small else x
        )
    else:
        print("No small classes — all ingroups above threshold.")
    return df


def make_splits(df: pd.DataFrame):
    train, temp = train_test_split(
        df, test_size=0.30, stratify=df["ingroup"], random_state=RANDOM_STATE
    )
    val, test = train_test_split(
        temp, test_size=0.50, stratify=temp["ingroup"], random_state=RANDOM_STATE
    )
    return train, val, test


def write_csvs(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    os.makedirs("data", exist_ok=True)
    train.to_csv("data/train.csv", index=False)
    val.to_csv("data/val.csv", index=False)
    test.to_csv("data/test.csv", index=False)
    print(f"  train: {len(train):,} rows → data/train.csv")
    print(f"  val:   {len(val):,} rows → data/val.csv")
    print(f"  test:  {len(test):,} rows → data/test.csv")


def write_split_to_db(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
):
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("ALTER TABLE results ADD COLUMN split TEXT")
    except sqlite3.OperationalError:
        pass  # column already exists

    for split_name, df in [("train", train), ("val", val), ("test", test)]:
        conn.executemany(
            "UPDATE results SET split = ? WHERE content = ?",
            [(split_name, c) for c in df["content"]],
        )
    conn.commit()
    conn.close()
    print("  split column written back to DB")


def compute_eda_stats(
    df: pd.DataFrame,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> dict:
    word_lengths = df["content"].str.split().str.len()

    stats = {
        "total_rows": len(df),
        "split_sizes": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
        "class_distribution": (
            df["ingroup"].value_counts().to_dict()
        ),
        "train_class_distribution": (
            train["ingroup"].value_counts().to_dict()
        ),
        "approx_word_lengths": {
            "mean": round(word_lengths.mean(), 1),
            "median": round(word_lengths.median(), 1),
            "p95": round(word_lengths.quantile(0.95), 1),
            "max": int(word_lengths.max()),
        },
        "source_breakdown": df["source"].value_counts().to_dict(),
        "baseline_stats": {
            "lg3_safe_rate_overall": round(
                (df["llama_response"] == "safe").mean(), 4
            ),
            "oai_flagged_rate_overall": round(df["flagged"].mean(), 4),
            "lg3_safe_by_ingroup": (
                df.groupby("ingroup")
                .apply(
                    lambda g: round(
                        (g["llama_response"] == "safe").mean(), 4
                    )
                )
                .to_dict()
            ),
            "oai_flagged_by_ingroup": (
                df.groupby("ingroup")["flagged"]
                .mean()
                .round(4)
                .to_dict()
            ),
        },
    }

    os.makedirs("results", exist_ok=True)
    with open("results/eda_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def print_summary(stats: dict):
    print("\n── Class distribution ──────────────────────")
    for ingroup, n in sorted(
        stats["class_distribution"].items(), key=lambda x: -x[1]
    ):
        lg3_catch = round(
            1 - stats["baseline_stats"]["lg3_safe_by_ingroup"].get(ingroup, 0), 3
        )
        oai_hit = stats["baseline_stats"]["oai_flagged_by_ingroup"].get(
            ingroup, 0
        )
        print(
            f"  {ingroup:<25} n={n:>5}   "
            f"LG3 catch={lg3_catch:.1%}   OAI flag={oai_hit:.1%}"
        )

    print("\n── Word length (approx tokens) ─────────────")
    wl = stats["approx_word_lengths"]
    print(
        f"  mean={wl['mean']}  median={wl['median']}  "
        f"p95={wl['p95']}  max={wl['max']}"
    )
    print("  Note: RoBERTa hard limit is 512 subword tokens.")
    print("        Run tokenizer on train set before training to check truncation rate.")

    print("\n── Source breakdown ────────────────────────")
    for src, n in sorted(
        stats["source_breakdown"].items(), key=lambda x: -x[1]
    ):
        print(f"  {src:<30} {n:>5}")

    print("\n── Baseline summary ────────────────────────")
    print(
        f"  LG3 safe rate overall:    {stats['baseline_stats']['lg3_safe_rate_overall']:.1%}"
    )
    print(
        f"  OAI flagged rate overall: {stats['baseline_stats']['oai_flagged_rate_overall']:.1%}"
    )


def main():
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df):,} rows\n")

    print("Checking ingroup distribution...")
    print(df["ingroup"].value_counts().to_string())
    print()

    print("Handling small classes...")
    df = handle_small_classes(df)

    print("\nMaking stratified 70/15/15 splits...")
    train, val, test = make_splits(df)
    write_csvs(train, val, test)
    write_split_to_db(train, val, test)

    print("\nComputing EDA stats...")
    stats = compute_eda_stats(df, train, val, test)
    print("  written → results/eda_stats.json")

    print_summary(stats)

    print("\n✓ Done. Test set is locked — don't look at it until after training.")


if __name__ == "__main__":
    main()