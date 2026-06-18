import json
import os
from collections import Counter

import pandas as pd

TEST_CSV = "data/test.csv"
OUTPUT_JSON = "results/baseline_results.json"


def load_test() -> pd.DataFrame:
    df = pd.read_csv(TEST_CSV)
    df["lg3_flagged"] = df["llama_response"].str.strip() != "safe"
    df["lg3_category"] = df["llama_response"].apply(
        lambda r: r.strip().split("\n")[1] if "\n" in str(r) else None
    )
    return df


def recall(series: pd.Series) -> float:
    return round(series.mean(), 4)


def per_ingroup_recall(df: pd.DataFrame, col: str) -> dict:
    return (
        df.groupby("ingroup")[col]
        .apply(recall)
        .sort_values(ascending=False)
        .to_dict()
    )


def divergence(df: pd.DataFrame) -> dict:
    lg3 = df["lg3_flagged"]
    oai = df["flagged"].astype(bool)
    return {
        "both_flagged":     int((lg3 & oai).sum()),
        "oai_only":         int((~lg3 & oai).sum()),
        "lg3_only":         int((lg3 & ~oai).sum()),
        "neither_flagged":  int((~lg3 & ~oai).sum()),
    }


def category_breakdown(df: pd.DataFrame) -> dict:
    flagged = df[df["lg3_flagged"]]
    counts = Counter(flagged["lg3_category"].dropna())
    total = sum(counts.values())
    return {
        cat: {"count": n, "pct": round(n / total, 4)}
        for cat, n in counts.most_common()
    }


def print_table(results: dict):
    lg3 = results["per_ingroup_recall"]["lg3"]
    oai = results["per_ingroup_recall"]["oai"]
    all_ingroups = sorted(lg3.keys(), key=lambda x: -oai.get(x, 0))

    print("\n── Per-ingroup recall on test set ──────────────────────────────")
    print(f"  {'ingroup':<25} {'LG3':>8} {'OAI':>8} {'gap':>8}")
    print(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*8}")
    for ig in all_ingroups:
        l = lg3.get(ig, 0)
        o = oai.get(ig, 0)
        gap = round(o - l, 4)
        print(f"  {ig:<25} {l:>7.1%} {o:>7.1%} {gap:>+7.1%}")

    print(f"\n  {'OVERALL':<25} {results['overall_recall']['lg3']:>7.1%} "
          f"{results['overall_recall']['oai']:>7.1%} "
          f"{results['overall_recall']['oai'] - results['overall_recall']['lg3']:>+7.1%}")

    print("\n── Agreement / divergence ──────────────────────────────────────")
    div = results["divergence"]
    n = sum(div.values())
    for label, count in div.items():
        print(f"  {label:<20} {count:>5}  ({count/n:.1%})")

    print("\n── LG3 category breakdown (when it flags) ──────────────────────")
    for cat, v in results["lg3_category_breakdown"].items():
        print(f"  {cat:<12} {v['count']:>5}  ({v['pct']:.1%})")


def main():
    print(f"Loading test set from {TEST_CSV}...")
    df = load_test()
    print(f"Test set: {len(df):,} rows\n")

    results = {
        "test_size": len(df),
        "overall_recall": {
            "lg3": recall(df["lg3_flagged"]),
            "oai": recall(df["flagged"].astype(bool)),
        },
        "per_ingroup_recall": {
            "lg3": per_ingroup_recall(df, "lg3_flagged"),
            "oai": per_ingroup_recall(df, "flagged"),
        },
        "divergence": divergence(df),
        "lg3_category_breakdown": category_breakdown(df),
    }

    os.makedirs("results", exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Written → {OUTPUT_JSON}")

    print_table(results)
    print(f"\n✓ Done. This is your 'APIs fail' table.")


if __name__ == "__main__":
    main()