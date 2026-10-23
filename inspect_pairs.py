
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

REPORTS_FILE = DATA_DIR / "processed_reports.tsv"
PAIRS_FILE = DATA_DIR / "similar_report_pairs.tsv"
OUTPUT_FILE = DATA_DIR / "review_similar_pairs.tsv"

reports = pd.read_csv(
    REPORTS_FILE, sep="\t", dtype=str, keep_default_na=False
)
pairs = pd.read_csv(
    PAIRS_FILE, sep="\t", dtype=str, keep_default_na=False
)

# Map tweet IDs to their original report details.
reports = reports.drop_duplicates("tweet_id").set_index("tweet_id")

if pairs.empty:
    print("No candidate pairs found. Nothing to inspect.")
else:
    rows = []

    for _, pair in pairs.iterrows():
        id_a = pair["tweet_id_a"]
        id_b = pair["tweet_id_b"]

        if id_a not in reports.index or id_b not in reports.index:
            continue

        a = reports.loc[id_a]
        b = reports.loc[id_b]

        rows.append({
            "tweet_id_a": id_a,
            "tweet_id_b": id_b,
            "event_a": a.get("event_name", ""),
            "event_b": b.get("event_name", ""),
            "similarity": pair["similarity"],
            "category_a": a.get("humanitarian_category", ""),
            "category_b": b.get("humanitarian_category", ""),
            "tweet_text_a": a.get("tweet_text", ""),
            "tweet_text_b": b.get("tweet_text", ""),
            "manual_review": "",
        })

    review = pd.DataFrame(rows)
    review["similarity_numeric"] = pd.to_numeric(
        review["similarity"], errors="coerce"
    )

    review = review.sort_values(
        "similarity_numeric", ascending=False
    ).drop(columns=["similarity_numeric"])

    review.to_csv(OUTPUT_FILE, sep="\t", index=False, encoding="utf-8")

    print(f"Candidate pairs: {len(review):,}")
    print(f"Saved review file: {OUTPUT_FILE}")
    print("\nTop 10 candidate pairs:\n")

    for _, row in review.head(10).iterrows():
        print("-" * 70)
        print("Similarity:", row["similarity"])
        print("Event:", row["event_a"])
        print("Category A:", row["category_a"])
        print("Tweet A:", row["tweet_text_a"])
        print("Category B:", row["category_b"])
        print("Tweet B:", row["tweet_text_b"])
