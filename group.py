
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

INPUT_FILE = DATA_DIR / "processed_reports.tsv"
GROUPS_FILE = DATA_DIR / "report_groups.tsv"
PAIRS_FILE = DATA_DIR / "similar_report_pairs.tsv"

# Tune this after inspecting actual examples.
SIMILARITY_THRESHOLD = 0.65

# Limit each report to its nearest candidates.
NEIGHBORS_PER_REPORT = 6


# --------------------------------------------------
# 2. GROUP REPORTS BY DISASTER EVENT
# --------------------------------------------------

def find_similar_pairs(event_df):
    """
    Find similar tweet pairs inside one event.

    Similarity is a candidate signal, not proof that two
    tweets describe the same incident.
    """

    texts = event_df["normalized_text"].fillna("").astype(str)
    usable = texts.str.len() > 0

    event_df = event_df.loc[usable].copy()
    texts = texts.loc[usable]

    if len(event_df) < 2:
        return []

    # Character n-grams help match wording with small variations.
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        max_features=100_000,
        sublinear_tf=True,
        dtype=float,
    )

    matrix = vectorizer.fit_transform(texts)

    # Sparse nearest-neighbor search avoids creating a full
    # all-pairs similarity matrix.
    k = min(NEIGHBORS_PER_REPORT + 1, len(event_df))

    search = NearestNeighbors(
        n_neighbors=k,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    search.fit(matrix)

    distances, indices = search.kneighbors(matrix)

    ids = event_df["tweet_id"].astype(str).tolist()
    labels = event_df["humanitarian_category"].astype(str).tolist()
    positions = event_df.index.tolist()

    pairs = {}

    for row_pos, (row_distances, row_indices) in enumerate(
        zip(distances, indices)
    ):
        for distance, neighbor_pos in zip(
            row_distances, row_indices
        ):
            if row_pos == neighbor_pos:
                continue

            similarity = 1.0 - float(distance)

            if similarity < SIMILARITY_THRESHOLD:
                continue

            id_a = ids[row_pos]
            id_b = ids[neighbor_pos]

            # Store each pair only once.
            pair_key = tuple(sorted((id_a, id_b)))

            if pair_key in pairs:
                continue

            pairs[pair_key] = {
                "tweet_id_a": id_a,
                "tweet_id_b": id_b,
                "row_index_a": positions[row_pos],
                "row_index_b": positions[neighbor_pos],
                "similarity": round(similarity, 4),
                "category_a": labels[row_pos],
                "category_b": labels[neighbor_pos],
                "same_category": labels[row_pos] == labels[neighbor_pos],
                "candidate_status": "REVIEW_SIMILARITY",
            }

    return list(pairs.values())


# --------------------------------------------------
# 3. MAIN PIPELINE
# --------------------------------------------------

def main():
    print("=" * 55)
    print("RESCUEGRID - MODULE 2: SIMILAR REPORT GROUPING")
    print("=" * 55)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing input: {INPUT_FILE}\n"
            "Run python extract.py first."
        )

    df = pd.read_csv(
        INPUT_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )

    required = [
        "tweet_id",
        "normalized_text",
        "humanitarian_category",
        "event_name",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Keep original rows and add candidate-group identifiers.
    df["candidate_group_id"] = ""
    all_pairs = []

    # An event name is a scope for comparison, not a confirmed
    # incident identifier.
    if (df["event_name"] == "").any():
        print("Warning: some rows have no event_name.")

    valid_events = df[df["event_name"] != ""]

    for event_name, event_df in valid_events.groupby("event_name"):
        print(f"Processing {event_name}: {len(event_df):,} reports")

        event_pairs = find_similar_pairs(event_df)
        all_pairs.extend(event_pairs)

    pairs_df = pd.DataFrame(
        all_pairs,
        columns=[
            "tweet_id_a",
            "tweet_id_b",
            "row_index_a",
            "row_index_b",
            "similarity",
            "category_a",
            "category_b",
            "same_category",
            "candidate_status",
        ],
    )

    # Give every report its own ID initially. Candidate pair
    # edges will later be used to construct connected components.
    # Do not merge them yet: similarity needs validation first.
    df["candidate_group_id"] = [
        f"R{i + 1:06d}" for i in range(len(df))
    ]

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(GROUPS_FILE, sep="\t", index=False, encoding="utf-8")
    pairs_df.to_csv(PAIRS_FILE, sep="\t", index=False, encoding="utf-8")

    print("\nSUMMARY")
    print(f"Reports processed: {len(df):,}")
    print(f"Similar candidate pairs: {len(pairs_df):,}")
    print(f"Similarity threshold: {SIMILARITY_THRESHOLD}")
    print(f"Report file: {GROUPS_FILE}")
    print(f"Candidate-pair file: {PAIRS_FILE}")
    print("\nModule 2 completed.")


if __name__ == "__main__":
    main()
