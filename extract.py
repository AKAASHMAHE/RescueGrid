
from pathlib import Path
import re
import pandas as pd


# --------------------------------------------------
# 1. PATH CONFIGURATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

INPUT_FILE = DATA_DIR / "humaid_reports_merged.tsv"
OUTPUT_FILE = DATA_DIR / "processed_reports.tsv"


# --------------------------------------------------
# 2. HUM AID LABEL MAPPING
# --------------------------------------------------
# These categories are derived from the existing labels.
# Original labels are preserved in the output.

LABEL_MAP = {
    "requests_or_urgent_needs": "urgent_needs",
    "injured_or_dead_people": "casualties",
    "displaced_people_and_evacuations": "evacuation",
    "infrastructure_and_utility_damage": "infrastructure_damage",
    "rescue_volunteering_or_donation_effort": "rescue_support",
    "caution_and_advice": "safety_advice",
    "sympathy_and_support": "emotional_support",
    "other_relevant_information": "general_information",
    "missing_or_found_people": "missing_people",
    "not_humanitarian": "non_humanitarian",
}


# These are rule-based indicators, not verified urgency judgments.
URGENCY_HINTS = {
    "urgent_needs": "HIGH_REVIEW",
    "casualties": "HIGH_REVIEW",
    "missing_people": "HIGH_REVIEW",
    "evacuation": "HIGH_REVIEW",
    "infrastructure_damage": "REVIEW",
    "rescue_support": "REVIEW",
    "safety_advice": "REVIEW",
    "emotional_support": "NORMAL_REVIEW",
    "general_information": "NORMAL_REVIEW",
    "non_humanitarian": "EXCLUDE_CANDIDATE",
    "unmapped": "MANUAL_REVIEW",
}


# --------------------------------------------------
# 3. TEXT CLEANING
# --------------------------------------------------

def clean_text(text):
    """Clean formatting without changing the original text column."""

    text = str(text or "")
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_text(text):
    """Create normalized text for later matching and grouping."""

    text = clean_text(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^\w\s#']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# --------------------------------------------------
# 4. CATEGORY AND KEYWORD EXTRACTION
# --------------------------------------------------

def map_category(label):
    label = clean_text(label).lower()
    return LABEL_MAP.get(label, "unmapped")


def find_need_indicators(text):
    """
    Find simple keyword indicators.
    These are clues for review, not confirmed resource requests.
    """

    text = normalize_text(text)
    indicators = []

    patterns = {
        "medical": [
            r"\bmedical\b",
            r"\bdoctor\b",
            r"\bambulance\b",
            r"\binjured\b",
            r"\bmedicine\b",
        ],
        "food_water": [
            r"\bfood\b",
            r"\bwater\b",
            r"\bdrinking water\b",
            r"\bfood supplies\b",
        ],
        "shelter": [
            r"\bshelter\b",
            r"\bhomeless\b",
            r"\btemporary housing\b",
        ],
        "rescue": [
            r"\brescue\b",
            r"\btrapped\b",
            r"\bstranded\b",
            r"\b救助\b",
        ],
        "evacuation": [
            r"\bevacuate\b",
            r"\bevacuation\b",
            r"\bevacuated\b",
        ],
        "infrastructure": [
            r"\bpower outage\b",
            r"\bblackout\b",
            r"\bbridge collapsed\b",
            r"\broad blocked\b",
            r"\bflooded road\b",
        ],
    }

    for category, keywords in patterns.items():
        if any(re.search(pattern, text) for pattern in keywords):
            indicators.append(category)

    return ", ".join(indicators)


# --------------------------------------------------
# 5. MAIN PROCESSING PIPELINE
# --------------------------------------------------

def main():
    print("=" * 55)
    print("RESCUEGRID - MODULE 1: REPORT EXTRACTION")
    print("=" * 55)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}\n"
            "Run prepare_humaid.py first."
        )

    df = pd.read_csv(
        INPUT_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )

    required_columns = ["tweet_id", "tweet_text", "original_label"]
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    print(f"Input rows: {len(df):,}")

    # Keep a separate cleaned text column.
    df["clean_text"] = df["tweet_text"].apply(clean_text)
    df["normalized_text"] = df["clean_text"].apply(normalize_text)

    # Map original dataset labels to project categories.
    df["humanitarian_category"] = df["original_label"].apply(
        map_category
    )

    # Attach transparent, rule-based review hints.
    df["review_hint"] = df["humanitarian_category"].map(
        URGENCY_HINTS
    ).fillna("MANUAL_REVIEW")

    # Extract possible resource/need indicators from the text.
    df["need_indicators"] = df["clean_text"].apply(
        find_need_indicators
    )

    # Mark which records have usable text.
    df["text_status"] = df["clean_text"].apply(
        lambda text: "available" if text else "empty"
    )

    # Save processed records.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        OUTPUT_FILE,
        sep="\t",
        index=False,
        encoding="utf-8",
    )

    # --------------------------------------------------
    # 6. BASIC PROCESSING REPORT
    # --------------------------------------------------

    print("\nCATEGORY COUNTS")
    print(
        df["humanitarian_category"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nREVIEW HINT COUNTS")
    print(
        df["review_hint"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nTEXT STATUS")
    print(df["text_status"].value_counts().to_string())

    unmapped = (
        df["humanitarian_category"] == "unmapped"
    ).sum()

    print(f"\nUnmapped labels: {unmapped:,}")
    print(f"Output rows: {len(df):,}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("\nModule 1 completed successfully.")


if __name__ == "__main__":
    main()
