
"""
RESCUEGRID - Emergency Report Priority Scoring
==============================================

Reads:
    data/processed_reports.tsv

Writes:
    data/priority_reports.tsv

Purpose:
    Assign transparent, rule-based review priorities to disaster
    reports using humanitarian categories and urgency indicators.

Important:
    This is a historical dataset prototype, not a live emergency
    dispatch system. Scores indicate review priority, not verified
    severity, authenticity, or the need for actual resource dispatch.
"""

from pathlib import Path
import re
import sys

import pandas as pd


# =========================================================
# 1. CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

INPUT_FILE = DATA_DIR / "processed_reports.tsv"
OUTPUT_FILE = DATA_DIR / "priority_reports.tsv"


# Category-based starting scores.
# These are configurable prototype heuristics, not calibrated
# probabilities or validated estimates of real-world severity.

CATEGORY_BASE_SCORES = {
    "urgent_needs": 45,
    "casualties": 40,
    "missing_people": 45,
    "evacuation": 35,
    "infrastructure_damage": 25,
    "rescue_support": 20,
    "safety_advice": 20,
    "general_information": 10,
    "emotional_support": 5,
    "non_humanitarian": 0,
}

CATEGORY_REASONS = {
    "urgent_needs": "Humanitarian urgent-needs category",
    "casualties": "Casualties category",
    "missing_people": "Missing-persons category",
    "evacuation": "Evacuation category",
    "infrastructure_damage": "Infrastructure-damage category",
    "rescue_support": "Rescue/support category",
    "safety_advice": "Safety-advice category",
    "general_information": "General-information category",
    "emotional_support": "Emotional-support category",
    "non_humanitarian": "Non-humanitarian category",
}


# Additional indicator weights. Only the strongest match in each
# group is applied, avoiding repeated counting of similar phrases.

INDICATOR_GROUPS = {
    "immediate_danger": {
        "weight": 35,
        "patterns": [
            r"\btrapped\b",
            r"\bpeople stranded\b",
            r"\bstranded people\b",
            r"\bneed rescue immediately\b",
            r"\bimmediate rescue\b",
            r"\blife[- ]threatening\b",
            r"\bin immediate danger\b",
            r"\bcritical emergency\b",
        ],
        "reason": "Immediate-danger language",
    },

    "urgent_request": {
        "weight": 25,
        "patterns": [
            r"\bneed(s|ed)? (?:urgent )?(?:help|rescue|assistance)\b",
            r"\burgent (?:help|rescue|assistance|aid)\b",
            r"\bsend help\b",
            r"\brequest(?:ing)? rescue\b",
            r"\bplease help\b",
            r"\bemergency supplies needed\b",
            r"\bneed food and water\b",
            r"\bneed medical assistance\b",
        ],
        "reason": "Explicit urgent request",
    },

    "medical_emergency": {
        "weight": 25,
        "patterns": [
            r"\bseverely injured\b",
            r"\bcritically injured\b",
            r"\bmedical emergency\b",
            r"\burgent medical care\b",
            r"\bneed(?:s)? an ambulance\b",
            r"\bambulance needed\b",
            r"\bpeople bleeding\b",
        ],
        "reason": "Medical-emergency language",
    },

    "missing_persons": {
        "weight": 20,
        "patterns": [
            r"\bpeople missing\b",
            r"\bperson missing\b",
            r"\bchild missing\b",
            r"\bchildren missing\b",
            r"\bmissing persons?\b",
            r"\bpeople unaccounted for\b",
        ],
        "reason": "Missing-person language",
    },

    "evacuation_need": {
        "weight": 15,
        "patterns": [
            r"\bneed to evacuate\b",
            r"\bevacuate immediately\b",
            r"\bevacuation needed\b",
            r"\bpeople need evacuation\b",
            r"\btrapped by flood\b",
            r"\bfloodwaters rising\b",
            r"\bwater level rising\b",
        ],
        "reason": "Possible evacuation need",
    },

    "essential_supplies": {
        "weight": 10,
        "patterns": [
            r"\bneed drinking water\b",
            r"\bfood supplies needed\b",
            r"\bno drinking water\b",
            r"\bno food supplies\b",
            r"\bshelter needed\b",
            r"\bmedicine needed\b",
            r"\bshortage of food\b",
        ],
        "reason": "Essential-supplies need",
    },

    "infrastructure_disruption": {
        "weight": 8,
        "patterns": [
            r"\bbridge collapsed\b",
            r"\broad washed away\b",
            r"\bpower outage\b",
            r"\bwater supply disrupted\b",
            r"\bhospital damaged\b",
            r"\bbuilding collapsed\b",
            r"\bcommunications? down\b",
        ],
        "reason": "Infrastructure-disruption language",
    },
}


# Review priority bands. Scores are capped at 100.

PRIORITY_LEVELS = [
    (70, "CRITICAL_REVIEW"),
    (45, "HIGH_REVIEW"),
    (20, "MEDIUM_REVIEW"),
    (1, "LOW_REVIEW"),
    (0, "INFORMATIONAL"),
]


# =========================================================
# 2. HELPERS
# =========================================================

def normalize_text(value):
    """Normalize report text for keyword matching."""
    if pd.isna(value):
        return ""

    text = str(value).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def get_category(row):
    """Return the standardized category when available."""
    category = str(
        row.get("humanitarian_category", "")
    ).strip().lower()

    if not category:
        category = str(row.get("original_label", "")).strip().lower()

    return category or "unknown"


def find_indicator_matches(text, patterns):
    """Return matched phrases for an indicator group."""
    matches = []

    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            matches.append(match.group(0))

    return matches


def get_priority_level(score):
    """Convert a numeric score into a review priority."""
    for threshold, level in PRIORITY_LEVELS:
        if score >= threshold:
            return level

    return "INFORMATIONAL"


# =========================================================
# 3. PRIORITY SCORING
# =========================================================

def score_report(row):
    """
    Calculate an explainable review score.

    Score =
        category starting score
        + strongest match from each urgency-indicator group

    The score is capped at 100. It is not a probability.
    """
    raw_text = row.get("tweet_text", "")
    text = normalize_text(raw_text)
    category = get_category(row)

    reasons = []
    matched_indicators = []
    score = CATEGORY_BASE_SCORES.get(category, 10)

    if category in CATEGORY_REASONS:
        reasons.append(CATEGORY_REASONS[category])
    else:
        reasons.append("Unknown category; default starting score used")

    # Exclude clearly non-humanitarian reports from urgency
    # escalation based solely on ordinary keyword matches.
    # Their category can still be reviewed separately.
    if category == "non_humanitarian":
        score = 0
    else:
        for group_name, config in INDICATOR_GROUPS.items():
            matches = find_indicator_matches(
                text,
                config["patterns"],
            )

            if matches:
                score += config["weight"]
                reasons.append(config["reason"])

                matched_indicators.append({
                    "indicator": group_name,
                    "weight": config["weight"],
                    "matched_phrases": "; ".join(
                        sorted(set(matches))
                    ),
                })

    score = min(100, max(0, int(score)))
    level = get_priority_level(score)

    return {
        "priority_score": score,
        "priority_level": level,
        "priority_reasons": " | ".join(reasons),
        "matched_indicators": matched_indicators,
        "matched_indicator_count": len(matched_indicators),
    }


# =========================================================
# 4. MAIN PIPELINE
# =========================================================

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Run extract.py first."
        )

    reports = pd.read_csv(
        INPUT_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    required_columns = {"tweet_id", "tweet_text"}

    missing_columns = required_columns - set(reports.columns)

    if missing_columns:
        raise ValueError(
            "processed_reports.tsv is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    print(f"Input reports: {len(reports):,}")

    scored_rows = []

    for _, row in reports.iterrows():
        result = score_report(row)

        output_row = row.to_dict()

        # Store only a readable summary of indicator matches.
        indicator_details = result.pop("matched_indicators")

        output_row.update(result)

        output_row["indicator_details"] = " | ".join(
            (
                f"{item['indicator']} (+{item['weight']}): "
                f"{item['matched_phrases']}"
            )
            for item in indicator_details
        )

        output_row["priority_method"] = "RULE_BASED"
        output_row["priority_is_verified_severity"] = "NO"

        scored_rows.append(output_row)

    output = pd.DataFrame(scored_rows)

    # Sort by priority score, keeping higher scores at the top.
    # Original row order breaks ties consistently.
    output["_original_order"] = range(len(output))

    output = output.sort_values(
        by=["priority_score", "_original_order"],
        ascending=[False, True],
    )

    output = output.drop(columns=["_original_order"])

    output.to_csv(
        OUTPUT_FILE,
        sep="\t",
        index=False,
        encoding="utf-8",
    )

    print("\nPriority scoring complete.")
    print(f"Reports scored: {len(output):,}")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\nPriority-level distribution:")

    distribution = (
        output["priority_level"]
        .value_counts()
        .reindex(
            [
                "CRITICAL_REVIEW",
                "HIGH_REVIEW",
                "MEDIUM_REVIEW",
                "LOW_REVIEW",
                "INFORMATIONAL",
            ],
            fill_value=0,
        )
    )

    for level, count in distribution.items():
        print(f"  {level}: {count:,}")

    print("\nTop 5 reports for review:\n")

    preview_columns = [
        "tweet_id",
        "event_name",
        "humanitarian_category",
        "priority_score",
        "priority_level",
        "priority_reasons",
        "tweet_text",
    ]

    available_columns = [
        column for column in preview_columns
        if column in output.columns
    ]

    for _, row in output[available_columns].head(5).iterrows():
        print("-" * 70)
        print(f"Tweet ID: {row.get('tweet_id', '')}")
        print(f"Event: {row.get('event_name', 'unknown')}")
        print(
            "Category: "
            f"{row.get('humanitarian_category', 'unknown')}"
        )
        print(f"Score: {row.get('priority_score', 0)}")
        print(f"Level: {row.get('priority_level', '')}")
        print(f"Reasons: {row.get('priority_reasons', '')}")
        print(f"Text: {row.get('tweet_text', '')}")

    print(
        "\nReminder: scores indicate review priority, not verified "
        "emergency severity or a dispatch recommendation."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
