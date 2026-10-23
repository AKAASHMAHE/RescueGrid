
"""
RESCUEGRID - Evidence-Aware Conflict Review
===========================================

Purpose:
1. Compare similar reports from the same disaster event.
2. Detect selected, explicit opposing claims.
3. Extract humanitarian counts using metric-specific patterns.
4. Distinguish approximate/lower-bound counts from exact counts.
5. Avoid treating geographic breakdowns as overall totals.
6. Save explainable review signals for human verification.

Important:
- A discrepancy is not proof of misinformation.
- Different reports may describe different times or locations.
- This is a rule-based prototype, not an automated fact-checker.
- Every flagged pair remains unverified.
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

REPORTS_FILE = DATA_DIR / "processed_reports.tsv"
PAIRS_FILE = DATA_DIR / "similar_report_pairs.tsv"
OUTPUT_FILE = DATA_DIR / "potential_conflicts.tsv"

MIN_SIMILARITY = 0.65

# Approximate counts within this relative difference are treated
# as potentially consistent. This is a heuristic, not a fact.
APPROXIMATE_TOLERANCE = 0.05

# If a report lists several number-location pairs without stating
# a clear total, don't infer an overall casualty count from it.
GEOGRAPHIC_BREAKDOWN_PATTERN = re.compile(
    r"\b\d[\d,]*(?:\.\d+)?\s*[-–]\s*"
    r"(?:[a-z][a-z .'-]{1,35})"
    r"(?=\s+\d[\d,]*(?:\.\d+)?\s*[-–]|\s*[,;]|$)",
    re.IGNORECASE,
)


# =========================================================
# 2. GENERAL HELPERS
# =========================================================

def clean_text(value):
    """Normalize text while preserving words and numeric values."""
    if pd.isna(value):
        return ""

    text = str(value).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def get_value(row, columns, default=""):
    """Return the first non-empty value among candidate columns."""
    for column in columns:
        if column in row.index and pd.notna(row[column]):
            value = str(row[column]).strip()
            if value:
                return value

    return default


def get_report(report_lookup, tweet_id):
    """Find a report using its tweet ID."""
    if tweet_id not in report_lookup.index:
        return None

    result = report_lookup.loc[tweet_id]

    if isinstance(result, pd.DataFrame):
        result = result.iloc[0]

    return result


def find_patterns(text, patterns):
    """Return patterns that match the supplied text."""
    return [
        pattern
        for pattern in patterns
        if re.search(pattern, text, flags=re.IGNORECASE)
    ]


# =========================================================
# 3. OPPOSING CLAIM DETECTION
# =========================================================

# These are intentionally limited phrase rules. They can identify
# selected opposing claims but cannot understand every sentence.

OPPOSING_CLAIMS = {
    "ROAD_STATUS": {
        "positive": [
            r"\broad is open\b",
            r"\broads are open\b",
            r"\broad reopened\b",
            r"\broad has reopened\b",
            r"\broad remains open\b",
            r"\broad accessible\b",
            r"\broad is accessible\b",
        ],
        "negative": [
            r"\broad is closed\b",
            r"\broads are closed\b",
            r"\broad closed\b",
            r"\broad remains closed\b",
            r"\broad inaccessible\b",
            r"\broad is inaccessible\b",
        ],
    },

    "POWER_STATUS": {
        "positive": [
            r"\bpower restored\b",
            r"\belectricity restored\b",
            r"\bpower is back\b",
            r"\belectricity is back\b",
            r"\bpower available\b",
        ],
        "negative": [
            r"\bno power\b",
            r"\bpower outage\b",
            r"\bwithout electricity\b",
            r"\bpower unavailable\b",
            r"\belectricity unavailable\b",
        ],
    },

    "BUILDING_STATUS": {
        "positive": [
            r"\bbuilding is safe\b",
            r"\bbuilding declared safe\b",
            r"\bbuilding is intact\b",
            r"\bbuilding undamaged\b",
        ],
        "negative": [
            r"\bbuilding is unsafe\b",
            r"\bbuilding declared unsafe\b",
            r"\bbuilding collapsed\b",
            r"\bbuilding destroyed\b",
            r"\bbuilding severely damaged\b",
        ],
    },

    "WATER_STATUS": {
        "positive": [
            r"\bwater supply restored\b",
            r"\bwater service restored\b",
            r"\bwater is available\b",
            r"\bclean water available\b",
        ],
        "negative": [
            r"\bno water supply\b",
            r"\bwater supply unavailable\b",
            r"\bwater service disrupted\b",
            r"\bwater unavailable\b",
            r"\bwithout clean water\b",
        ],
    },

    "EVACUATION_STATUS": {
        "positive": [
            r"\bevacuations? ended\b",
            r"\bevacuation order lifted\b",
            r"\bresidents allowed to return\b",
            r"\bresidents can return\b",
        ],
        "negative": [
            r"\bevacuation order remains in effect\b",
            r"\bevacuation ordered\b",
            r"\bresidents must evacuate\b",
            r"\bevacuations? still underway\b",
        ],
    },

    "RESCUE_STATUS": {
        "positive": [
            r"\ball people rescued\b",
            r"\beveryone rescued\b",
            r"\ball survivors rescued\b",
            r"\ball people accounted for\b",
        ],
        "negative": [
            r"\bpeople still trapped\b",
            r"\bsurvivors remain trapped\b",
            r"\bpeople remain missing\b",
            r"\bpeople still missing\b",
        ],
    },
}


def detect_opposing_claims(text_a, text_b):
    """Flag opposite phrase patterns appearing across two reports."""
    signals = []

    for claim_name, sides in OPPOSING_CLAIMS.items():
        a_positive = find_patterns(text_a, sides["positive"])
        a_negative = find_patterns(text_a, sides["negative"])
        b_positive = find_patterns(text_b, sides["positive"])
        b_negative = find_patterns(text_b, sides["negative"])

        if (
            (a_positive and b_negative)
            or (a_negative and b_positive)
        ):
            signals.append(f"POSSIBLE_OPPOSING_{claim_name}")

    return signals


# =========================================================
# 4. METRIC-SPECIFIC NUMBER EXTRACTION
# =========================================================

# Supported examples:
# 60 dead
# 60+ dead
# at least 60 dead
# over 2,500 injured
# 13k people rescued
# death toll reaches 61
# death toll rises to 25
#
# The patterns require a number to be directly associated with
# a metric phrase. Unrelated numbers elsewhere in the text are
# not automatically assigned to that metric.

NUMBER = (
    r"(?P<number>\d[\d,]*(?:\.\d+)?)"
    r"\s*(?P<scale>million|billion|thousand|k)?"
)

QUALIFIER = (
    r"(?P<qualifier>"
    r"at least|more than|over|above|greater than|"
    r"nearly|almost|about|approximately|around|"
    r"roughly|up to|less than|under"
    r")?\s*"
)

COUNT_BEFORE_METRIC = {
    "DEATH_TOLL": [
        re.compile(
            r"\b" + QUALIFIER + NUMBER
            + r"\s*\+?\s*(?:people\s+)?"
            + r"(?:were\s+)?(?:killed|dead|deaths|fatalities)\b"
        ),
    ],

    "INJURY_COUNT": [
        re.compile(
            r"\b" + QUALIFIER + NUMBER
            + r"\s*\+?\s*(?:people\s+)?"
            + r"(?:were\s+)?(?:injured|wounded)\b"
        ),
    ],

    "EVACUATION_COUNT": [
        re.compile(
            r"\b" + QUALIFIER + NUMBER
            + r"\s*\+?\s*(?:people\s+)?"
            + r"(?:were\s+)?(?:evacuated|displaced)\b"
        ),
        re.compile(
            r"\b" + QUALIFIER + NUMBER
            + r"\s*\+?\s*(?:evacuees)\b"
        ),
    ],

    "MISSING_COUNT": [
        re.compile(
            r"\b" + QUALIFIER + NUMBER
            + r"\s*\+?\s*(?:people\s+)?"
            + r"(?:are\s+)?(?:missing|unaccounted for)\b"
        ),
    ],

    "RESCUE_COUNT": [
        re.compile(
            r"\b" + QUALIFIER + NUMBER
            + r"\s*\+?\s*(?:people\s+)?"
            + r"(?:were\s+)?(?:rescued|saved)\b"
        ),
    ],

    "DAMAGE_COUNT": [
        re.compile(
            r"\b" + QUALIFIER + NUMBER
            + r"\s*\+?\s*(?:homes|houses|buildings|structures)\s+"
            + r"(?:were\s+)?(?:destroyed|damaged)\b"
        ),
    ],
}


# Metric-first patterns for headlines such as "death toll at 61".
# They intentionally require the metric wording before the number.
METRIC_BEFORE_COUNT = {
    "DEATH_TOLL": [
        re.compile(
            r"\bdeath toll\b"
            r"(?:\s+(?:has|had))?"
            r"\s*(?:rises?|rose|reaches?|reached|stands?|stood)?"
            r"\s*(?:to|at|of|is|:)?\s*"
            + NUMBER
        ),
        re.compile(
            r"\b(?:deaths|fatalities|people killed)\b"
            r"\s*(?:rise|rose|reaches?|reached|of|:)?\s*"
            + NUMBER
        ),
    ],

    "INJURY_COUNT": [
        re.compile(
            r"\b(?:injury count|injuries|people injured|wounded)\b"
            r"\s*(?:rise|rose|reaches?|reached|of|:|to|at)?\s*"
            + NUMBER
        ),
    ],

    "EVACUATION_COUNT": [
        re.compile(
            r"\b(?:evacuees|evacuation count|people evacuated|"
            r"people displaced)\b"
            r"\s*(?:reaches?|reached|of|:|to|at)?\s*"
            + NUMBER
        ),
    ],

    "MISSING_COUNT": [
        re.compile(
            r"\b(?:missing people|people missing|missing)\b"
            r"\s*(?:reaches?|reached|of|:|to|at)?\s*"
            + NUMBER
        ),
    ],

    "RESCUE_COUNT": [
        re.compile(
            r"\b(?:people rescued|people saved|rescue count)\b"
            r"\s*(?:reaches?|reached|of|:|to|at)?\s*"
            + NUMBER
        ),
    ],

    "DAMAGE_COUNT": [
        re.compile(
            r"\b(?:homes|houses|buildings|structures)\b"
            r"\s*(?:destroyed|damaged)\s*(?:count)?"
            r"\s*(?:of|:|at)?\s*"
            + NUMBER
        ),
    ],
}


def convert_number(number_text, scale_text=""):
    """Convert comma-separated and abbreviated values to numbers."""
    value = float(number_text.replace(",", ""))
    scale = (scale_text or "").lower()

    if scale in ("k", "thousand"):
        value *= 1_000
    elif scale == "million":
        value *= 1_000_000
    elif scale == "billion":
        value *= 1_000_000_000

    return value


def get_qualifier(match, text, metric_name):
    """
    Return a count qualifier.

    A trailing '+' is recognized from the original matched text.
    """
    qualifier = (match.groupdict().get("qualifier") or "").lower()

    # A '+' directly after the numeric expression indicates a
    # lower-bound style count, e.g. "60+ dead".
    number_end = match.end("number")

    scale = match.groupdict().get("scale") or ""
    if scale:
        number_end = match.end("scale")

    after_number = text[number_end:match.end() + 3]
    if "+" in after_number and qualifier == "":
        qualifier = "at least"

    if qualifier in ("over", "above", "more than", "greater than"):
        return "lower_bound"

    if qualifier == "at least":
        return "lower_bound"

    if qualifier in ("up to", "less than", "under"):
        return "upper_bound"

    if qualifier in (
        "nearly", "almost", "about", "approximately", "around", "roughly"
    ):
        return "approximate"

    return "exact"


def parse_match(match, text, metric_name):
    """Convert a regex match into a structured metric observation."""
    number_text = match.groupdict().get("number")
    scale_text = match.groupdict().get("scale") or ""

    if not number_text:
        return None

    try:
        value = convert_number(number_text, scale_text)
    except (TypeError, ValueError):
        return None

    qualifier = get_qualifier(match, text, metric_name)

    # Convert a count into a comparable interval.
    # Exact values have no interval width. Lower/upper bounds are
    # treated as intervals for consistency checks.
    if qualifier == "lower_bound":
        lower, upper = value, float("inf")
    elif qualifier == "upper_bound":
        lower, upper = 0.0, value
    elif qualifier == "approximate":
        lower = value * (1 - APPROXIMATE_TOLERANCE)
        upper = value * (1 + APPROXIMATE_TOLERANCE)
    else:
        lower, upper = value, value

    return {
        "value": value,
        "qualifier": qualifier,
        "lower": lower,
        "upper": upper,
        "matched_text": match.group(0),
    }



def has_geographic_breakdown(text):
    """
    Detect country/location-specific count lists such as:
    '2-Haiti 4-Dominican Republic 1-NE Colombia'.

    This is a heuristic. It identifies multiple count-location
    segments so they are not mistaken for an overall total.
    """
    text = clean_text(text)

    # Number immediately followed by a hyphen and a location name.
    # Examples: 2-Haiti, 4-Dominican Republic, 1-NE Colombia.
    pattern = re.compile(
        r"(?<!\w)\d[\d,]*(?:\.\d+)?\s*[-–]\s*"
        r"[a-z][a-z .'-]{1,40}"
        r"(?="
        r"\s+\d[\d,]*(?:\.\d+)?\s*[-–]"
        r"|\s*[,;]"
        r"|$"
        r")",
        re.IGNORECASE,
    )

    matches = list(pattern.finditer(text))

    # Two or more number-location entries indicate a likely breakdown.
    return len(matches) >= 2



def has_explicit_total(text, metric_name):
    """
    Return True only when a total is explicitly stated separately
    from a country/location breakdown.

    Merely mentioning 'death toll' does not establish that the
    following number is a global or overall total.
    """
    text = clean_text(text)

    if metric_name == "DEATH_TOLL":
        # Explicit totals such as:
        # "overall death toll is 25"
        # "total deaths: 25"
        patterns = [
            r"\b(?:overall|total)\s+death toll\s+"
            r"(?:is|reaches|stands at|of|:)\s*\d",

            r"\btotal\s+(?:deaths|fatalities)\s*"
            r"(?:is|reaches|stands at|of|:)\s*\d",
        ]

        return any(re.search(pattern, text) for pattern in patterns)

    return False



def extract_metric_values(text):
    """
    Extract metric observations from a report.

    Important:
    If a report contains a geographic breakdown such as
    '2-Haiti 4-Dominican Republic 1-NE Colombia', do not
    interpret the first number as an overall death toll unless
    a separately stated total is available.
    """
    observations = {}

    text = clean_text(text)

    # Detect geographic breakdown BEFORE extracting any metrics.
    geographic_breakdown = has_geographic_breakdown(text)

    for metric_name in COUNT_BEFORE_METRIC:
        # Avoid interpreting country-specific numbers as an overall
        # death toll. For example, 'death toll: 2-Haiti 4-Dominican...'
        if (
            metric_name == "DEATH_TOLL"
            and geographic_breakdown
            and not has_explicit_total(text, metric_name)
        ):
            continue

        found = []

        # Pattern: number followed by metric.
        for pattern in COUNT_BEFORE_METRIC[metric_name]:
            for match in pattern.finditer(text):
                observation = parse_match(
                    match, text, metric_name
                )
                if observation:
                    found.append(observation)

        # Pattern: metric followed by number.
        for pattern in METRIC_BEFORE_COUNT[metric_name]:
            for match in pattern.finditer(text):
                observation = parse_match(
                    match, text, metric_name
                )
                if observation:
                    found.append(observation)

        # Remove duplicate observations.
        unique = {}

        for observation in found:
            key = (
                observation["value"],
                observation["qualifier"],
            )
            unique[key] = observation

        if unique:
            observations[metric_name] = list(unique.values())

    return observations

      


# =========================================================
# 5. COMPARE NUMERICAL OBSERVATIONS
# =========================================================

def intervals_overlap(a, b):
    """Return True if two reported value intervals overlap."""
    return max(a["lower"], b["lower"]) <= min(a["upper"], b["upper"])


def format_number(value):
    """Format counts for review output."""
    if float(value).is_integer():
        return f"{int(value):,}"

    return f"{value:,.2f}".rstrip("0").rstrip(".")


def format_observation(observation):
    """Format the value and qualifier for a human reviewer."""
    value = format_number(observation["value"])
    qualifier = observation["qualifier"]

    if qualifier == "lower_bound":
        return f"at least {value}"

    if qualifier == "upper_bound":
        return f"up to {value}"

    if qualifier == "approximate":
        return f"about {value}"

    return value


def compare_metric_observations(text_a, text_b):
    """
    Compare observations for the same metric.

    Approximate and bound values are not flagged when their possible
    ranges overlap. Different exact values are flagged for review.
    """
    metrics_a = extract_metric_values(text_a)
    metrics_b = extract_metric_values(text_b)

    signals = []

    for metric_name in sorted(set(metrics_a) & set(metrics_b)):
        values_a = metrics_a[metric_name]
        values_b = metrics_b[metric_name]

        # If any values are compatible, don't call the metric
        # contradictory solely because other extracted values differ.
        # If no pair of observations overlaps, flag the discrepancy.
        compatible = any(
            intervals_overlap(a, b)
            for a in values_a
            for b in values_b
        )

        if compatible:
            continue

        a_text = "; ".join(
            sorted({format_observation(x) for x in values_a})
        )
        b_text = "; ".join(
            sorted({format_observation(x) for x in values_b})
        )

        signals.append(
            f"NUMBER_DIFFERENCE_REVIEW_{metric_name}"
            f"(A:{a_text};B:{b_text})"
        )

    return signals


# =========================================================
# 6. LOAD INPUT FILES
# =========================================================

def load_data():
    if not REPORTS_FILE.exists():
        raise FileNotFoundError(
            f"Missing reports file: {REPORTS_FILE}. Run extract.py first."
        )

    if not PAIRS_FILE.exists():
        raise FileNotFoundError(
            f"Missing pairs file: {PAIRS_FILE}. Run group.py first."
        )

    reports = pd.read_csv(
        REPORTS_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    pairs = pd.read_csv(
        PAIRS_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    required_reports = {"tweet_id", "tweet_text"}
    missing_reports = required_reports - set(reports.columns)

    if missing_reports:
        raise ValueError(
            "processed_reports.tsv is missing: "
            + ", ".join(sorted(missing_reports))
        )

    required_pairs = {"tweet_id_a", "tweet_id_b"}
    missing_pairs = required_pairs - set(pairs.columns)

    if missing_pairs:
        raise ValueError(
            "similar_report_pairs.tsv is missing: "
            + ", ".join(sorted(missing_pairs))
        )

    reports = reports.drop_duplicates("tweet_id", keep="first")
    report_lookup = reports.set_index("tweet_id", drop=False)

    return reports, pairs, report_lookup


# =========================================================
# 7. MAIN PROCESSING
# =========================================================

def main():
    reports, pairs, report_lookup = load_data()

    print(f"Input reports: {len(reports):,}")
    print(f"Input candidate pairs: {len(pairs):,}")

    results = []
    skipped_low_similarity = 0
    skipped_missing_reports = 0
    skipped_cross_event = 0

    for _, pair in pairs.iterrows():
        tweet_id_a = str(pair["tweet_id_a"])
        tweet_id_b = str(pair["tweet_id_b"])

        report_a = get_report(report_lookup, tweet_id_a)
        report_b = get_report(report_lookup, tweet_id_b)

        if report_a is None or report_b is None:
            skipped_missing_reports += 1
            continue

        similarity_raw = get_value(
            pair,
            ["similarity", "similarity_score", "score"],
            default="0",
        )

        try:
            similarity = float(similarity_raw)
        except (TypeError, ValueError):
            similarity = 0.0

        if similarity < MIN_SIMILARITY:
            skipped_low_similarity += 1
            continue

        event_a = get_value(report_a, ["event_name"], default="unknown")
        event_b = get_value(report_b, ["event_name"], default="unknown")

        # Only compare reports from the same dataset event.
        if event_a != event_b:
            skipped_cross_event += 1
            continue

        raw_text_a = get_value(report_a, ["tweet_text", "clean_text"])
        raw_text_b = get_value(report_b, ["tweet_text", "clean_text"])

        text_a = clean_text(raw_text_a)
        text_b = clean_text(raw_text_b)

        category_a = get_value(
            report_a,
            ["humanitarian_category", "original_label"],
            default="unknown",
        )
        category_b = get_value(
            report_b,
            ["humanitarian_category", "original_label"],
            default="unknown",
        )

        signals = []
        signals.extend(detect_opposing_claims(text_a, text_b))
        signals.extend(compare_metric_observations(text_a, text_b))

        if not signals:
            continue

        results.append({
            "tweet_id_a": tweet_id_a,
            "tweet_id_b": tweet_id_b,
            "event_name": event_a,
            "similarity": round(similarity, 4),
            "category_a": category_a,
            "category_b": category_b,
            "tweet_text_a": raw_text_a,
            "tweet_text_b": raw_text_b,
            "conflict_signals": " | ".join(signals),
            "review_status": "NEEDS_HUMAN_REVIEW",
            "confirmed_conflict": "NOT_VERIFIED",
        })

    # =====================================================
    # 8. SAVE RESULTS
    # =====================================================

    columns = [
        "tweet_id_a",
        "tweet_id_b",
        "event_name",
        "similarity",
        "category_a",
        "category_b",
        "tweet_text_a",
        "tweet_text_b",
        "conflict_signals",
        "review_status",
        "confirmed_conflict",
    ]

    output_df = pd.DataFrame(results, columns=columns)

    if not output_df.empty:
        output_df = output_df.sort_values(
            by="similarity",
            ascending=False,
        )

    output_df.to_csv(
        OUTPUT_FILE,
        sep="\t",
        index=False,
        encoding="utf-8",
    )

    print("\nConflict review complete.")
    print(f"Candidate pairs checked: {len(pairs):,}")
    print(f"Pairs flagged for review: {len(output_df):,}")
    print(f"Skipped for low similarity: {skipped_low_similarity:,}")
    print(f"Skipped for missing reports: {skipped_missing_reports:,}")
    print(f"Skipped across different events: {skipped_cross_event:,}")
    print(f"Saved to: {OUTPUT_FILE}")

    if output_df.empty:
        print(
            "\nNo pairs matched the current rules. "
            "This does not prove that no conflicts exist."
        )
        return

    print("\nSample flagged pairs:\n")

    for _, row in output_df.head(5).iterrows():
        print("-" * 70)
        print(f"Event: {row['event_name']}")
        print(f"Similarity: {row['similarity']}")
        print(f"Signals: {row['conflict_signals']}")
        print(f"Report A: {row['tweet_text_a']}")
        print(f"Report B: {row['tweet_text_b']}")
        print("Status: NEEDS HUMAN REVIEW — NOT VERIFIED")


# =========================================================
# 9. BASIC SELF-TESTS
# =========================================================

def run_self_tests():
    """Quick checks for known number-extraction edge cases."""

    # Greece: 60+ dead and 74 dead should be compatible lower-bound
    # and exact observations; 150+ injuries must remain separate.
    a = clean_text(
        "60+ dead 150+ injured 1,000 evacuated since 2007"
    )
    b = clean_text(
        "74 dead 150+ injured 1,000+ evacuated since 2007"
    )
    assert "DEATH_TOLL" in extract_metric_values(a)
    assert "INJURY_COUNT" in extract_metric_values(a)
    assert not any(
        "NUMBER_DIFFERENCE_REVIEW" in s
        for s in compare_metric_observations(a, b)
    ), "Greece-style compatible counts should not be flagged."

    # Ecuador: 2,527 versus over 2,500 is potentially consistent.
    a = clean_text("246 dead and 2,527 injured")
    b = clean_text("246 dead and over 2,500 injured")
    assert not any(
        "INJURY_COUNT" in s
        for s in compare_metric_observations(a, b)
    ), "Over 2,500 should be compatible with 2,527."

    # Different exact counts should be flagged.
    a = clean_text("death toll at 61")
    b = clean_text("death toll at 69")
    assert any(
        "NUMBER_DIFFERENCE_REVIEW_DEATH_TOLL" in s
        for s in compare_metric_observations(a, b)
    ), "Different exact death tolls should be flagged."

    # "11 photos" must not be read as 11 deaths.
    text = clean_text("11 photos from an earthquake that left 61 dead")
    extracted = extract_metric_values(text)
    assert extracted["DEATH_TOLL"][0]["value"] == 61, (
        "The photo count must not be mistaken for the death toll."
    )

    # A country breakdown must not be inferred as a total.
    text = clean_text(
        "death toll: 2-Haiti 4-Dominican Republic "
        "1-NE Colombia 1-Grenadines Islands"
    )
    extracted = extract_metric_values(text)
    assert "DEATH_TOLL" not in extracted, (
        "Country-specific figures without a stated total must be skipped."
    )

    print("Self-tests passed.")


if __name__ == "__main__":
    try:
        if "--self-test" in sys.argv:
            run_self_tests()
        else:
            main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
