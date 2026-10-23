
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

EVENT_ROOT = DATA_DIR / "HumAID_data_events_set1_47K" / "events_set1"
COMBINED_ROOT = DATA_DIR / "HumAID_data_all_combined" / "all_combined"

OUTPUT_FILE = DATA_DIR / "humaid_reports_merged.tsv"
AUDIT_FILE = DATA_DIR / "humaid_merge_audit.txt"

print("DATA_DIR:", DATA_DIR)
print("EVENT_ROOT:", EVENT_ROOT)
print("EVENT_ROOT exists:", EVENT_ROOT.exists())

if EVENT_ROOT.exists():
    print("TSV files found:")
    for file in EVENT_ROOT.rglob("*.tsv"):
        print(file)
else:
    print("Folder not found. Check the path above.")
def read_tsv(path):
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig"
    )
    df.columns = df.columns.str.strip()
    return df


def normalize_id(series):
    return series.astype(str).str.strip()


# --------------------------------------------------
# 1. Load event-wise files: these contain tweet text.
# --------------------------------------------------
event_frames = []
event_files = sorted(EVENT_ROOT.rglob("*.tsv"))

for path in event_files:
    df = read_tsv(path)

    required = {"tweet_id", "tweet_text", "class_label"}
    if not required.issubset(df.columns):
        print(f"Skipping unexpected columns: {path}")
        continue

    df["tweet_id"] = normalize_id(df["tweet_id"])
    df["event_name"] = path.parent.name
    df["event_source_file"] = str(path.relative_to(DATA_DIR))

    event_frames.append(
        df[[
            "tweet_id",
            "tweet_text",
            "class_label",
            "event_name",
            "event_source_file"
        ]]
    )

if not event_frames:
    raise RuntimeError("No valid event-wise TSV files found.")

events = pd.concat(event_frames, ignore_index=True)

# --------------------------------------------------
# 2. Load combined files and retain their split names.
# --------------------------------------------------
combined_frames = []
combined_files = sorted(COMBINED_ROOT.glob("*.tsv"))

for path in combined_files:
    df = read_tsv(path)

    required = {"tweet_id", "class_label"}
    if not required.issubset(df.columns):
        print(f"Skipping unexpected columns: {path}")
        continue

    df["tweet_id"] = normalize_id(df["tweet_id"])
    df["combined_split"] = path.stem.replace("all_", "")

    combined_frames.append(
        df[["tweet_id", "class_label", "combined_split"]]
        .rename(columns={"class_label": "combined_class_label"})
    )

if not combined_frames:
    raise RuntimeError("No valid combined TSV files found.")

combined = pd.concat(combined_frames, ignore_index=True)

# --------------------------------------------------
# 3. Check duplicates before matching.
# --------------------------------------------------
event_duplicate_ids = int(events["tweet_id"].duplicated().sum())
combined_duplicate_ids = int(combined["tweet_id"].duplicated().sum())

# Do not silently discard ambiguous IDs.
if event_duplicate_ids:
    print(
        "WARNING: repeated tweet IDs across event files:",
        event_duplicate_ids
    )

if combined_duplicate_ids:
    print(
        "WARNING: repeated tweet IDs across combined splits:",
        combined_duplicate_ids
    )

# Keep the original event-wise records in the merged output.
# Match only IDs that appear exactly once in each dataset.
unique_event_ids = set(
    events.loc[
        ~events["tweet_id"].duplicated(keep=False), "tweet_id"
    ]
)
unique_combined_ids = set(
    combined.loc[
        ~combined["tweet_id"].duplicated(keep=False), "tweet_id"
    ]
)

safe_event = events[events["tweet_id"].isin(unique_event_ids)].copy()
safe_combined = combined[
    combined["tweet_id"].isin(unique_combined_ids)
].copy()

merged = safe_event.merge(
    safe_combined,
    on="tweet_id",
    how="left",
    validate="one_to_one",
    indicator=True
)

merged["combined_match"] = merged["_merge"].eq("both")
merged["label_match"] = (
    merged["combined_match"]
    & merged["class_label"].eq(merged["combined_class_label"])
)

# Preserve actual data; missing metadata remains blank.
merged["data_origin"] = "real"
merged["timestamp"] = ""
merged["lat"] = ""
merged["lon"] = ""

merged = merged.rename(columns={"class_label": "original_label"})

output_columns = [
    "tweet_id",
    "tweet_text",
    "original_label",
    "event_name",
    "combined_split",
    "combined_class_label",
    "combined_match",
    "label_match",
    "data_origin",
    "timestamp",
    "lat",
    "lon",
    "event_source_file"
]

merged[output_columns].to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False,
    encoding="utf-8"
)

# --------------------------------------------------
# 4. Generate an audit report.
# --------------------------------------------------
matched = merged[merged["combined_match"]]
label_matches = int(matched["label_match"].sum())
label_mismatches = int((~matched["label_match"]).sum())

audit = [
    "HUMAID MERGE AUDIT",
    "=" * 50,
    f"Event-wise files found: {len(event_files)}",
    f"Combined files found: {len(combined_files)}",
    f"Event-wise rows: {len(events):,}",
    f"Combined rows: {len(combined):,}",
    f"Repeated event-wise IDs: {event_duplicate_ids:,}",
    f"Repeated combined IDs: {combined_duplicate_ids:,}",
    f"Unique event-wise IDs: {len(unique_event_ids):,}",
    f"Unique combined IDs: {len(unique_combined_ids):,}",
    f"Matched event-wise records: {len(matched):,}",
    f"Unmatched event-wise records: {len(merged) - len(matched):,}",
    f"Matching labels: {label_matches:,}",
    f"Label mismatches among matched records: {label_mismatches:,}",
    "",
    "ROWS PER EVENT:",
    events.groupby("event_name").size().to_string(),
    "",
    "MATCHES BY COMBINED SPLIT:",
    matched.groupby("combined_split").size().to_string()
    if not matched.empty else "No matched records",
    "",
    "LABEL MATCH STATUS:",
    matched["label_match"].value_counts().to_string()
    if not matched.empty else "No matched records",
    "",
    f"Output: {OUTPUT_FILE.name}"
]

AUDIT_FILE.write_text("\n".join(audit), encoding="utf-8")

print("\n".join(audit))
print(f"\nMerged dataset saved to: {OUTPUT_FILE}")
print(f"Audit report saved to: {AUDIT_FILE}")
