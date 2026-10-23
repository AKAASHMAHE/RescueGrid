
from pathlib import Path
import pandas as pd

# This script should be saved directly inside the "data" folder.
DATA_DIR = Path(__file__).resolve().parent

SUMMARY_FILE = DATA_DIR / "humaid_inspection_summary.csv"
REPORT_FILE = DATA_DIR / "humaid_inspection_report.txt"

# Find all TSV files inside both dataset folders.
tsv_files = sorted(
    p for p in DATA_DIR.rglob("*")
    if p.is_file()
    and p.suffix.lower() == ".tsv"
    and p.name.lower() not in {
        SUMMARY_FILE.name.lower(),
        REPORT_FILE.name.lower(),
    }
)

if not tsv_files:
    print("No TSV files found!")
    print(f"Checked folder: {DATA_DIR}")
    raise SystemExit(1)

summary_rows = []
report_lines = []

def shorten(value, limit=180):
    value = str(value).replace("\n", " ").replace("\r", " ")
    return value if len(value) <= limit else value[:limit] + "..."

print(f"\nDataset folder: {DATA_DIR}")
print(f"Total TSV files found: {len(tsv_files)}")

for number, file_path in enumerate(tsv_files, start=1):
    print("\n" + "=" * 80)
    print(f"[{number}/{len(tsv_files)}] FILE: {file_path.relative_to(DATA_DIR)}")

    # Try common encodings in case a file is not UTF-8.
    df = None
    errors = []

    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(
                file_path,
                sep="\t",
                dtype=str,
                encoding=encoding,
                keep_default_na=False,
                on_bad_lines="warn",
                low_memory=False,
            )
            break
        except Exception as exc:
            errors.append(f"{encoding}: {exc}")

    if df is None:
        message = f"FAILED TO READ: {file_path}\n" + "\n".join(errors)
        print(message)
        report_lines.append(message)
        continue

    columns = list(df.columns)
    print(f"Rows: {len(df):,}")
    print(f"Columns ({len(columns)}): {columns}")

    # Guess which dataset this file belongs to.
    path_lower = str(file_path).lower()
    if "combined" in path_lower:
        dataset_type = "combined"
    elif "events_set1" in path_lower:
        dataset_type = "event_wise"
    else:
        dataset_type = "other"

    summary_rows.append({
        "file": str(file_path.relative_to(DATA_DIR)),
        "dataset_type": dataset_type,
        "rows": len(df),
        "column_count": len(columns),
        "columns": " | ".join(columns),
    })

    report_lines.append("\n" + "=" * 80)
    report_lines.append(f"FILE: {file_path.relative_to(DATA_DIR)}")
    report_lines.append(f"DATASET TYPE: {dataset_type}")
    report_lines.append(f"ROWS: {len(df):,}")
    report_lines.append(f"COLUMNS: {columns}")

    # Show missing values and exact duplicate rows.
    missing = {
        col: int((df[col].astype(str).str.strip() == "").sum())
        for col in columns
    }
    missing = {k: v for k, v in missing.items() if v > 0}

    duplicate_rows = int(df.duplicated().sum())
    print(f"Exact duplicate rows: {duplicate_rows:,}")
    print(f"Blank values by column: {missing or 'None'}")

    report_lines.append(f"EXACT DUPLICATE ROWS: {duplicate_rows:,}")
    report_lines.append(f"BLANK VALUES: {missing or 'None'}")

    # Look for likely tweet ID columns.
    id_columns = [
        col for col in columns
        if "id" in col.lower()
        or "tweet_id" in col.lower()
        or "tweetid" in col.lower()
    ]

    for col in id_columns:
        nonblank = df[col].astype(str).str.strip()
        nonblank = nonblank[nonblank != ""]
        duplicate_ids = int(nonblank.duplicated().sum())
        print(
            f"ID-like column '{col}': "
            f"{len(nonblank):,} nonblank values; "
            f"{duplicate_ids:,} repeated values"
        )
        report_lines.append(
            f"ID COLUMN '{col}': {len(nonblank):,} nonblank; "
            f"{duplicate_ids:,} repeated values"
        )

    # Show only three sample rows; long text is truncated.
    print("\nSample rows:")
    report_lines.append("\nSAMPLE ROWS:")

    if df.empty:
        print("(File has headers but no data rows.)")
        report_lines.append("(No data rows.)")
    else:
        for _, row in df.head(3).iterrows():
            sample = {
                col: shorten(row[col])
                for col in columns
            }
            print(sample)
            report_lines.append(str(sample))

    # Print likely useful columns for our project.
    important_words = (
        "text", "tweet", "label", "class", "event",
        "id", "date", "time", "location", "source"
    )
    useful_columns = [
        col for col in columns
        if any(word in col.lower() for word in important_words)
    ]

    print(f"\nPotentially useful columns: {useful_columns}")
    report_lines.append(f"POTENTIALLY USEFUL COLUMNS: {useful_columns}")

# Save a compact file inventory.
if summary_rows:
    pd.DataFrame(summary_rows).to_csv(SUMMARY_FILE, index=False)

REPORT_FILE.write_text("\n".join(report_lines), encoding="utf-8")

print("\n" + "=" * 80)
print("INSPECTION COMPLETE")
print(f"TSV files inspected: {len(summary_rows)} / {len(tsv_files)}")
print(f"Summary saved to: {SUMMARY_FILE}")
print(f"Detailed report saved to: {REPORT_FILE}")
print("\nNext: open huma id_inspection_report.txt and check the results.")
