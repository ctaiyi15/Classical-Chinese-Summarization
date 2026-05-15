import argparse
import csv
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "summary_clean"
DEFAULT_OUTPUT_CSV = DEFAULT_INPUT_ROOT / "opening_review_queue.csv"
DEFAULT_OUTPUT_JSON = DEFAULT_INPUT_ROOT / "opening_review_queue.json"


FIRST_PARAGRAPH_PATTERNS = [
    ("first_line_meta_subject", re.compile(r"^\s*(?:the|this)\s+text\b", re.IGNORECASE)),
    (
        "first_line_begins_by",
        re.compile(r"^\s*(?:it|this|the\s+text)\s+begins\s+(?:by|with)\b", re.IGNORECASE),
    ),
    (
        "generic_history_opening",
        re.compile(
            r"^\s*(?:the\s+history|the\s+lineage|the\s+lives|the\s+deeds|the\s+role|"
            r"the\s+principles|the\s+evolution|a\s+biographical\s+account|"
            r"a\s+detailed\s+(?:census|historical)\s+record|a\s+comprehensive\s+"
            r"(?:historical\s+overview|system))\b",
            re.IGNORECASE,
        ),
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit summary openings for likely residual meta phrasing."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    return parser.parse_args()


def iter_summary_files(root: Path):
    yield from sorted(root.rglob("target.summary.txt"))


def main():
    args = parse_args()
    input_root = args.input_root.resolve()
    output_csv = args.output_csv.resolve()
    output_json = args.output_json.resolve()

    rows = []
    counts = {}

    for path in iter_summary_files(input_root):
        text = path.read_text(encoding="utf-8", errors="ignore")
        first_paragraph = text.split("\n\n", 1)[0].strip()
        if not first_paragraph:
            continue

        matched_rules = []
        for name, pattern in FIRST_PARAGRAPH_PATTERNS:
            if pattern.search(first_paragraph):
                matched_rules.append(name)
                counts[name] = counts.get(name, 0) + 1

        if not matched_rules:
            continue

        rows.append(
            {
                "relative_path": str(path.relative_to(input_root)),
                "rules": ";".join(matched_rules),
                "first_paragraph_preview": first_paragraph[:500],
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["relative_path", "rules", "first_paragraph_preview"],
        )
        writer.writeheader()
        writer.writerows(rows)

    output_json.write_text(
        json.dumps(
            {
                "input_root": str(input_root),
                "flagged_file_count": len(rows),
                "rule_counts": counts,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Flagged {len(rows)} files for review")
    print(f"Wrote {output_csv}")
    print(f"Wrote {output_json}")


if __name__ == "__main__":
    main()
