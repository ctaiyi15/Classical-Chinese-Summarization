import argparse
import csv
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "summary"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "summary_clean"


SUMMARY_HEADER_RE = re.compile(r"^\s*#{1,6}\s*summary\b.*$", re.IGNORECASE)
INTRO_ONLY_RE = re.compile(
    r"^\s*based\s+(solely|strictly)\s+on\s+the\s+provided\s+text\s*,?\s*"
    r"(here\s+is\s+a\s+summary(?:(?:\s+of)?[^:]*)?:?|the\s+summary\s+is\s+as\s+follows:?)?\s*$",
    re.IGNORECASE,
)
INTRO_PREFIX_RE = re.compile(
    r"^\s*based\s+(solely|strictly)\s+on\s+the\s+provided\s+text\s*,?\s*",
    re.IGNORECASE,
)
META_PREFIX_RE = re.compile(
    r"^\s*(here\s+is\s+a\s+summary(?:(?:\s+of)?[^:]*)?:?|the\s+summary\s+is\s+as\s+follows:?)\s*",
    re.IGNORECASE,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove LLM-style preambles from summary files without modifying originals."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def strip_leading_blank_lines(lines):
    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    return lines[idx:]


def clean_first_content_line(line):
    original = line
    line = INTRO_PREFIX_RE.sub("", line, count=1)
    line = META_PREFIX_RE.sub("", line, count=1)
    line = line.lstrip(" :-\u2014")
    if line and line[0].islower():
        line = line[0].upper() + line[1:]
    return line if line != original else original


def clean_summary_text(text):
    lines = text.splitlines()
    original_lines = list(lines)
    lines = strip_leading_blank_lines(lines)

    actions = []

    if not lines:
        return text, actions

    first = lines[0].strip()

    if SUMMARY_HEADER_RE.match(first):
        actions.append("removed_markdown_summary_header")
        lines = lines[1:]
        lines = strip_leading_blank_lines(lines)
        if not lines:
            return "", actions
        first = lines[0].strip()

    if INTRO_ONLY_RE.match(first):
        actions.append("removed_standalone_intro_line")
        lines = lines[1:]
        lines = strip_leading_blank_lines(lines)
        if not lines:
            return "", actions
    else:
        cleaned_line = clean_first_content_line(lines[0])
        if cleaned_line != lines[0]:
            actions.append("trimmed_inline_intro_prefix")
            lines[0] = cleaned_line

        if INTRO_ONLY_RE.match(lines[0].strip()):
            actions.append("removed_residual_intro_line")
            lines = lines[1:]
            lines = strip_leading_blank_lines(lines)

    cleaned_text = "\n".join(lines).strip()
    if cleaned_text:
        cleaned_text += "\n"

    if cleaned_text == text and lines == original_lines:
        actions = []

    return cleaned_text, actions


def write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    report_rows = []
    action_counts = {}
    processed_count = 0
    modified_count = 0

    for path in sorted(input_root.rglob("*")):
        relative_path = path.relative_to(input_root)
        output_path = output_root / relative_path

        if path.is_dir():
            output_path.mkdir(parents=True, exist_ok=True)
            continue

        if path.name == ".gitkeep":
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
            continue

        if path.suffix != ".txt":
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(path.read_bytes())
            continue

        processed_count += 1
        original_text = path.read_text(encoding="utf-8", errors="ignore")
        cleaned_text, actions = clean_summary_text(original_text)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(cleaned_text, encoding="utf-8")

        if actions:
            modified_count += 1
            for action in actions:
                action_counts[action] = action_counts.get(action, 0) + 1

        report_rows.append(
            {
                "relative_path": str(relative_path),
                "modified": "yes" if actions else "no",
                "actions": ";".join(actions),
            }
        )

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "processed_file_count": processed_count,
        "modified_file_count": modified_count,
        "action_counts": action_counts,
    }

    write_csv(
        output_root / "cleanup_report.csv",
        report_rows,
        ["relative_path", "modified", "actions"],
    )
    (output_root / "cleanup_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Processed {processed_count} summary files")
    print(f"Modified {modified_count} files")
    print(f"Wrote cleaned summaries to {output_root}")
    print(f"Wrote report to {output_root / 'cleanup_report.json'}")


if __name__ == "__main__":
    main()
