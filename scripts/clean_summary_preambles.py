import argparse
import csv
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "summary"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "summary_clean"


SUMMARY_HEADER_RE = re.compile(r"^\s*#{1,6}\s*summary\b.*$", re.IGNORECASE)
BOLD_SUMMARY_HEADER_RE = re.compile(
    r"^\s*\*{1,2}\s*summary\b.*?\*{1,2}\s*$",
    re.IGNORECASE,
)
PLAIN_SUMMARY_HEADER_RE = re.compile(
    r"^\s*summary\b.*$",
    re.IGNORECASE,
)
GENERIC_MARKDOWN_TITLE_RE = re.compile(
    r"^\s*#{1,6}\s+\S.*$",
    re.IGNORECASE,
)
GENERIC_BOLD_TITLE_RE = re.compile(
    r"^\s*\*{1,2}[^*\n]{2,120}\*{1,2}\s*$",
    re.IGNORECASE,
)
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
OPENING_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
DROP_OPENING_SENTENCE_RE = re.compile(
    r"^(?:the|this)\s+text\s+begins\s+(?:by|with)\b.*$",
    re.IGNORECASE,
)
REWRITE_OPENING_SENTENCE_PATTERNS = [
    (
        re.compile(
            r"^(?:the|this)\s+text\s+"
            r"(?:discusses|describes|details|recounts|summarizes|chronicles|examines|"
            r"presents|outlines|contains|traces)\s+",
            re.IGNORECASE,
        ),
        "trimmed_meta_opening_prefix",
    ),
    (
        re.compile(
            r"^(?:the|this)\s+text\s+provides\s+",
            re.IGNORECASE,
        ),
        "trimmed_meta_opening_prefix",
    ),
    (
        re.compile(
            r"^(?:the|this)\s+text\s+is\s+",
            re.IGNORECASE,
        ),
        "trimmed_meta_opening_prefix",
    ),
    (
        re.compile(
            r"^(?:the|this)\s+text\s+focuses\s+on\s+",
            re.IGNORECASE,
        ),
        "trimmed_meta_opening_prefix",
    ),
    (
        re.compile(
            r"^(?:the|this)\s+text\s+is\s+about\s+",
            re.IGNORECASE,
        ),
        "trimmed_meta_opening_prefix",
    ),
    (
        re.compile(
            r"^(?:it|this)\s+begins\s+by\s+noting\s+that\s+",
            re.IGNORECASE,
        ),
        "trimmed_meta_opening_prefix",
    ),
    (
        re.compile(
            r"^(?:it|this)\s+begins\s+by\s+noting\s+",
            re.IGNORECASE,
        ),
        "trimmed_meta_opening_prefix",
    ),
    (
        re.compile(
            r"^(?:it|this)\s+begins\s+with\s+",
            re.IGNORECASE,
        ),
        "trimmed_meta_opening_prefix",
    ),
]


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


def normalize_meta_opening_sentences(text):
    if not text.strip():
        return text, []

    parts = text.split("\n\n", 1)
    first_paragraph = parts[0].strip()
    if not first_paragraph:
        return text, []

    sentences = OPENING_SENTENCE_SPLIT_RE.split(first_paragraph)
    actions = []

    idx = 0
    reviewed = 0
    while idx < len(sentences) and reviewed < 2:
        sentence = sentences[idx].strip()
        if not sentence:
            idx += 1
            continue

        if DROP_OPENING_SENTENCE_RE.match(sentence):
            actions.append("removed_meta_opening_sentence")
            sentences.pop(idx)
            reviewed += 1
            continue

        replaced = False
        for pattern, action in REWRITE_OPENING_SENTENCE_PATTERNS:
            match = pattern.match(sentence)
            if not match:
                continue
            rewritten = sentence[match.end():].lstrip(" :-\u2014")
            if rewritten:
                rewritten = rewritten[0].upper() + rewritten[1:]
                sentences[idx] = rewritten
                actions.append(action)
            else:
                sentences.pop(idx)
                actions.append("removed_meta_opening_sentence")
            replaced = True
            break

        if not replaced:
            break

        reviewed += 1
        idx += 1

    rebuilt_first_paragraph = " ".join(sentence.strip() for sentence in sentences if sentence.strip())
    if not rebuilt_first_paragraph:
        remaining = parts[1] if len(parts) > 1 else ""
        return remaining.lstrip(), actions

    rebuilt = rebuilt_first_paragraph
    if len(parts) > 1:
        rebuilt += "\n\n" + parts[1].lstrip("\n")
    return rebuilt, actions


def remove_internal_title_lines(text):
    if not text.strip():
        return text, []

    actions = []
    kept_lines = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped and GENERIC_MARKDOWN_TITLE_RE.match(stripped):
            actions.append("removed_internal_markdown_title")
            continue
        if stripped and GENERIC_BOLD_TITLE_RE.match(stripped):
            actions.append("removed_internal_bold_title")
            continue
        kept_lines.append(line)

    cleaned_text = "\n".join(kept_lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text).strip()
    return cleaned_text, actions


def clean_summary_text(text):
    lines = text.splitlines()
    original_lines = list(lines)
    lines = strip_leading_blank_lines(lines)

    actions = []

    if not lines:
        return text, actions

    while lines:
        first = lines[0].strip()

        if SUMMARY_HEADER_RE.match(first):
            actions.append("removed_markdown_summary_header")
            lines = lines[1:]
            lines = strip_leading_blank_lines(lines)
            continue

        if BOLD_SUMMARY_HEADER_RE.match(first):
            actions.append("removed_bold_summary_header")
            lines = lines[1:]
            lines = strip_leading_blank_lines(lines)
            continue

        if PLAIN_SUMMARY_HEADER_RE.match(first):
            actions.append("removed_plain_summary_header")
            lines = lines[1:]
            lines = strip_leading_blank_lines(lines)
            continue

        if GENERIC_MARKDOWN_TITLE_RE.match(first):
            actions.append("removed_generic_markdown_title")
            lines = lines[1:]
            lines = strip_leading_blank_lines(lines)
            continue

        if GENERIC_BOLD_TITLE_RE.match(first):
            actions.append("removed_generic_bold_title")
            lines = lines[1:]
            lines = strip_leading_blank_lines(lines)
            continue

        break

    if not lines:
        return "", actions

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
    cleaned_text, opening_actions = normalize_meta_opening_sentences(cleaned_text)
    actions.extend(opening_actions)
    cleaned_text, internal_title_actions = remove_internal_title_lines(cleaned_text)
    actions.extend(internal_title_actions)
    cleaned_text = cleaned_text.strip()
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
