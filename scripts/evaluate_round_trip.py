import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.evaluation.metrics import bleu, chrF, edit_similarity, length_ratio
from models.evaluation.rouge import rouge_1


DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "data" / "raw" / "晋书"
DEFAULT_BACK_ROOT = PROJECT_ROOT / "data" / "processed" / "translated_back" / "晋书"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "evaluation" / "晋书_round_trip"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate round-trip translation quality for matched target.txt files."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--back-root", type=Path, default=DEFAULT_BACK_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def list_target_files(root):
    return sorted(path.relative_to(root) for path in root.rglob("target.txt"))


def read_lines(path):
    text = path.read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines()]


def calc_line_metrics(orig, tran):
    rouge1_precision, rouge1_recall, rouge1_f1 = safe_rouge_1(orig, tran)
    return {
        "rouge1_precision": rouge1_precision,
        "rouge1_recall": rouge1_recall,
        "rouge1_f1": rouge1_f1,
        "chrf": chrF(orig, tran),
        "bleu": bleu(orig, tran),
        "edit_similarity": edit_similarity(orig, tran),
        "length_ratio": length_ratio(orig, tran),
        "exact_match": 1.0 if orig == tran else 0.0,
    }


def safe_rouge_1(orig, tran):
    try:
        return rouge_1(orig, tran)
    except ZeroDivisionError:
        orig_count = Counter(list(orig))
        tran_count = Counter(list(tran))
        combine = orig_count & tran_count
        overlap_count = sum(combine.values())
        precision = overlap_count / len(tran) if tran else 0.0
        recall = overlap_count / len(orig) if orig else 0.0
        denominator = precision + recall
        f1 = (2 * precision * recall) / denominator if denominator else 0.0
        return precision, recall, f1


def mean_for(records, key):
    values = [record[key] for record in records]
    return mean(values) if values else 0.0


def build_reliability_checks(corpus_summary):
    checks = [
        {
            "metric": "matched_file_coverage",
            "target": ">= 0.90",
            "value": corpus_summary["matched_file_coverage"],
            "passed": corpus_summary["matched_file_coverage"] >= 0.90,
            "why": "Most translated chapters should have a round-trip counterpart before we trust corpus-level averages.",
        },
        {
            "metric": "aligned_line_coverage",
            "target": ">= 0.98",
            "value": corpus_summary["aligned_line_coverage"],
            "passed": corpus_summary["aligned_line_coverage"] >= 0.98,
            "why": "Most lines should survive round-trip without alignment loss.",
        },
        {
            "metric": "mean_chrf",
            "target": ">= 0.50",
            "value": corpus_summary["mean_chrf"],
            "passed": corpus_summary["mean_chrf"] >= 0.50,
            "why": "Character-level overlap should remain reasonably strong for Chinese round-trip text.",
        },
        {
            "metric": "mean_edit_similarity",
            "target": ">= 0.55",
            "value": corpus_summary["mean_edit_similarity"],
            "passed": corpus_summary["mean_edit_similarity"] >= 0.55,
            "why": "Round-trip text should stay more similar than dissimilar at the character level.",
        },
        {
            "metric": "mean_length_ratio",
            "target": "between 0.85 and 1.15",
            "value": corpus_summary["mean_length_ratio"],
            "passed": 0.85 <= corpus_summary["mean_length_ratio"] <= 1.15,
            "why": "Reliable round-trip translation should not systematically compress or expand too much.",
        },
    ]
    return checks


def write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_markdown_report(summary, file_summaries, missing_files, output_root):
    top_by_chrf = sorted(file_summaries, key=lambda item: item["mean_chrf"], reverse=True)[:5]
    bottom_by_chrf = sorted(file_summaries, key=lambda item: item["mean_chrf"])[:5]

    lines = [
        "# 晋书 Round-Trip Translation Evaluation",
        "",
        "## Overview",
        "",
        f"- Source root: `{summary['source_root']}`",
        f"- Back-translation root: `{summary['back_root']}`",
        f"- Source files found: {summary['source_file_count']}",
        f"- Back-translation files found: {summary['back_file_count']}",
        f"- Matched files evaluated: {summary['matched_file_count']}",
        f"- Missing round-trip files: {summary['missing_in_back_count']}",
        f"- Aligned line pairs evaluated: {summary['aligned_line_count']}",
        f"- Unaligned source-only lines: {summary['source_only_line_count']}",
        f"- Unaligned back-only lines: {summary['back_only_line_count']}",
        "",
        "## Corpus-Level Metrics",
        "",
        f"- Mean ROUGE-1 F1: {summary['mean_rouge1_f1']:.4f}",
        f"- Mean chrF: {summary['mean_chrf']:.4f}",
        f"- Mean BLEU: {summary['mean_bleu']:.4f}",
        f"- Mean edit similarity: {summary['mean_edit_similarity']:.4f}",
        f"- Mean length ratio: {summary['mean_length_ratio']:.4f}",
        f"- Exact-match rate: {summary['exact_match_rate']:.4f}",
        f"- Matched file coverage: {summary['matched_file_coverage']:.4f}",
        f"- Aligned line coverage: {summary['aligned_line_coverage']:.4f}",
        "",
        "## Reliability Checks",
        "",
    ]

    for check in summary["reliability_checks"]:
        status = "PASS" if check["passed"] else "FLAG"
        lines.append(
            f"- {status} `{check['metric']}`: observed {check['value']:.4f}, target {check['target']}. {check['why']}"
        )

    lines.extend([
        "",
        "## Strongest Files by chrF",
        "",
    ])
    for item in top_by_chrf:
        lines.append(
            f"- `{item['relative_path']}`: chrF {item['mean_chrf']:.4f}, edit similarity {item['mean_edit_similarity']:.4f}, aligned lines {item['aligned_line_count']}"
        )

    lines.extend([
        "",
        "## Weakest Files by chrF",
        "",
    ])
    for item in bottom_by_chrf:
        lines.append(
            f"- `{item['relative_path']}`: chrF {item['mean_chrf']:.4f}, edit similarity {item['mean_edit_similarity']:.4f}, aligned lines {item['aligned_line_count']}"
        )

    lines.extend([
        "",
        "## Missing Round-Trip Files",
        "",
    ])
    if missing_files:
        for relative_path in missing_files:
            lines.append(f"- `{relative_path}`")
    else:
        lines.append("- None")

    lines.extend([
        "",
        "## Output Files",
        "",
        f"- Line-level records: `{output_root / 'line_metrics.csv'}`",
        f"- File-level summary: `{output_root / 'file_summary.csv'}`",
        f"- JSON summary: `{output_root / 'summary.json'}`",
    ])

    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    source_root = args.source_root.resolve()
    back_root = args.back_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    source_files = list_target_files(source_root)
    back_files = list_target_files(back_root)
    source_set = set(source_files)
    back_set = set(back_files)

    common_files = sorted(source_set & back_set)
    missing_in_back = sorted(source_set - back_set)

    line_records = []
    file_summaries = []
    source_only_line_count = 0
    back_only_line_count = 0

    for relative_path in common_files:
        source_lines = read_lines(source_root / relative_path)
        back_lines = read_lines(back_root / relative_path)
        aligned_count = min(len(source_lines), len(back_lines))
        source_only = max(len(source_lines) - aligned_count, 0)
        back_only = max(len(back_lines) - aligned_count, 0)
        source_only_line_count += source_only
        back_only_line_count += back_only

        file_line_records = []
        for idx, (orig, tran) in enumerate(zip(source_lines[:aligned_count], back_lines[:aligned_count]), start=1):
            metrics = calc_line_metrics(orig, tran)
            row = {
                "relative_path": str(relative_path),
                "line_number": idx,
                "orig_text": orig,
                "back_text": tran,
                **metrics,
            }
            line_records.append(row)
            file_line_records.append(row)

        file_summary = {
            "relative_path": str(relative_path),
            "source_line_count": len(source_lines),
            "back_line_count": len(back_lines),
            "aligned_line_count": aligned_count,
            "source_only_line_count": source_only,
            "back_only_line_count": back_only,
            "mean_rouge1_f1": mean_for(file_line_records, "rouge1_f1"),
            "mean_chrf": mean_for(file_line_records, "chrf"),
            "mean_bleu": mean_for(file_line_records, "bleu"),
            "mean_edit_similarity": mean_for(file_line_records, "edit_similarity"),
            "mean_length_ratio": mean_for(file_line_records, "length_ratio"),
            "exact_match_rate": mean_for(file_line_records, "exact_match"),
        }
        file_summaries.append(file_summary)

    matched_file_coverage = len(common_files) / len(source_files) if source_files else 0.0
    aligned_line_count = len(line_records)
    total_source_lines = aligned_line_count + source_only_line_count
    aligned_line_coverage = aligned_line_count / total_source_lines if total_source_lines else 0.0

    corpus_summary = {
        "source_root": str(source_root),
        "back_root": str(back_root),
        "source_file_count": len(source_files),
        "back_file_count": len(back_files),
        "matched_file_count": len(common_files),
        "missing_in_back_count": len(missing_in_back),
        "missing_in_back_files": [str(path) for path in missing_in_back],
        "aligned_line_count": aligned_line_count,
        "source_only_line_count": source_only_line_count,
        "back_only_line_count": back_only_line_count,
        "matched_file_coverage": matched_file_coverage,
        "aligned_line_coverage": aligned_line_coverage,
        "mean_rouge1_f1": mean_for(line_records, "rouge1_f1"),
        "mean_chrf": mean_for(line_records, "chrf"),
        "mean_bleu": mean_for(line_records, "bleu"),
        "mean_edit_similarity": mean_for(line_records, "edit_similarity"),
        "mean_length_ratio": mean_for(line_records, "length_ratio"),
        "exact_match_rate": mean_for(line_records, "exact_match"),
    }
    corpus_summary["reliability_checks"] = build_reliability_checks(corpus_summary)

    write_csv(
        output_root / "line_metrics.csv",
        line_records,
        [
            "relative_path",
            "line_number",
            "orig_text",
            "back_text",
            "rouge1_precision",
            "rouge1_recall",
            "rouge1_f1",
            "chrf",
            "bleu",
            "edit_similarity",
            "length_ratio",
            "exact_match",
        ],
    )
    write_csv(
        output_root / "file_summary.csv",
        file_summaries,
        [
            "relative_path",
            "source_line_count",
            "back_line_count",
            "aligned_line_count",
            "source_only_line_count",
            "back_only_line_count",
            "mean_rouge1_f1",
            "mean_chrf",
            "mean_bleu",
            "mean_edit_similarity",
            "mean_length_ratio",
            "exact_match_rate",
        ],
    )
    (output_root / "summary.json").write_text(
        json.dumps(corpus_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "summary.md").write_text(
        build_markdown_report(corpus_summary, file_summaries, corpus_summary["missing_in_back_files"], output_root),
        encoding="utf-8",
    )

    print(f"Wrote line metrics to {output_root / 'line_metrics.csv'}")
    print(f"Wrote file summary to {output_root / 'file_summary.csv'}")
    print(f"Wrote summary JSON to {output_root / 'summary.json'}")
    print(f"Wrote summary Markdown to {output_root / 'summary.md'}")


if __name__ == "__main__":
    main()
