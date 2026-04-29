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


DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw"
DEFAULT_TRANSLATED_ROOT = PROJECT_ROOT / "data" / "processed" / "translated"
DEFAULT_BACK_ROOT = PROJECT_ROOT / "data" / "processed" / "translated_back"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "evaluation" / "all_round_trip"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate all corpora with raw and round-trip back-translation data."
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--translated-root", type=Path, default=DEFAULT_TRANSLATED_ROOT)
    parser.add_argument("--back-root", type=Path, default=DEFAULT_BACK_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def list_target_files(root):
    if not root.exists():
        return []
    return sorted(path.relative_to(root) for path in root.rglob("target.txt"))


def read_lines(path):
    text = path.read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines()]


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


def mean_for(records, key):
    values = [record[key] for record in records]
    return mean(values) if values else 0.0


def build_reliability_checks(summary):
    return [
        {
            "metric": "matched_file_coverage",
            "target": ">= 0.90",
            "value": summary["matched_file_coverage"],
            "passed": summary["matched_file_coverage"] >= 0.90,
        },
        {
            "metric": "aligned_line_coverage",
            "target": ">= 0.98",
            "value": summary["aligned_line_coverage"],
            "passed": summary["aligned_line_coverage"] >= 0.98,
        },
        {
            "metric": "mean_rouge1_f1",
            "target": ">= 0.45",
            "value": summary["mean_rouge1_f1"],
            "passed": summary["mean_rouge1_f1"] >= 0.45,
        },
        {
            "metric": "mean_chrf",
            "target": ">= 0.18",
            "value": summary["mean_chrf"],
            "passed": summary["mean_chrf"] >= 0.18,
        },
        {
            "metric": "mean_edit_similarity",
            "target": ">= 0.35",
            "value": summary["mean_edit_similarity"],
            "passed": summary["mean_edit_similarity"] >= 0.35,
        },
        {
            "metric": "mean_length_ratio",
            "target": "between 0.90 and 1.15",
            "value": summary["mean_length_ratio"],
            "passed": 0.90 <= summary["mean_length_ratio"] <= 1.15,
        },
    ]


def evaluate_corpus(corpus_name, raw_root, translated_root, back_root):
    raw_files = list_target_files(raw_root / corpus_name)
    translated_files = list_target_files(translated_root / corpus_name)
    back_files = list_target_files(back_root / corpus_name)

    raw_set = set(raw_files)
    back_set = set(back_files)
    translated_set = set(translated_files)

    common_files = sorted(raw_set & back_set)
    missing_in_back = sorted(raw_set - back_set)
    back_without_raw = sorted(back_set - raw_set)
    translated_without_back = sorted(translated_set - back_set)

    line_records = []
    source_only_line_count = 0
    back_only_line_count = 0

    for relative_path in common_files:
        raw_lines = read_lines(raw_root / corpus_name / relative_path)
        back_lines = read_lines(back_root / corpus_name / relative_path)

        aligned_count = min(len(raw_lines), len(back_lines))
        source_only = max(len(raw_lines) - aligned_count, 0)
        back_only = max(len(back_lines) - aligned_count, 0)
        source_only_line_count += source_only
        back_only_line_count += back_only

        for idx, (orig, tran) in enumerate(zip(raw_lines[:aligned_count], back_lines[:aligned_count]), start=1):
            metrics = calc_line_metrics(orig, tran)
            line_records.append(
                {
                    "corpus": corpus_name,
                    "relative_path": str(relative_path),
                    "line_number": idx,
                    **metrics,
                }
            )

    aligned_line_count = len(line_records)
    total_source_lines = aligned_line_count + source_only_line_count

    summary = {
        "corpus": corpus_name,
        "raw_file_count": len(raw_files),
        "translated_file_count": len(translated_files),
        "back_file_count": len(back_files),
        "matched_file_count": len(common_files),
        "missing_in_back_count": len(missing_in_back),
        "back_without_raw_count": len(back_without_raw),
        "translated_without_back_count": len(translated_without_back),
        "aligned_line_count": aligned_line_count,
        "source_only_line_count": source_only_line_count,
        "back_only_line_count": back_only_line_count,
        "matched_file_coverage": len(common_files) / len(raw_files) if raw_files else 0.0,
        "aligned_line_coverage": aligned_line_count / total_source_lines if total_source_lines else 0.0,
        "mean_rouge1_f1": mean_for(line_records, "rouge1_f1"),
        "mean_chrf": mean_for(line_records, "chrf"),
        "mean_bleu": mean_for(line_records, "bleu"),
        "mean_edit_similarity": mean_for(line_records, "edit_similarity"),
        "mean_length_ratio": mean_for(line_records, "length_ratio"),
        "exact_match_rate": mean_for(line_records, "exact_match"),
        "missing_in_back_files": [str(path) for path in missing_in_back],
        "back_without_raw_files": [str(path) for path in back_without_raw],
        "translated_without_back_files": [str(path) for path in translated_without_back],
    }
    summary["reliability_checks"] = build_reliability_checks(summary)
    return summary, line_records


def write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_markdown_report(inventory_rows, evaluated_summaries, overall_summary, output_root):
    lines = [
        "# All-Corpus Round-Trip Evaluation",
        "",
        "## Corpus Inventory",
        "",
        "| Corpus | Raw | Translated | Back | Evaluable |",
        "| --- | ---: | ---: | ---: | --- |",
    ]

    for row in inventory_rows:
        lines.append(
            f"| {row['corpus']} | {row['raw_file_count']} | {row['translated_file_count']} | {row['back_file_count']} | {'yes' if row['is_evaluable'] else 'no'} |"
        )

    lines.extend([
        "",
        "## Overall Summary",
        "",
        f"- Corpora with translated files: {overall_summary['translated_corpus_count']}",
        f"- Corpora with back-translation files: {overall_summary['back_corpus_count']}",
        f"- Corpora evaluated with raw references: {overall_summary['evaluated_corpus_count']}",
        f"- Total matched files evaluated: {overall_summary['matched_file_count']}",
        f"- Total aligned line pairs evaluated: {overall_summary['aligned_line_count']}",
        f"- Mean ROUGE-1 F1: {overall_summary['mean_rouge1_f1']:.4f}",
        f"- Mean chrF: {overall_summary['mean_chrf']:.4f}",
        f"- Mean BLEU: {overall_summary['mean_bleu']:.4f}",
        f"- Mean edit similarity: {overall_summary['mean_edit_similarity']:.4f}",
        f"- Mean length ratio: {overall_summary['mean_length_ratio']:.4f}",
        "",
        "## Per-Corpus Results",
        "",
    ])

    for summary in evaluated_summaries:
        lines.extend([
            f"### {summary['corpus']}",
            "",
            f"- Matched files: {summary['matched_file_count']} / {summary['raw_file_count']}",
            f"- Aligned line pairs: {summary['aligned_line_count']}",
            f"- Mean ROUGE-1 F1: {summary['mean_rouge1_f1']:.4f}",
            f"- Mean chrF: {summary['mean_chrf']:.4f}",
            f"- Mean BLEU: {summary['mean_bleu']:.4f}",
            f"- Mean edit similarity: {summary['mean_edit_similarity']:.4f}",
            f"- Mean length ratio: {summary['mean_length_ratio']:.4f}",
        ])
        flags = [check for check in summary["reliability_checks"] if not check["passed"]]
        if flags:
            lines.append("- Flags:")
            for check in flags:
                lines.append(
                    f"  - {check['metric']}: observed {check['value']:.4f}, target {check['target']}"
                )
        else:
            lines.append("- Flags: none")
        if summary["missing_in_back_files"]:
            lines.append(f"- Missing round-trip files: {len(summary['missing_in_back_files'])}")
        lines.append("")

    lines.extend([
        "## Output Files",
        "",
        f"- Corpus inventory: `{output_root / 'corpus_inventory.csv'}`",
        f"- Corpus metrics: `{output_root / 'corpus_metrics.csv'}`",
        f"- Line metrics: `{output_root / 'line_metrics.csv'}`",
        f"- JSON summary: `{output_root / 'summary.json'}`",
    ])

    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    raw_root = args.raw_root.resolve()
    translated_root = args.translated_root.resolve()
    back_root = args.back_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    raw_corpora = {path.name for path in raw_root.iterdir() if path.is_dir()}
    translated_corpora = {path.name for path in translated_root.iterdir() if path.is_dir()}
    back_corpora = {path.name for path in back_root.iterdir() if path.is_dir()}
    all_corpora = sorted(translated_corpora | back_corpora)

    inventory_rows = []
    evaluated_summaries = []
    all_line_records = []

    for corpus in all_corpora:
        raw_count = sum(1 for _ in (raw_root / corpus).rglob("target.txt")) if (raw_root / corpus).exists() else 0
        translated_count = sum(1 for _ in (translated_root / corpus).rglob("target.txt")) if (translated_root / corpus).exists() else 0
        back_count = sum(1 for _ in (back_root / corpus).rglob("target.txt")) if (back_root / corpus).exists() else 0

        row = {
            "corpus": corpus,
            "raw_file_count": raw_count,
            "translated_file_count": translated_count,
            "back_file_count": back_count,
            "is_evaluable": "yes" if corpus in raw_corpora and corpus in back_corpora else "no",
        }
        inventory_rows.append(row)

        if row["is_evaluable"] == "yes":
            summary, line_records = evaluate_corpus(corpus, raw_root, translated_root, back_root)
            evaluated_summaries.append(summary)
            all_line_records.extend(line_records)

    overall_summary = {
        "translated_corpus_count": len(translated_corpora),
        "back_corpus_count": len(back_corpora),
        "evaluated_corpus_count": len(evaluated_summaries),
        "matched_file_count": sum(item["matched_file_count"] for item in evaluated_summaries),
        "aligned_line_count": sum(item["aligned_line_count"] for item in evaluated_summaries),
        "mean_rouge1_f1": mean_for(all_line_records, "rouge1_f1"),
        "mean_chrf": mean_for(all_line_records, "chrf"),
        "mean_bleu": mean_for(all_line_records, "bleu"),
        "mean_edit_similarity": mean_for(all_line_records, "edit_similarity"),
        "mean_length_ratio": mean_for(all_line_records, "length_ratio"),
        "corpora": evaluated_summaries,
        "inventory": inventory_rows,
    }

    write_csv(
        output_root / "corpus_inventory.csv",
        inventory_rows,
        ["corpus", "raw_file_count", "translated_file_count", "back_file_count", "is_evaluable"],
    )
    write_csv(
        output_root / "corpus_metrics.csv",
        evaluated_summaries,
        [
            "corpus",
            "raw_file_count",
            "translated_file_count",
            "back_file_count",
            "matched_file_count",
            "missing_in_back_count",
            "back_without_raw_count",
            "translated_without_back_count",
            "aligned_line_count",
            "source_only_line_count",
            "back_only_line_count",
            "matched_file_coverage",
            "aligned_line_coverage",
            "mean_rouge1_f1",
            "mean_chrf",
            "mean_bleu",
            "mean_edit_similarity",
            "mean_length_ratio",
            "exact_match_rate",
        ],
    )
    write_csv(
        output_root / "line_metrics.csv",
        all_line_records,
        [
            "corpus",
            "relative_path",
            "line_number",
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
    (output_root / "summary.json").write_text(
        json.dumps(overall_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "summary.md").write_text(
        build_markdown_report(inventory_rows, evaluated_summaries, overall_summary, output_root),
        encoding="utf-8",
    )

    print(f"Wrote corpus inventory to {output_root / 'corpus_inventory.csv'}")
    print(f"Wrote corpus metrics to {output_root / 'corpus_metrics.csv'}")
    print(f"Wrote line metrics to {output_root / 'line_metrics.csv'}")
    print(f"Wrote summary JSON to {output_root / 'summary.json'}")
    print(f"Wrote summary Markdown to {output_root / 'summary.md'}")


if __name__ == "__main__":
    main()
