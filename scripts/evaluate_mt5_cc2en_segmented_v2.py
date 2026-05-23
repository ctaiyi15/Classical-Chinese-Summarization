import argparse
import json
from collections import OrderedDict
from pathlib import Path
from statistics import mean

from rouge_score import rouge_scorer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GENERATIONS_FILE = (
    PROJECT_ROOT / "outputs" / "mt5-cc2en-segmented-v2-epoch1" / "test_generations.jsonl"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate chunk and article summaries from mT5 generations."
    )
    parser.add_argument("--generations_file", type=Path, default=DEFAULT_GENERATIONS_FILE)
    parser.add_argument("--output_dir", type=Path)
    return parser.parse_args()


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def score_pair(scorer, generated, gold):
    scores = scorer.score(gold, generated)
    return {name: value.fmeasure for name, value in scores.items()}


def mean_scores(rows):
    if not rows:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    return {
        metric: mean(row[metric] for row in rows)
        for metric in ("rouge1", "rouge2", "rougeL")
    }


def article_rows(rows):
    grouped = OrderedDict()
    for row in rows:
        key = row["article_path"]
        if key not in grouped:
            grouped[key] = {
                "corpus": row["corpus"],
                "article_path": key,
                "ids": [],
                "chunk_ids": [],
                "gold_parts": [],
                "generated_parts": [],
            }
        grouped[key]["ids"].append(row["id"])
        grouped[key]["chunk_ids"].extend(row["chunk_ids"])
        grouped[key]["gold_parts"].append(row["gold_target"])
        grouped[key]["generated_parts"].append(row["generated"])

    output = []
    for data in grouped.values():
        output.append(
            {
                "corpus": data["corpus"],
                "article_path": data["article_path"],
                "ids": data["ids"],
                "chunk_ids": data["chunk_ids"],
                "gold_target": "\n".join(data["gold_parts"]),
                "generated": "\n".join(data["generated_parts"]),
            }
        )
    return output


def main():
    args = parse_args()
    output_dir = args.output_dir or args.generations_file.parent
    rows = read_jsonl(args.generations_file)
    if not rows:
        raise ValueError(f"No generations found in {args.generations_file}.")

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )
    chunk_scores = [
        score_pair(scorer, row["generated"], row["gold_target"]) for row in rows
    ]

    articles = article_rows(rows)
    article_scores = []
    for row in articles:
        row.update(score_pair(scorer, row["generated"], row["gold_target"]))
        article_scores.append({metric: row[metric] for metric in ("rouge1", "rouge2", "rougeL")})

    article_path = output_dir / "article_generations.jsonl"
    eval_path = output_dir / "eval_results.json"
    write_jsonl(article_path, articles)
    eval_results = {
        "generations_file": str(args.generations_file),
        "chunk_samples": len(rows),
        "article_samples": len(articles),
        "chunk_level": mean_scores(chunk_scores),
        "article_level": mean_scores(article_scores),
    }
    eval_path.write_text(
        json.dumps(eval_results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(eval_results, ensure_ascii=False, indent=2))
    print(f"saved article generations: {article_path}")
    print(f"saved evaluation results: {eval_path}")


if __name__ == "__main__":
    main()
