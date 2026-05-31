import argparse
import json
import random
import re
from collections import OrderedDict
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "data" / "segmented_v2"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "mt5_cc2en_segmented_v2"
CHUNK_SPLIT_RE = re.compile(r"^Chunk\s+(\d+):\s*$", re.MULTILINE)
SUMMARY_SPLIT_RE = re.compile(r"^Summary line\s+(\d+)\s*$", re.MULTILINE | re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare article-split CC->EN mT5 samples from segmented_v2 chunk summaries."
    )
    parser.add_argument("--input_root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max_target_words", type=int, default=300)
    parser.add_argument("--train_ratio", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_text(text):
    return " ".join(line.strip() for line in text.splitlines() if line.strip())


def target_word_count(text):
    return len(text.split())


def source_char_count(text):
    return len("".join(text.split()))


def section_text(block, start_label, end_label=None):
    start_match = re.search(rf"^{re.escape(start_label)}\s*$", block, re.MULTILINE)
    if not start_match:
        return ""

    start = start_match.end()
    if end_label is None:
        end = len(block)
    else:
        end_match = re.search(
            rf"^{re.escape(end_label)}\s*$",
            block[start:],
            re.MULTILINE,
        )
        end = start + end_match.start() if end_match else len(block)
    return normalize_text(block[start:end])


def parse_concat_file(path):
    content = path.read_text(encoding="utf-8")
    matches = list(CHUNK_SPLIT_RE.finditer(content))
    pairs = []

    for idx, match in enumerate(matches):
        block_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
        block = content[match.end():block_end]
        source = section_text(block, "original:", "translation:")
        target = section_text(block, "summary:")
        if source and target:
            pairs.append(
                {
                    "chunk_id": int(match.group(1)),
                    "source": source,
                    "target": target,
                    "source_alignment": "exact_segment_original_section",
                }
            )
    return pairs


def parse_segment_sum_file(path):
    content = path.read_text(encoding="utf-8")
    matches = list(SUMMARY_SPLIT_RE.finditer(content))
    pairs = []

    for idx, match in enumerate(matches):
        block_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
        block = content[match.end():block_end]
        source = section_text(block, "original:", "translation:")
        target = section_text(block, "summary:")
        if source and target:
            pairs.append(
                {
                    "chunk_id": int(match.group(1)),
                    "source": source,
                    "target": target,
                    "source_alignment": "exact_segment_original_section",
                }
            )
    return pairs


def parse_segment_file(path):
    if path.name == "segment_sum_concat.txt":
        return parse_concat_file(path)
    return parse_segment_sum_file(path)


def candidate_source_paths(article_dir, input_root):
    relative_dir = article_dir.relative_to(input_root)
    roots = [
        PROJECT_ROOT / "data" / "raw",
        PROJECT_ROOT / "data" / "processed",
        PROJECT_ROOT / "data" / "segmented_v2",
        PROJECT_ROOT / "data" / "segmented",
    ]
    names = ["source.txt", "original.txt", "classical.txt", "segment_all.txt"]
    candidates = []
    for root in roots:
        for name in names:
            candidates.append(root / relative_dir / name)
    candidates.append(article_dir / "segment_sum_concat.txt")
    return candidates


def article_metadata(article_dir, input_root):
    relative_dir = article_dir.relative_to(input_root)
    parts = relative_dir.parts
    corpus = parts[0] if parts else ""
    return corpus, relative_dir.as_posix()


def has_cjk(text):
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def group_article_pairs(pairs, corpus, article_path, max_target_words):
    filtered = []
    removed_long = 0
    for pair in pairs:
        if target_word_count(pair["target"]) > max_target_words:
            removed_long += 1
            continue
        filtered.append(pair)

    grouped = OrderedDict()
    for pair in filtered:
        target = pair["target"]
        if target not in grouped:
            grouped[target] = {
                "corpus": corpus,
                "article_path": article_path,
                "chunk_ids": [],
                "source_parts": [],
                "target": target,
            }
        grouped[target]["chunk_ids"].append(pair["chunk_id"])
        grouped[target]["source_parts"].append(pair["source"])

    samples = []
    for group_idx, data in enumerate(grouped.values()):
        samples.append(
            {
                "id": f"{article_path}::group-{group_idx:03d}",
                "corpus": data["corpus"],
                "article_path": data["article_path"],
                "chunk_ids": [str(chunk_id) for chunk_id in data["chunk_ids"]],
                "source_language": "classical_chinese",
                "target_language": "english",
                "source": "\n\n".join(data["source_parts"]),
                "target": data["target"],
            }
        )
    return samples, removed_long


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_articles(article_samples, train_ratio, seed):
    article_paths = sorted(article_samples)
    rng = random.Random(seed)
    rng.shuffle(article_paths)

    if len(article_paths) <= 1:
        train_count = len(article_paths)
    else:
        train_count = round(len(article_paths) * train_ratio)
        train_count = max(1, min(train_count, len(article_paths) - 1))

    train_articles = set(article_paths[:train_count])
    train_samples = []
    test_samples = []
    for article_path in sorted(article_samples):
        destination = train_samples if article_path in train_articles else test_samples
        destination.extend(article_samples[article_path])
    return train_articles, train_samples, test_samples


def main():
    args = parse_args()
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train_ratio must be between 0 and 1.")

    segment_files = sorted(args.input_root.rglob("segment_sum_concat.txt"))
    input_format = "segment_sum_concat"
    if not segment_files:
        segment_files = sorted(args.input_root.rglob("segment_sum.txt"))
        input_format = "segment_sum"

    if not segment_files:
        raise FileNotFoundError(
            f"No segment_sum_concat.txt or segment_sum.txt files found below {args.input_root}."
        )

    article_samples = {}
    corpora = set()
    raw_pairs = 0
    removed_long = 0
    missing_articles = []
    exact_classical_chunk_articles = 0
    fallback_aligned_articles = 0

    for concat_file in segment_files:
        corpus, article_path = article_metadata(concat_file.parent, args.input_root)
        corpora.add(corpus)
        pairs = parse_segment_file(concat_file)
        raw_pairs += len(pairs)

        has_classical_source = bool(pairs) and all(has_cjk(pair["source"]) for pair in pairs)
        if not has_classical_source:
            missing_articles.append(
                {
                    "article_path": article_path,
                    "candidate_paths": [
                        str(path) for path in candidate_source_paths(concat_file.parent, args.input_root)
                    ],
                }
            )
            continue

        exact_classical_chunk_articles += 1
        samples, article_removed_long = group_article_pairs(
            pairs,
            corpus,
            article_path,
            args.max_target_words,
        )
        removed_long += article_removed_long
        if samples:
            article_samples[article_path] = samples

    train_articles, train_samples, test_samples = split_articles(
        article_samples,
        args.train_ratio,
        args.seed,
    )
    grouped_samples = train_samples + test_samples
    avg_source_chars = (
        mean(source_char_count(row["source"]) for row in grouped_samples)
        if grouped_samples
        else 0.0
    )
    avg_target_words = (
        mean(target_word_count(row["target"]) for row in grouped_samples)
        if grouped_samples
        else 0.0
    )

    write_jsonl(args.output_root / "train.jsonl", train_samples)
    write_jsonl(args.output_root / "test.jsonl", test_samples)

    stats = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "source_from": f"{args.input_root}/**/{input_format}.txt original sections",
        "alignment": f"exact chunk-level original sections from {input_format}.txt",
        "used_fallback_approximate_chunking": False,
        "max_target_words": args.max_target_words,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "total_corpora": len(corpora),
        "input_format": input_format,
        "segment_files": len(segment_files),
        "total_articles_discovered": len(segment_files),
        "articles_with_classical_source_found": exact_classical_chunk_articles,
        "articles_missing_classical_source": len(missing_articles),
        "raw_pairs": raw_pairs,
        "removed_long_summary_pairs": removed_long,
        "grouped_samples": len(grouped_samples),
        "train_articles": len(train_articles),
        "test_articles": len(article_samples) - len(train_articles),
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "average_source_characters": avg_source_chars,
        "average_target_words": avg_target_words,
        "fallback_aligned_article_count": fallback_aligned_articles,
        "exact_classical_chunk_article_count": exact_classical_chunk_articles,
        "missing_articles": missing_articles,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"total corpora: {stats['total_corpora']}")
    print(f"total articles discovered: {stats['total_articles_discovered']}")
    print(f"articles with classical source found: {stats['articles_with_classical_source_found']}")
    print(f"articles missing classical source: {stats['articles_missing_classical_source']}")
    print(f"raw pairs: {stats['raw_pairs']}")
    print(f"removed long-summary pairs: {stats['removed_long_summary_pairs']}")
    print(f"grouped samples: {stats['grouped_samples']}")
    print(f"train/test articles: {stats['train_articles']} / {stats['test_articles']}")
    print(f"train/test samples: {stats['train_samples']} / {stats['test_samples']}")
    print(f"average source characters: {stats['average_source_characters']:.2f}")
    print(f"average target words: {stats['average_target_words']:.2f}")
    print(f"fallback-aligned article count: {stats['fallback_aligned_article_count']}")
    print(f"exact-classical-chunk article count: {stats['exact_classical_chunk_article_count']}")
    if missing_articles:
        print("missing classical source articles:")
        for item in missing_articles[:20]:
            print(f"- {item['article_path']}")
            print("  expected candidate paths:")
            for candidate in item["candidate_paths"]:
                print(f"  - {candidate}")
        if len(missing_articles) > 20:
            print(f"... {len(missing_articles) - 20} more missing articles")


if __name__ == "__main__":
    main()
