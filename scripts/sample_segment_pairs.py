import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "data" / "segmented_v2"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "evaluation" / "segment_audit"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample chunk-summary pairs from segmented_v2 for manual auditing."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--per-corpus", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_segment_sum(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"(?=^Chunk \d+:\s*$)", text, flags=re.M)
    pairs = []
    for block in blocks:
        chunk_match = re.search(r"^Chunk\s+(\d+):\s*$", block, flags=re.M)
        if not chunk_match:
            continue
        original_match = re.search(
            r"original:\s*\n(.*?)\ntranslation:\s*\n",
            block,
            flags=re.S,
        )
        translation_match = re.search(
            r"translation:\s*\n(.*?)\nsummary:\s*\n",
            block,
            flags=re.S,
        )
        summary_match = re.search(r"summary line\s+(\d+):\s*(.*)", block, flags=re.S)
        if not (original_match and translation_match and summary_match):
            continue

        pairs.append(
            {
                "chunk_id": int(chunk_match.group(1)),
                "summary_idx": int(summary_match.group(1)),
                "original": original_match.group(1).strip(),
                "translation": translation_match.group(1).strip(),
                "summary": summary_match.group(2).strip(),
            }
        )
    return pairs


def main():
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    rng = random.Random(args.seed)

    pairs_by_corpus = defaultdict(list)
    for path in sorted(input_root.rglob("segment_sum.txt")):
        corpus = path.relative_to(input_root).parts[0]
        for pair in parse_segment_sum(path):
            pair["file"] = str(path.relative_to(input_root))
            pair["corpus"] = corpus
            pairs_by_corpus[corpus].append(pair)

    sample = []
    for corpus, pairs in sorted(pairs_by_corpus.items()):
        chosen = pairs if len(pairs) <= args.per_corpus else rng.sample(pairs, args.per_corpus)
        for item in chosen:
            sample.append(item)

    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "sample_pairs.json"
    md_path = output_root / "sample_pairs.md"

    json_path.write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")

    with md_path.open("w", encoding="utf-8") as handle:
        for idx, item in enumerate(sample, start=1):
            handle.write(f"## Pair {idx}\n")
            handle.write(f"- corpus: {item['corpus']}\n")
            handle.write(f"- file: {item['file']}\n")
            handle.write(f"- chunk_id: {item['chunk_id']}\n")
            handle.write(f"- summary_idx: {item['summary_idx']}\n\n")
            handle.write("### Summary\n")
            handle.write(item["summary"] + "\n\n")
            handle.write("### Translation Chunk\n")
            handle.write(item["translation"] + "\n\n")
            handle.write("### Original Chunk\n")
            handle.write(item["original"] + "\n\n")

    print(f"Wrote {len(sample)} sampled pairs")
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
