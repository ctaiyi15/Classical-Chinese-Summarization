from pathlib import Path
from transformers import AutoTokenizer
from collections import Counter

original_root = Path("../data/raw/original")
translated_root = Path("../data/processed/translated")
summary_root = Path("../data/processed/summary_clean")
output_root = Path("../data/segmented_v4")
    
def rouge_1(orig, tran):
    orig_tokens = orig.lower().split()
    tran_tokens = tran.lower().split()
    if not orig_tokens or not tran_tokens:
        return (0, 0, 0)
    orig_count = Counter(orig_tokens)
    tran_count = Counter(tran_tokens)
    combine = orig_count & tran_count
    overlap_count = sum(combine.values())
    precision = overlap_count / len(tran_tokens)
    recall = overlap_count / len(orig_tokens)
    if precision + recall == 0:
        F1 = 0
    else:
        F1 = (2 * precision * recall) / (precision + recall)
    return (precision, recall, F1)

def generate_token_spans(source, len_tokenizer, lower_limit, upper_limit, step=3):
    spans = []
    token_source = [len(len_tokenizer.encode(sen)) for sen in source]

    n = len(source)

    for start in range(0, n, step):
        current_span = []
        current_size = 0

        for end in range(start, n):
            sen_size = token_source[end]

            if current_size + sen_size > upper_limit:
                break

            current_span.append(source[end])
            current_size += sen_size

            if current_size >= lower_limit:
                span_text = " ".join(current_span)

                spans.append({
                    "start": start,
                    "end": end,
                    "token_size": current_size,
                    "text": span_text
                })

    return spans

if __name__ == "__main__":

    lower_limit = 1024
    upper_limit = 2048

    mt5_tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

    for file_count,summary_file in enumerate(summary_root.rglob("target.summary.txt")):
        
        # if file_count == 16:
            print(f"Processing file {file_count + 1}: {summary_file}")
            relative_parent = summary_file.parent.relative_to(summary_root)

            translated_file = (
                translated_root /
                relative_parent /
                "target.txt"
            )
            original_file = original_root / relative_parent / "source.txt"
            output_sum_file = output_root / relative_parent / "segment_sum.txt"

            with open(summary_file, "r", encoding="utf-8") as f:
                sum_text = [line.strip() for line in f if line.strip()]

            with open(translated_file, "r", encoding="utf-8") as f:
                tran_text = [line.strip() for line in f if line.strip()]

            with open(original_file, "r", encoding="utf-8") as f:
                orig_text = [line.strip() for line in f if line.strip()]

            if not sum_text or not tran_text:
                continue

            spans = generate_token_spans(tran_text, mt5_tokenizer, lower_limit, upper_limit, 3)
            print(f"{len(spans)} spans found.\n")

            matches = []

            for idx, sum_line in enumerate(sum_text):

                best_score = float("-inf")
                best_span = None

                for span in spans:
                    span_text = span["text"]

                    precision, recall, f1 = rouge_1(
                        sum_line,
                        span_text
                    )
                    summary_len = len(sum_line.split())
                    span_len = len(span_text.split())

                    length_ratio = span_len / summary_len if summary_len > 0 else 1
                    length_penalty = 0.02 * length_ratio
                    score = recall - length_penalty

                    if score > best_score:
                        best_score = score
                        best_span = span
                        
                if best_span is None:
                    continue

                matches.append({
                    "start": best_span["start"],
                    "end": best_span["end"],
                    "score": best_score,
                    "summary_sentences": [sum_line]
                })

            output_sum_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_sum_file, "w", encoding="utf-8") as out:

                for i, span in enumerate(matches):

                    start = span["start"]
                    end = span["end"]
                    score = span["score"]

                    out.write(f"Summary line {i}\n")
                    out.write(
                        f"Span: {start}-{end} "
                        f"(score={score:.4f})\n\n"
                    )

                    out.write("original:\n")
                    for sen in orig_text[start:end+1]:
                        out.write(sen + "\n")

                    out.write("\ntranslation:\n")
                    for sen in tran_text[start:end+1]:
                        out.write(sen + "\n")

                    out.write("\nsummary:\n")
                    out.write(sum_text[i] + "\n")

                    out.write("\n" + "=" * 80 + "\n\n")
