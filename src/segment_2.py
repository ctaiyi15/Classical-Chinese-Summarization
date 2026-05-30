from pathlib import Path
from transformers import MT5Tokenizer
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from collections import Counter

original_root = Path("../data/raw/original/史记/七十列传")
translated_root = Path("../data/processed/translated/史记/七十列传")
summary_root = Path("../data/processed/summary_clean/史记/七十列传")
output_root = Path("../data/segmented_v3/史记/七十列传")

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()
    ).float()
    return (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1)


def sentence_similarity(sen1, sen2):
    encoded = tokenizer(
        [sen1, sen2],
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        output = model(**encoded)
    embeddings = mean_pooling(
        output,
        encoded["attention_mask"]
    )
    similarity = F.cosine_similarity(
        embeddings[0].unsqueeze(0),
        embeddings[1].unsqueeze(0)
    )
    return similarity.item()

def similarity_compare(chunk1, chunk2, sen):
    if not chunk1 and not chunk2:
        return 1
    if not chunk1:
        return 2
    if not chunk2:
        return 1
    sim1 = sum([sentence_similarity(s, sen) for s in chunk1]) / len(chunk1)
    sim2 = sum([sentence_similarity(s, sen) for s in chunk2]) / len(chunk2)
    if sim1 >= sim2:
        return 1
    else:
        return 2
    
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

def get_embedding(text, tokenizer, model):
    encoded = tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        output = model(**encoded)

    token_embeddings = output.last_hidden_state
    attention_mask = encoded["attention_mask"]

    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    emb = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1)

    emb = F.normalize(emb, p=2, dim=1)
    return emb[0]

def generate_token_spans(source, len_tokenizer, sem_tokenizer, model, lower_limit, upper_limit, step=3):
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
                span_emb = get_embedding(span_text, sem_tokenizer, model)

                spans.append({
                    "start": start,
                    "end": end,
                    "token_size": current_size,
                    "embedding": span_emb
                })

    return spans

if __name__ == '__main__':

    lower_limit = 1024
    upper_limit = 2048

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    mt5_tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

    for file_count,summary_file in enumerate(summary_root.rglob("target.summary.txt")):
        
        # if file_count == 16:
            print(f'Processing file {file_count + 1}: {summary_file}')
            relative_parent = summary_file.parent.relative_to(summary_root)

            translated_file = (
                translated_root /
                relative_parent /
                "target.txt"
            )
            original_file = original_root / relative_parent / 'source.txt'
            output_all_file = output_root / relative_parent / "segment_all.txt"
            output_sum_file = output_root / relative_parent / "segment_sum.txt"

            with open(summary_file, "r", encoding="utf-8") as f:
                sum_text = [line.strip() for line in f if line.strip()]

            with open(translated_file, "r", encoding="utf-8") as f:
                tran_text = [line.strip() for line in f if line.strip()]

            with open(original_file, "r", encoding="utf-8") as f:
                orig_text = [line.strip() for line in f if line.strip()]

            if not sum_text or not tran_text:
                continue

            chunks = []
            current_chunk = []
            current_chunk_size = 0
            orig_chunks = []
            orig_current_chunk = []
            token_source = [len(mt5_tokenizer.encode(sen)) for sen in tran_text]

            spans = generate_token_spans(tran_text, mt5_tokenizer, tokenizer, model, lower_limit, upper_limit, 3)
            print(f'{len(spans)} spans found.\n')
            span_embeddings = torch.stack([span["embedding"] for span in spans])

            matches = []

            for idx, sum_line in enumerate(sum_text):
                sum_emb = get_embedding(sum_line, tokenizer, model)

                scores = torch.matmul(
                    span_embeddings,
                    sum_emb
                )

                best_idx = torch.argmax(scores).item()
                best_span = spans[best_idx]

                best_score = scores[best_idx].item()

                matches.append({
                    "start": best_span["start"],
                    "end": best_span["end"],
                    "score": best_score
                })

            
            output_sum_file.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            with open(output_sum_file, 'w', encoding='utf-8') as out:

                for i, span in enumerate(matches):

                    start = span["start"]
                    end = span["end"]
                    score = span["score"]

                    out.write(f'Summary line {i}\n')
                    out.write(
                        f'Span: {start}-{end} '
                        f'(score={score:.4f})\n\n'
                    )

                    out.write('original:\n')
                    for sen in orig_text[start:end+1]:
                        out.write(sen + '\n')

                    out.write('\ntranslation:\n')
                    for sen in tran_text[start:end+1]:
                        out.write(sen + '\n')

                    out.write('\nsummary:\n')
                    out.write(sum_text[i] + '\n')

                    out.write('\n' + '=' * 80 + '\n\n')
