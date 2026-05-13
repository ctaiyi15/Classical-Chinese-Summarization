from pathlib import Path
from transformers import MT5Tokenizer
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from collections import Counter


original_root = Path("../data/raw/original/后汉书")
translated_root = Path("../data/processed/translated/后汉书")
summary_root = Path("../data/processed/summary/后汉书")
output_root = Path("../data/segmented/后汉书")

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

if __name__ == '__main__':

    lower_limit = 512
    upper_limit = 1024

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    for file_count,summary_file in enumerate(summary_root.rglob("target.summary.txt")):

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
        token_source = [len(tokenizer.encode(sen)) for sen in tran_text]

        for i,sen in enumerate(tran_text):
            if current_chunk_size + token_source[i] < lower_limit:
                current_chunk.append(sen)
                orig_current_chunk.append(orig_text[i])
                current_chunk_size += token_source[i]
            elif current_chunk_size + token_source[i] > upper_limit:
                chunks.append(current_chunk)
                orig_chunks.append(orig_current_chunk)
                current_chunk = [sen]
                orig_current_chunk = [orig_text[i]]
                current_chunk_size = token_source[i]
            else:
                j = i + 1
                next_chunk = []
                next_chunk_size = 0
                while j < len(tran_text):
                    if next_chunk_size + token_source[j] > lower_limit:
                        break
                    next_chunk.append(tran_text[j])
                    next_chunk_size += token_source[j]
                    j += 1
                if similarity_compare(current_chunk, next_chunk, sen) == 1:
                    current_chunk.append(sen)
                    orig_current_chunk.append(orig_text[i])
                    current_chunk_size += token_source[i]
                else:
                    chunks.append(current_chunk)
                    orig_chunks.append(orig_current_chunk)

                    current_chunk = [sen]
                    orig_current_chunk = [orig_text[i]]
                    current_chunk_size = token_source[i]
        if current_chunk:
            chunks.append(current_chunk)
            orig_chunks.append(orig_current_chunk)

        sum_chunk = []
        for sen in sum_text:
            scores = []
            for c in chunks:
                text = ' '.join(c)
                score, _, _ = rouge_1(text, sen)
                scores.append(score)
            idx = scores.index(max(scores))
            sum_chunk.append(idx)

        with open(output_sum_file, 'w', encoding='utf-8') as out:
            out.write(f'Summary uses chunk {sum_chunk}\n\n')
            for i,chunk_num in enumerate(sum_chunk):
                out.write(f'Chunk {chunk_num}: \n')
                out.write('original: \n')
                for sen in orig_chunks[chunk_num]:
                    out.write(sen + '\n')
                out.write('translation: \n')
                for sen in chunks[chunk_num]:
                    out.write(sen + '\n')
                
                out.write('\nsummary:\n')
                out.write(f'summary line {i}: {sum_text[i]}')
                out.write('\n\n')

        
        output_all_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_all_file, 'w', encoding='utf-8') as out:
            for i,chunk in enumerate(chunks):
                out.write(f'Chunk {i}: \n')
                out.write('original: \n')
                for sen in orig_chunks[i]:
                    out.write(sen + '\n')
                out.write('translation: \n')
                for sen in chunk:
                    out.write(sen + '\n')
                out.write('\n\n')

