from collections import Counter
from math import exp, log


def _safe_divide(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _char_ngrams(text, n):
    if n <= 0 or len(text) < n:
        return []
    return [text[i:i + n] for i in range(len(text) - n + 1)]


def _ngram_counts(tokens, n):
    if n <= 0 or len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _simple_tokenize(text):
    text = text.strip()
    if not text:
        return []

    if " " in text:
        return [token for token in text.split() if token]

    return list(text)


def _levenshtein_distance(orig, tran):
    m, n = len(orig), len(tran)
    if m == 0:
        return n
    if n == 0:
        return m

    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            substitution_cost = 0 if orig[i - 1] == tran[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + substitution_cost,
            )
        prev, curr = curr, prev

    return prev[n]


def chrF(orig, tran, max_order=6, beta=2.0):
    beta_sq = beta ** 2
    f_scores = []

    for n in range(1, max_order + 1):
        orig_count = Counter(_char_ngrams(orig, n))
        tran_count = Counter(_char_ngrams(tran, n))

        if not orig_count and not tran_count:
            f_scores.append(0.0)
            continue

        overlap = sum((orig_count & tran_count).values())
        precision = _safe_divide(overlap, sum(tran_count.values()))
        recall = _safe_divide(overlap, sum(orig_count.values()))

        if precision == 0.0 and recall == 0.0:
            f_scores.append(0.0)
            continue

        f_score = ((1 + beta_sq) * precision * recall) / (beta_sq * precision + recall)
        f_scores.append(f_score)

    return sum(f_scores) / len(f_scores)


def bleu(orig, tran, max_order=4, smooth=True):
    reference = _simple_tokenize(orig)
    hypothesis = _simple_tokenize(tran)

    if not hypothesis:
        return 0.0

    precisions = []
    for n in range(1, max_order + 1):
        hyp_count = _ngram_counts(hypothesis, n)
        ref_count = _ngram_counts(reference, n)

        matches = sum((hyp_count & ref_count).values())
        total = sum(hyp_count.values())

        if total == 0:
            precisions.append(0.0)
            continue

        if smooth:
            precisions.append((matches + 1) / (total + 1))
        else:
            precisions.append(matches / total if matches > 0 else 0.0)

    if min(precisions) == 0.0:
        return 0.0

    ref_len = len(reference)
    hyp_len = len(hypothesis)
    if hyp_len == 0:
        return 0.0

    if hyp_len > ref_len:
        brevity_penalty = 1.0
    else:
        brevity_penalty = exp(1 - _safe_divide(ref_len, hyp_len))

    score = brevity_penalty * exp(sum(log(p) for p in precisions) / max_order)
    return score


def edit_similarity(orig, tran):
    max_len = max(len(orig), len(tran))
    if max_len == 0:
        return 1.0

    distance = _levenshtein_distance(orig, tran)
    return 1.0 - (distance / max_len)


def length_ratio(orig, tran):
    if len(orig) == 0:
        return 0.0 if len(tran) > 0 else 1.0
    return len(tran) / len(orig)


def embedding_similarity(orig, tran, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "embedding_similarity requires the optional dependency "
            "'sentence-transformers'. Install it first to enable this metric."
        ) from exc

    model = SentenceTransformer(model_name)
    orig_embedding, tran_embedding = model.encode([orig, tran], normalize_embeddings=True)
    return float(orig_embedding @ tran_embedding)


def score_all(orig, tran, include_embedding=False, embedding_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    scores = {
        "chrf": chrF(orig, tran),
        "bleu": bleu(orig, tran),
        "edit_similarity": edit_similarity(orig, tran),
        "length_ratio": length_ratio(orig, tran),
    }

    if include_embedding:
        scores["embedding_similarity"] = embedding_similarity(
            orig,
            tran,
            model_name=embedding_model_name,
        )

    return scores
