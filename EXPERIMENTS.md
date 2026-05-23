# mT5 Segmented v2 Experiments

This repo currently keeps two mT5 baselines over `data/segmented_v2`.

## EN -> EN

- Task: English translated chunk -> English summary line
- Input: `translation:` section from `data/segmented_v2/**/segment_sum_concat.txt`
- Target: `summary:` section from `data/segmented_v2/**/segment_sum_concat.txt`
- Data directory: `data/mt5_en2en_segmented_v2/`
- Scripts:
  - `scripts/prepare_mt5_en2en_segmented_v2.py`
  - `scripts/train_mt5_en2en_segmented_v2.py`
  - `scripts/evaluate_mt5_en2en_segmented_v2.py`
- Legacy compatible scripts:
  - `scripts/prepare_mt5_segmented_v2.py`
  - `scripts/train_mt5_segmented_v2.py`
  - `scripts/evaluate_mt5_segmented_v2.py`
- Existing legacy output directories are preserved:
  - `outputs/mt5-small-segmented-v2`
  - `outputs/mt5-small-segmented-v2-epoch10`
  - `outputs/mt5-small-segmented-v2-smoke`
  - `outputs/mt5-small-segmented-v2-smoke-fp32`
- Current 10-epoch legacy article-level ROUGE-1/2/L:
  - `0.282 / 0.059 / 0.157`
- Current 10-epoch chunk-level ROUGE-1/2/L:
  - `0.177 / 0.034 / 0.136`
- Average generated length on the 296 chunk test set:
  - `124.9` words, compared with `127.3` gold-target words

Current 10-epoch EN -> EN sampling rerun results:

- Checkpoint: `outputs/mt5-small-segmented-v2-epoch10`
- Generations: `outputs/mt5-en2en-segmented-v2-epoch10-sampling`
- Sampling config: `do_sample=True`, `temperature=0.8`, `top_p=0.9`, `top_k=50`, `no_repeat_ngram_size=3`, `repetition_penalty=1.2`
- Chunk-level ROUGE-1/2/L: `0.279 / 0.065 / 0.155`
- Article-level ROUGE-1/2/L: `0.396 / 0.111 / 0.171`
- Empty generations: `0`
- Outputs containing `<extra_id`: `0`
- Average generated length: `64.7` words
- Rough repeated sentence count: `0`
- Rough repeated trigram count: `1`

## CC -> EN

- Task: Classical Chinese chunk -> English summary line
- Input: `original:` section from `data/segmented_v2/**/segment_sum_concat.txt`
- Target: `summary:` section from `data/segmented_v2/**/segment_sum_concat.txt`
- Data directory: `data/mt5_cc2en_segmented_v2/`
- Scripts:
  - `scripts/prepare_mt5_cc2en_segmented_v2.py`
  - `scripts/train_mt5_cc2en_segmented_v2.py`
  - `scripts/evaluate_mt5_cc2en_segmented_v2.py`
- Output directories:
  - `outputs/mt5-cc2en-segmented-v2-smoke`
  - `outputs/mt5-cc2en-segmented-v2-epoch1`
  - `outputs/mt5-cc2en-segmented-v2-epoch10`

The CC -> EN data preparation script first uses exact chunk-level `original:` sections
already present in `segment_sum_concat.txt`. It records in `stats.json` whether any
fallback approximate alignment was needed.

Current 10-epoch CC -> EN results:

- Chunk-level ROUGE-1/2/L: `0.143 / 0.017 / 0.115`
- Article-level ROUGE-1/2/L: `0.215 / 0.030 / 0.138`
- Test generations: `296`
- Empty generations: `0`
- Average generated length: `133.4` words, compared with `127.3` gold-target words
- Fallback approximate chunking: `0` articles
- Exact classical chunk articles: `409`

Current 10-epoch CC -> EN sampling rerun results:

- Checkpoint: `outputs/mt5-cc2en-segmented-v2-epoch10`
- Generations: `outputs/mt5-cc2en-segmented-v2-epoch10-sampling`
- Sampling config: `do_sample=True`, `temperature=0.8`, `top_p=0.9`, `top_k=50`, `no_repeat_ngram_size=3`, `repetition_penalty=1.2`
- Chunk-level ROUGE-1/2/L: `0.243 / 0.033 / 0.135`
- Article-level ROUGE-1/2/L: `0.360 / 0.078 / 0.154`
- Empty generations: `0`
- Outputs containing `<extra_id`: `0`
- Average generated length: `66.6` words
- Rough repeated sentence count: `0`
- Rough repeated trigram count: `0`

## Current Comparison

| Pipeline | Chunk R-1 | Chunk R-2 | Chunk R-L | Article R-1 | Article R-2 | Article R-L |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EN -> EN epoch10 | 0.177 | 0.034 | 0.136 | 0.282 | 0.059 | 0.157 |
| EN -> EN epoch10 sampling | 0.279 | 0.065 | 0.155 | 0.396 | 0.111 | 0.171 |
| CC -> EN epoch10 | 0.143 | 0.017 | 0.115 | 0.215 | 0.030 | 0.138 |
| CC -> EN epoch10 sampling | 0.243 | 0.033 | 0.135 | 0.360 | 0.078 | 0.154 |

The direct CC -> EN baseline trails the EN -> EN baseline when both use the same
decoding setup, which is expected because CC -> EN combines classical Chinese
understanding, translation, and summarization in one step.

Sampling improves both pipelines substantially, especially article-level
ROUGE-1/2. The generation stats also show that the rough repetition signals drop
to near zero. However, both sampling runs generate much shorter outputs
(`64.7` words for EN -> EN and `66.6` words for CC -> EN, versus `127.3` gold
words), so the ROUGE gains should be checked qualitatively for coverage and
possible under-generation.
