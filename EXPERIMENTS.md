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

## Current Comparison

| Pipeline | Chunk R-1 | Chunk R-2 | Chunk R-L | Article R-1 | Article R-2 | Article R-L |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EN -> EN epoch10 | 0.177 | 0.034 | 0.136 | 0.282 | 0.059 | 0.157 |
| CC -> EN epoch10 | 0.143 | 0.017 | 0.115 | 0.215 | 0.030 | 0.138 |

The direct CC -> EN baseline trails the EN -> EN baseline, especially on ROUGE-2.
This is expected because CC -> EN combines classical Chinese understanding,
translation, and summarization in one step. Sample generations are non-empty but
often repetitive and generic, so the current result is best read as a successful
end-to-end baseline rather than a strong final model.
