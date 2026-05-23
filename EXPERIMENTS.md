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

## CC -> EN

- Task: Classical Chinese chunk -> English summary line
- Input: `original:` section from `data/segmented_v2/**/segment_sum_concat.txt`
- Target: `summary:` section from `data/segmented_v2/**/segment_sum_concat.txt`
- Data directory: `data/mt5_cc2en_segmented_v2/`
- Scripts:
  - `scripts/prepare_mt5_cc2en_segmented_v2.py`
  - `scripts/train_mt5_cc2en_segmented_v2.py`
  - `scripts/evaluate_mt5_cc2en_segmented_v2.py`
- Planned output directories:
  - `outputs/mt5-cc2en-segmented-v2-smoke`
  - `outputs/mt5-cc2en-segmented-v2-epoch1`
  - `outputs/mt5-cc2en-segmented-v2-epoch10`

The CC -> EN data preparation script first uses exact chunk-level `original:` sections
already present in `segment_sum_concat.txt`. It records in `stats.json` whether any
fallback approximate alignment was needed.
