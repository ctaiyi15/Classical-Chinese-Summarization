# All-Corpus Round-Trip Evaluation

## Corpus Inventory

| Corpus | Raw | Translated | Back | Evaluable |
| --- | ---: | ---: | ---: | --- |
| 史记 | 129 | 112 | 112 | yes |
| 后汉书 | 100 | 98 | 95 | yes |
| 晋书 | 47 | 43 | 39 | yes |
| 梁书 | 56 | 51 | 47 | yes |
| 汉书 | 112 | 107 | 107 | yes |

## Overall Summary

- Corpora with translated files: 5
- Corpora with back-translation files: 5
- Corpora evaluated with raw references: 5
- Total matched files evaluated: 400
- Total aligned line pairs evaluated: 125800
- Mean ROUGE-1 F1: 0.5303
- Mean chrF: 0.2060
- Mean BLEU: 0.2247
- Mean edit similarity: 0.4111
- Mean length ratio: 1.1149

## Per-Corpus Results

### 史记

- Matched files: 112 / 129
- Aligned line pairs: 27902
- Mean ROUGE-1 F1: 0.5773
- Mean chrF: 0.2361
- Mean BLEU: 0.2447
- Mean edit similarity: 0.4561
- Mean length ratio: 1.0465
- Flags:
  - matched_file_coverage: observed 0.8682, target >= 0.90
- Missing round-trip files: 17

### 后汉书

- Matched files: 95 / 100
- Aligned line pairs: 25501
- Mean ROUGE-1 F1: 0.5458
- Mean chrF: 0.2077
- Mean BLEU: 0.2265
- Mean edit similarity: 0.4255
- Mean length ratio: 1.1359
- Flags: none
- Missing round-trip files: 5

### 晋书

- Matched files: 39 / 47
- Aligned line pairs: 22059
- Mean ROUGE-1 F1: 0.4848
- Mean chrF: 0.1874
- Mean BLEU: 0.2184
- Mean edit similarity: 0.3733
- Mean length ratio: 1.1504
- Flags:
  - matched_file_coverage: observed 0.8298, target >= 0.90
  - mean_length_ratio: observed 1.1504, target between 0.90 and 1.15
- Missing round-trip files: 8

### 梁书

- Matched files: 47 / 56
- Aligned line pairs: 12781
- Mean ROUGE-1 F1: 0.5116
- Mean chrF: 0.1932
- Mean BLEU: 0.2173
- Mean edit similarity: 0.3921
- Mean length ratio: 1.1096
- Flags:
  - matched_file_coverage: observed 0.8393, target >= 0.90
- Missing round-trip files: 9

### 汉书

- Matched files: 107 / 112
- Aligned line pairs: 37557
- Mean ROUGE-1 F1: 0.5180
- Mean chrF: 0.1978
- Mean BLEU: 0.2148
- Mean edit similarity: 0.3966
- Mean length ratio: 1.1324
- Flags: none
- Missing round-trip files: 5

## Output Files

- Corpus inventory: `/Users/aojiang/nltk_data/taiyichen/Classical-Chinese-Summarization/data/processed/evaluation/all_round_trip/corpus_inventory.csv`
- Corpus metrics: `/Users/aojiang/nltk_data/taiyichen/Classical-Chinese-Summarization/data/processed/evaluation/all_round_trip/corpus_metrics.csv`
- Line metrics: `/Users/aojiang/nltk_data/taiyichen/Classical-Chinese-Summarization/data/processed/evaluation/all_round_trip/line_metrics.csv`
- JSON summary: `/Users/aojiang/nltk_data/taiyichen/Classical-Chinese-Summarization/data/processed/evaluation/all_round_trip/summary.json`
