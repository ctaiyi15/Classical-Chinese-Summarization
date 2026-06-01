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
| EN -> EN clean v4 epoch10 sampling | 0.266 | 0.066 | 0.157 | 0.390 | 0.114 | 0.174 |
| CC -> EN clean v4 epoch10 sampling | 0.236 | 0.030 | 0.136 | 0.382 | 0.078 | 0.155 |
| EN -> EN clean v6 epoch10 sampling | 0.267 | 0.070 | 0.168 | 0.441 | 0.129 | 0.188 |
| EN -> EN clean v6 epoch10 beam4 | 0.268 | 0.094 | 0.187 | 0.394 | 0.145 | 0.204 |

The direct CC -> EN baseline trails the EN -> EN baseline when both use the same
decoding setup, which is expected because CC -> EN combines classical Chinese
understanding, translation, and summarization in one step.

Sampling improves both pipelines substantially, especially article-level
ROUGE-1/2. The generation stats also show that the rough repetition signals drop
to near zero. However, both sampling runs generate much shorter outputs
(`64.7` words for EN -> EN and `66.6` words for CC -> EN, versus `127.3` gold
words), so the ROUGE gains should be checked qualitatively for coverage and
possible under-generation.

## Segmented v4 Clean Sampling Runs

`data/segmented_v4` contains 409 `segment_sum.txt` files. Unlike v2, it does not
use `segment_sum_concat.txt`; each file contains `Summary line`, `Span`,
`original`, `translation`, and `summary` sections. The preparation scripts now
auto-detect both formats.

The first v4 sampling runs accidentally included separator-only lines such as
`=======` in the training targets. Those contaminated runs should not be cited as
final results. The preparation scripts now strip separator-only lines from source
and target text.

Prepared clean v4 data:

- Raw pairs: `2578`
- Removed long-summary pairs: `56`
- Grouped samples: `2522`
- Train/test articles: `338 / 60`
- Train/test samples: `2163 / 359`
- Average target length: `105.1` words

EN -> EN clean v4 epoch10 sampling:

- Generations: `outputs/mt5-en2en-segmented-v4-clean-epoch10-sampling`
- Chunk-level ROUGE-1/2/L: `0.266 / 0.066 / 0.157`
- Article-level ROUGE-1/2/L: `0.390 / 0.114 / 0.174`
- Empty generations: `0`
- Outputs containing `<extra_id`: `0`
- Average generated length: `48.0` words
- Rough repeated sentence count: `0`
- Rough repeated trigram count: `1`

CC -> EN clean v4 epoch10 sampling:

- Generations: `outputs/mt5-cc2en-segmented-v4-clean-epoch10-sampling`
- Chunk-level ROUGE-1/2/L: `0.236 / 0.030 / 0.136`
- Article-level ROUGE-1/2/L: `0.382 / 0.078 / 0.155`
- Empty generations: `0`
- Outputs containing `<extra_id`: `0`
- Average generated length: `55.8` words
- Rough repeated sentence count: `0`
- Rough repeated trigram count: `0`

Qualitative samples show that clean v4 EN -> EN generations are usually related
to the input names and events, but they still contain false or mixed details.
Clean v4 CC -> EN remains weaker and often generates fluent but poorly grounded
English.

Important caveat: v4 span windows were designed mainly for CC -> EN. Classical
Chinese is compact, but the English translations of those spans are much longer.
For EN -> EN, many sources exceed the model input length, so training can pair
partially truncated evidence with targets that refer to information outside the
visible input.

## Segmented v6 EN -> EN Runs

`data/segmented_v6` was introduced to reduce hallucination from broad v4
span-summary pairs. It splits some summary lines into smaller targets, reduces
the allowed span window, and groups spans with large overlap into one sample at
the end.

Prepared EN -> EN v6 data:

- Raw pairs: `4168`
- Removed long-summary pairs: `63`
- Grouped samples: `4105`
- Train/test articles: `345 / 61`
- Train/test samples: `3533 / 572`
- Average source length: `326.1` words
- Average target length: `66.4` words
- Separator contamination: none found

Tokenizer length audit for `google/mt5-small` with the `summarize:` prefix:

- Source median: `441` tokens
- Source mean: `502.5` tokens
- Source `>512`: `41.9%`
- Source `>768`: `21.1%`
- Source `>1024`: `2.6%`
- Target median: `84` tokens
- Target mean: `114.2` tokens

EN -> EN clean v6 epoch10 sampling:

- Generations: `outputs/mt5-en2en-segmented-v6-clean-epoch10-sampling`
- Decoding: `do_sample=True`, `temperature=0.8`, `top_p=0.9`, `top_k=50`, `no_repeat_ngram_size=3`, `repetition_penalty=1.2`
- Chunk-level ROUGE-1/2/L: `0.267 / 0.070 / 0.168`
- Article-level ROUGE-1/2/L: `0.441 / 0.129 / 0.188`
- Empty generations: `0`
- Outputs containing `<extra_id`: `0`
- Average generated length: `38.9` words
- Rough repeated sentence count: `0`
- Rough repeated trigram count: `0`

EN -> EN clean v6 beam-search rerun:

- Generations: `outputs/mt5-en2en-segmented-v6-clean-epoch10-beam4`
- Checkpoint: `outputs/mt5-en2en-segmented-v6-clean-epoch10-sampling`
- Decoding: `--generate_only`, `do_sample=False`, `num_beams=4`, `no_repeat_ngram_size=3`, `repetition_penalty=1.2`
- Chunk-level ROUGE-1/2/L: `0.268 / 0.094 / 0.187`
- Article-level ROUGE-1/2/L: `0.394 / 0.145 / 0.204`
- Empty generations: `0`
- Outputs containing `<extra_id`: `0`
- Average generated length: `28.7` words
- Rough repeated sentence count: `0`
- Rough repeated trigram count: `0`

The v6 sampling run improves article-level ROUGE over clean v4 and appears to
reduce the broad, unfocused generation problem. The beam-search rerun is shorter
and more conservative: it lowers article-level ROUGE-1 but improves ROUGE-2 and
ROUGE-L, suggesting better phrase-level precision at the cost of coverage.
