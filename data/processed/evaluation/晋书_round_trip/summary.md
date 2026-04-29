# 晋书 Round-Trip Translation Evaluation

## Overview

- Source root: `/Users/aojiang/nltk_data/taiyichen/Classical-Chinese-Summarization/data/raw/晋书`
- Back-translation root: `/Users/aojiang/nltk_data/taiyichen/Classical-Chinese-Summarization/data/processed/translated_back/晋书`
- Source files found: 47
- Back-translation files found: 39
- Matched files evaluated: 39
- Missing round-trip files: 8
- Aligned line pairs evaluated: 22059
- Unaligned source-only lines: 0
- Unaligned back-only lines: 39

## Corpus-Level Metrics

- Mean ROUGE-1 F1: 0.4848
- Mean chrF: 0.1874
- Mean BLEU: 0.2184
- Mean edit similarity: 0.3733
- Mean length ratio: 1.1504
- Exact-match rate: 0.0049
- Matched file coverage: 0.8298
- Aligned line coverage: 1.0000

## Reliability Checks

- FLAG `matched_file_coverage`: observed 0.8298, target >= 0.90. Most translated chapters should have a round-trip counterpart before we trust corpus-level averages.
- PASS `aligned_line_coverage`: observed 1.0000, target >= 0.98. Most lines should survive round-trip without alignment loss.
- FLAG `mean_chrf`: observed 0.1874, target >= 0.50. Character-level overlap should remain reasonably strong for Chinese round-trip text.
- FLAG `mean_edit_similarity`: observed 0.3733, target >= 0.55. Round-trip text should stay more similar than dissimilar at the character level.
- FLAG `mean_length_ratio`: observed 1.1504, target between 0.85 and 1.15. Reliable round-trip translation should not systematically compress or expand too much.

## Strongest Files by chrF

- `帝纪/第四章/target.txt`: chrF 0.3761, edit similarity 0.6294, aligned lines 382
- `帝纪/第九章/target.txt`: chrF 0.3317, edit similarity 0.5875, aligned lines 448
- `帝纪/第三章/target.txt`: chrF 0.3153, edit similarity 0.5693, aligned lines 636
- `志/第十九章/target.txt`: chrF 0.3138, edit similarity 0.5607, aligned lines 946
- `帝纪/第五章/target.txt`: chrF 0.2964, edit similarity 0.5675, aligned lines 420

## Weakest Files by chrF

- `志/第四章/target.txt`: chrF 0.0264, edit similarity 0.0794, aligned lines 833
- `志/第三章/target.txt`: chrF 0.0414, edit similarity 0.1228, aligned lines 1529
- `志/第五章/target.txt`: chrF 0.0438, edit similarity 0.1176, aligned lines 467
- `帝纪/第一章/target.txt`: chrF 0.0648, edit similarity 0.1385, aligned lines 472
- `列传/第十七章/target.txt`: chrF 0.0655, edit similarity 0.1426, aligned lines 296

## Missing Round-Trip Files

- `帝纪/第二章/target.txt`
- `帝纪/第六章/target.txt`
- `志/第一章/target.txt`
- `志/第七章/target.txt`
- `志/第十八章/target.txt`
- `志/第十六章/target.txt`
- `志/第十四章/target.txt`
- `志/第十章/target.txt`

## Output Files

- Line-level records: `/Users/aojiang/nltk_data/taiyichen/Classical-Chinese-Summarization/data/processed/evaluation/晋书_round_trip/line_metrics.csv`
- File-level summary: `/Users/aojiang/nltk_data/taiyichen/Classical-Chinese-Summarization/data/processed/evaluation/晋书_round_trip/file_summary.csv`
- JSON summary: `/Users/aojiang/nltk_data/taiyichen/Classical-Chinese-Summarization/data/processed/evaluation/晋书_round_trip/summary.json`
