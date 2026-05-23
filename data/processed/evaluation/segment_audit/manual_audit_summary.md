# segmented_v2 Manual Audit

## Scope

- Audit target: `data/segmented_v2`
- Sample size: 40 chunk-summary pairs
- Sampling method: fixed-seed random sampling, 8 pairs per corpus across:
  - `史记`
  - `后汉书`
  - `晋书`
  - `梁书`
  - `汉书`
- Pair source file: `segment_sum.txt`
- Audit file: `manual_audit.csv`

## Evaluation Criteria

For each pair, we checked:

- `Coverage`: how much of the summary can be supported by the chunk
- `Alignment`: whether the summary actually matches the current chunk
- `Better chunk`: whether another chunk is clearly a better match
- `Multi-chunk`: whether the summary sentence really needs multiple chunks
- `Too long / weird`: whether the pair should be filtered out for training

## Overall Result

- `keep`: 14 / 40
- `borderline`: 1 / 40
- `drop`: 25 / 40

This means only about `35%` of the sampled pairs look clean enough to use directly as mT5 training pairs, with another `2.5%` marginal case. The remaining `62.5%` should be excluded unless the alignment pipeline is improved.

## Distribution of Issues

- `coverage = high`: 14
- `coverage = medium`: 8
- `coverage = low`: 18

- `alignment = yes`: 14
- `alignment = partial`: 16
- `alignment = no`: 10

- `better_chunk = yes`: 18
- `better_chunk = maybe`: 8
- `better_chunk = no`: 14

- `multi_chunk = yes`: 16
- `multi_chunk = no`: 24

## By Corpus

- `史记`: 3 keep, 5 drop
- `后汉书`: 3 keep, 5 drop
- `晋书`: 1 keep, 7 drop
- `梁书`: 4 keep, 1 borderline, 3 drop
- `汉书`: 3 keep, 5 drop

`晋书` performed worst in this sample, mainly because many selected summaries from `志` and some `列传` were broad, synthetic, or encyclopedic, while the assigned chunk only covered one subsection.

## Main Failure Modes

### 1. Summary spans multiple chunks

Many summaries are not local paraphrases of one chunk. Instead, they compress a long biographical or annalistic sequence.

Representative cases:

- `史记/七十列传/田儋列传` pair 4
- `史记/七十列传/汲郑列传` pair 5
- `后汉书/列传/刘玄刘盆子列传` pair 9
- `汉书/纪/昭帝纪` pair 40

These are poor mT5 training pairs because the model would be trained to summarize information not actually present in the input chunk.

### 2. Wrong chunk selected

Several pairs were clearly misaligned: the summary is about one person or one section, but the chunk belongs to another.

Representative cases:

- `后汉书/列传/刘赵淳于江刘周赵列传` pair 14
- `后汉书/列传/文苑列传下` pair 16
- `晋书/列传/第七章` pair 18
- `梁书/列传/卷五十五` pair 30
- `梁书/列传/卷二十二` pair 32

These should definitely be filtered out.

### 3. Encyclopedic summaries paired with narrow chunks

This problem is common in `志` material. The summary often gives a broad overview, but the chunk only contains the introduction or one subsection.

Representative cases:

- `晋书/志/第十五章` pair 21
- `晋书/志/第八章` pair 24
- `汉书/传/西域传上` pair 36
- `汉书/传/叙传下` pair 37

These are not noisy because the summaries are wrong. They are noisy because they are too global for the chosen chunk.

## Positive Cases

The good pairs tend to have one common property: the summary stays local and the chunk is centered on the same event/person segment.

Representative clean cases:

- `史记/七十列传/孟尝君列传` pair 1
- `史记/七十列传/佞幸列传` pair 2
- `史记/七十列传/廉颇蔺相如列传` pair 6
- `后汉书/列传/光武十王列传` pair 12
- `后汉书/列传/文苑列传上` pair 15
- `晋书/列传/第十四章` pair 23
- `梁书/列传/卷五十` pair 26
- `梁书/列传/卷二十一` pairs 28 and 31
- `汉书/传/司马相如传上` pair 34
- `汉书/传/叙传上` pair 35
- `汉书/传/萧望之传` pair 39

## Conclusion for mT5 Training

`segmented_v2` is **not yet reliable enough to use directly as full-scale mT5 training data**.

It is better understood as a candidate pool that still needs filtering.

### Practical recommendation

Use only pairs that satisfy all of the following:

- `coverage = high`
- `alignment = yes`
- `better_chunk = no`
- `multi_chunk = no`
- `too_long_or_weird = no`

Under this stricter standard, the audited sample suggests a usable precision around `35%`.

### What this implies

- If you train on all of `segmented_v2` as-is, a large portion of the supervision signal will be noisy.
- If you filter aggressively, the remaining subset can still be valuable.
- The biggest improvement opportunity is not language cleanup, but better chunk-summary alignment.

## Recommended Next Step

Before using this as training data, add an automatic filter that drops pairs when:

- the summary is too global for a single chunk
- the assigned chunk is not the most obviously relevant one
- the chunk belongs to the wrong biography / wrong subsection
- the pair comes from `志`-style material with broad synthetic summaries unless a stricter local alignment method is used
