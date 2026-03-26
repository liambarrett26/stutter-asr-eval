# Exploratory Data Analysis: Stuttered Speech Datasets

Figures are in `docs/eda/figures/`. Analysis code in `docs/eda/generate_eda.py`.

---

## 1. Dataset Summary

| Dataset | Sessions | Hours | Speakers | Words/Utts | With stutter type |
| ------- | -------- | ----- | -------- | ---------- | ----------------- |
| SLASS (curated subset) | 148 | 9.1 | 62 | 28,138 | 6,429 |
| SLASS (full archive) | 9,780 | 329.6 | — | — | — |
| UCLASS | 304 | 10.2 | 120 | 989 | 0 |
| FluencyBank | 332 | 46.9 | 113 | 61,033 | 0 |
| LibriSpeech (test+dev) | 11,126 | 21.2 | 146 | 11,126 | 11,126 (all fluent) |

Two sources of stutter type labels exist for SLASS: (1) Jason's manually annotated usage matrices (12,520 words, 68 sessions) with explicit Prolongation/Block/PWR/WWR labels, and (2) the standardised annotations extracted from SFS stutter layers (28,138 words, 132 sessions) with classifier-derived labels including a novel WWR detection method based on consecutive identical words in the orthographic layer.

![Dataset summary](figures/dataset_summary.png)

---

## 2. Disfluency Type Distribution

### Jason matrices (manual annotation, n=2,212 stuttered words)

| Type | Count | Proportion |
| ---- | ----- | ---------- |
| Prolongation | 535 | 26.1% |
| Block | 323 | 15.8% |
| PWR | 284 | 13.9% |
| WWR | 185 | 9.0% |
| Combined | 723 | 35.3% |

![Type distribution — Jason](figures/type_distribution_slass_jason.png)

### SLASS standardised (automated classification, n=1,129 disfluent words)

| Type | Count | Proportion |
| ---- | ----- | ---------- |
| WWR | 678 | 60.1% |
| Block | 156 | 13.8% |
| Combined | 135 | 12.0% |
| Prolongation | 82 | 7.3% |
| PWR | 78 | 6.9% |

![Type distribution — standardised](figures/type_distribution_slass_standardised.png)

The discrepancy between the two sources reflects methodological differences. The automated WWR detection (consecutive identical words in the orthographic layer) captures all whole-word repetitions including those that Jason's manual annotation categorised as Combined types. The Jason matrices use a finer clinical distinction where combined disfluencies (e.g., block + WWR) are labelled as Combined rather than counted under WWR alone. Both perspectives are valid and complementary.

---

## 3. Word Duration by Disfluency Type

### SLASS — Jason matrices (word-level durations from consecutive timestamps)

![KDE duration — Jason](figures/kde_duration_slass_jason.png)

Duration distributions replicate the pattern from Barrett, Tang and Howell:

| Type | n | Mean (s) | Median (s) |
| ---- | - | -------- | ---------- |
| Fluent | 9,779 | 0.60 | 0.30 |
| Prolongation | 482 | 1.25 | 0.84 |
| PWR | 262 | 1.17 | 0.89 |
| Block | 237 | 2.35 | 1.37 |
| Combined | 425 | 2.93 | 1.75 |

Fluent words peak at approximately 0.2s. Prolongations and PWR peak at 0.5s with moderate right tails. Blocks show the broadest distribution (0.5–2.0s+), consistent with variable-length silent pauses. Combined disfluencies are longest, reflecting multiple overlapping mechanisms.

### SLASS — standardised (annotation span durations)

![KDE duration — standardised](figures/kde_duration_slass_standardised.png)

Annotation spans from SFS files are longer than individual word durations because they encompass the full disfluency event including surrounding boundary markers.

### Cross-dataset comparison

![Cross-dataset duration](figures/kde_duration_cross_dataset.png)

SLASS stuttered words show a clearly broader, flatter duration distribution than SLASS fluent words. UCLASS words (mixed fluent and stuttered without type labels) show an intermediate distribution.

---

## 4. Stuttering Rate by Syllable Count

![Syllable rate](figures/syllable_rate_slass.png)

| Syllables | Total | Stuttered | Rate |
| --------- | ----- | --------- | ---- |
| 1 | 9,988 | 1,565 | 15.7% |
| 2 | 2,018 | 494 | 24.5% |
| 3 | 432 | 126 | 29.2% |
| 4 | 63 | 20 | 31.7% |

Stuttering rate doubles from monosyllabic to polysyllabic words, consistent with established findings and our previous work (Barrett, Tang and Howell). Longer words present greater motor planning demands, increasing disfluency probability.

---

## 5. Content vs Function Word Analysis

![Word type analysis](figures/word_type_slass.png)

Content words are stuttered more frequently (20.8%) than function words (14.2%). The disfluency profile differs:

- **Prolongation**: more common in content words (26.6% vs 20.3%)
- **WWR**: much more common in function words (16.5% vs 3.4%)
- **Block** and **PWR**: distributed more evenly

The elevated WWR rate in function words reflects their typical position at utterance or clause boundaries, where speakers frequently repeat short function words while preparing the following content word.

---

## 6. Most Frequently Stuttered Words

![Most stuttered words](figures/most_stuttered_words_slass.png)

The top 20 are dominated by short, high-frequency function words. "and" is stuttered at approximately 3x the rate of the next most common word ("i"), reflecting its high base frequency and typical clause-boundary position.

---

## 7. Within-Word Co-occurrence of Disfluency Types

![Co-occurrence heatmap](figures/cooccurrence_heatmap.png)

Stuttering events frequently involve multiple simultaneous disfluency types (35.3% of all stuttered words in the Jason matrices). The most common co-occurring pairs within a single word are:

| Pair | Count | Interpretation |
| ---- | ----- | -------------- |
| PWR + Prolongation | 441 | Sound is prolonged then repeated |
| Block + Prolongation | 226 | Articulatory block followed by prolonged release |
| Block + PWR | 221 | Block with partial-word repetition attempts |
| Prolongation + WWR | 64 | Less common: prolongation with whole-word repetition |
| PWR + WWR | 57 | Part-word and whole-word repetition together |
| Block + WWR | 38 | Least common pairing |

PWR + Prolongation is the dominant co-occurrence, occurring twice as frequently as any other pair. WWR is the least likely to co-occur with other types, consistent with its nature as a word-level rather than sub-word phenomenon.

---

## 8. Word-to-Word Transition Probabilities

### Full transition matrix

![Transition probability heatmap](figures/transition_probability_heatmap.png)

The transition matrix shows P(current word type | previous word type). The dominant pattern across all previous-word types is transition to fluent (68–85%), reflecting that most stuttering events are followed by recovery.

### Disfluency persistence

![Self-transition rates](figures/self_transition_rates.png)

Self-transition rate measures the probability that the same disfluency type occurs on the next word:

| Type | Self-transition rate | n |
| ---- | -------------------- | - |
| Block | 15.8% | 323 |
| Combined | 12.5% | 720 |
| Prolongation | 6.4% | 528 |
| PWR | 6.4% | 283 |
| WWR | 1.6% | 185 |

Blocks are the most persistent disfluency type. When a speaker experiences a block, there is a 15.8% probability that the following word also involves a block. This is 2.5x the persistence rate of prolongation or PWR and 10x that of WWR. Combined disfluencies also show high persistence (12.5%), suggesting that compound disfluency events tend to cluster.

WWR has the lowest persistence (1.6%), indicating that whole-word repetitions are typically isolated events that resolve after one or two repetition cycles.

### What follows each disfluency type

![Transition stacked bar](figures/transition_stacked_bar.png)

After a fluent word, 85.3% of following words are also fluent. After any disfluent word, the probability of the next word being fluent drops:

| Previous word | P(next = fluent) |
| ------------- | ---------------- |
| Fluent | 85.3% |
| Prolongation | 80.1% |
| WWR | 78.4% |
| PWR | 76.0% |
| Combined | 69.3% |
| Block | 68.1% |

Blocks and combined disfluencies create the most sustained disruptions, with only 68–69% recovery to fluency on the next word. This has direct implications for ASR systems: blocks create extended non-standard input sequences that are most likely to cause cascading errors.

### Disfluency-to-disfluency transitions

![Disfluent only transitions](figures/transition_disfluent_only.png)

When both the current and next word are disfluent, the conditional transition probabilities reveal which disfluency types tend to follow each other (excluding fluent transitions).

---

## 9. Stutter Clustering

![Run lengths](figures/stutter_run_lengths.png)

Analysis of consecutive disfluent word runs (1,513 total runs):

| Run length | Count | Proportion |
| ---------- | ----- | ---------- |
| 1 word (isolated) | 1,206 | 79.7% |
| 2 words | 202 | 13.4% |
| 3 words | 55 | 3.6% |
| 4 words | 25 | 1.7% |
| 5+ words | 25 | 1.7% |

The majority (79.7%) of disfluency events are isolated single words followed by fluent speech. However, 20.3% of events occur in clusters of 2 or more consecutive disfluent words. These clusters are critical for ASR evaluation because they represent sustained periods of non-standard acoustic input where model errors are most likely to compound.

---

## 10. Implications for ASR Evaluation

1. **Duration-based failure modes**: Disfluent words are 2–5x longer than fluent words. ASR models with aggressive endpointing may truncate stuttered words.
2. **Block persistence**: Blocks are the most persistent disfluency type (15.8% self-transition). ASR systems will encounter sustained sequences of blocked speech that create extended silence or non-standard input, likely triggering hallucinations or segment abandonment.
3. **Co-occurring disfluency types**: 35.3% of stuttered words involve multiple simultaneous types (most commonly PWR + Prolongation). These compound events create acoustic signals that deviate from expected patterns in multiple ways simultaneously.
4. **Function word vulnerability**: High-frequency function words stutter at 14.2%, potentially confusing language model priors that expect these words to be produced fluently.
5. **Clustering effects**: 20.3% of disfluency events occur in clusters of 2+ consecutive words. The co-dependency analysis (Phase 4) should examine whether ASR error rates are elevated within these clusters compared to isolated disfluencies.
6. **Syllable complexity confound**: Longer words stutter more (15.7% for 1-syllable to 31.7% for 4-syllable) and may also produce more ASR errors on fluent speech. Analyses must control for word complexity.
7. **Type-specific ASR failure modes**: The distinct acoustic profiles of each disfluency type (prolongation = extended formant, block = silence, PWR = rapid sub-word repetition, WWR = whole-word repetition) likely interact differently with encoder and decoder components. The mechanistic analysis (Phase 5) should test these interactions directly.
