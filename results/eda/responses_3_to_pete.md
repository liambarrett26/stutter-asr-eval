# Responses to Pete's EDA comments (round 2)

Replies to `responses_2_slass_eda.md`, page by page. Sections marked
**[fix]** are analysis bugs or write-up issues I can act on now. Sections
marked **[new]** are new analyses that need a follow-up pass. Sections
marked **[Jason]** sit with Jason for annotation review.

---

## Page 1 — SLASS full archive row of the summary table

**[fix]** Reasonable. Of the 9,961 SFS files in the full archive, **1,122
filenames carry a parseable participant code** (formats like
`f_0050_7_11y6m`, `M_0017_8y9m_1`, `m_0014_7_10y10m`); these yield
**255 unique speaker IDs spanning the range 11–2410**. The remaining
8,839 files use schemas without a recoverable ID (e.g. `0050c116.sfs`,
`0017c105.sfs`) — the `c###` / `d###` suffix is presumably an index into
a master spreadsheet on the speech-lab Windows store. I will surface
those in the next pass once the egress lands (the
`Participant information-raw data.xls` master rosters are in the egress
manifest).

The summary-table cell will read:
**"≥255 speakers (minimum, derived from filename participant codes;
8,839/9,961 files use opaque naming and need cross-reference to the
master roster to count)"**.

I will also re-extract from the matched Jason matrices to confirm the
overlap, and revise once the c-suffix mapping is in.

---

## Page 2 — Disfluency type distribution, Jason matrices

**[Jason]** Agreed that WWR=9.0% in the Jason table looks low for a
manually-annotated reference. Worth Jason double-checking whether all
WWRs are being labelled (some may end up under Combined because of
sound-onset prolongation of the repeated word). Pure-WWR count in the
matrices is 185 (9.0%), but WWR also appears as a co-occurring tag in
83 further events (PWR+WWR 18, Block+WWR 15, Prolongation+WWR 24,
WWR+PWR 2, WWR+Prolongation 2, and 22 triplet/quadruplet events). If
Jason is content that those are correctly tagged as Combined, the 9%
is right; if some "Combined" ought to be pure WWR, the count moves up.

---

## Pages 3–4 — Disfluency type distribution, SLASS standardised

**[done in code, awaits re-run]** Pete's diagnosis is correct, and his
follow-up question — "why do we need this lookback at all when WWRs and
PWRs are already labelled?" — was the right one. After looking at the
actual stutter-layer encoding:

- `{xN}` markers in a stutter-layer label mark sub-phoneme repetition
  inside one segment — these are PWRs. (206 events in SLASS.)
- Space-separated repeated tokens within one stutter-layer label
  (e.g. `'AA AA AA'`, `'I I'`) mark whole-fragment repetition encoded
  as a single annotation span — these are WWRs. (79 events in SLASS.)
- The 60.1% WWR figure came from the standardiser's *additional*
  lookback over the orthographic layer that flagged any adjacent
  duplicate ortho tokens as WWR. That over-fired on fluent repetitions
  (list reading, backchannel "yeah yeah", sentence restarts).

Decision (Pete, confirmed): **drop the ortho-layer lookback entirely
and trust the stutter layer.** Implemented in `src/data/standardise.py`
in three edits:

1. `classify_sfs_stutter` now returns `pwr` for `{xN}` and `wwr` for
   space-separated repeats, rather than both being called "repetition".
2. The `wwr_timestamps` ortho-layer detector and its usage are removed.
3. The `.replace("repetition", "pwr")` rewrite is removed.

Projected post-re-run distribution at the stutter-layer-event level
(qualitative — final figures need the pipeline to re-run):

| Type | Old | Projected new |
|---|---:|---:|
| WWR | 60.1% | ~2% |
| Block | 13.8% | ~11% (largest single type) |
| PWR | 6.9% | ~0.8% |
| Prolongation | 7.3% | ~0.9% |
| Combined | 12.0% | ~1.6% |

Known limitation we accept: WWRs that the annotator spread across
multiple consecutive annotation entries (rather than collapsing them
into one annotation span) are not detected. This is a deliberate trade
in favour of trusting the annotator and avoiding the over-fire.

---

## Pages 5–6 — Word duration by disfluency type

### WWR missing from the Jason duration table — bug confirmed

**[fix]** When I dug in, WWR rows in the Jason matrices have a
**different timestamp format** from the other rows:

- Fluent/Prolongation/Block/PWR rows: single onset timestamp,
  e.g. `2.08580`. The current EDA computes duration as the gap to the
  *next* row's onset.
- WWR rows: **start | end** pair, e.g. `78.92902 | 79.33220`. The
  current EDA's `float(ts)` call silently fails to parse the `start | end`
  string and drops the row.

Result: **0 of 290 WWR-containing rows contribute to the KDE**,
explaining why WWR is absent from the duration table on page 5. The
fix is one line in `generate_eda.py` — detect the `|` separator and
parse `end − start` directly. Pure-WWR durations should then drop into
the same range Pete shows from the Frontiers paper (whole-word
repetition events of ~0.4–1.2 s). I'll re-issue the KDE.

### Standardised durations are longer than Jason durations — clearer explanation

**[fix]** Pete is right that "encompass the full disfluency event
including surrounding boundary markers" was hand-wavy. Concretely:

- **Jason matrices** time *each word individually* via the inter-onset
  interval (i.e. the duration field for word *i* is
  `Timestamp[i+1] − Timestamp[i]`). For a long block followed by speech
  resumption, this captures essentially just one word's slice of time.
- **SFS-derived standardised durations** time *the entire annotation
  span* from the stutter layer. A block annotation in SFS commonly
  covers the silent pre-articulation interval *plus* the word being
  produced *plus* any post-block prolongation, marked from the start of
  the silent block to the end of the recovered speech. A PWR
  annotation typically covers the whole repetition cluster
  (`the the the`) as one event rather than tagging each fragment
  separately.

So the standardised set is measuring something genuinely longer (the
event), not the per-token articulation. WWR durations are roughly
comparable across the two because the Jason matrices treat each repeated
token as its own row, but the standardised set treats the cluster as a
single event — so the standardised WWR distribution actually shows the
event span (longer) while the corrected Jason WWR distribution will
show the within-cluster token span (shorter).

I'll re-word the write-up accordingly and, in the corrected figure,
plot the two duration definitions side by side so the apples-to-oranges
nature is explicit.

---

## Page 7 — Cross-dataset comparison

Pete: "fluent words are long". Reading the figure again, I read SLASS
fluent peaking around 0.3 s, with a long tail out to ~1.5 s. That tail
is largely **utterance-final words and tokens preceding an inter-utterance
pause** — the inter-onset duration captures the silent gap as part of
the preceding word's "duration". For a fair fluent-vs-stuttered
comparison I should either (a) clip durations at a percentile to suppress
the pause tail, or (b) compute durations from a forced-aligner rather
than inter-onset. (a) is a one-line change; (b) is the proper fix and
something I can do once we have alignments. I'll switch to (a) for the
immediate redo and flag (b) as a follow-up.

---

## Page 8 — Stuttering rate by syllable count

**[new — initial result below]** Pete's hypothesis is broadly supported.
Breakdown computed live from the Jason matrices, expressed as
(stuttered words of type T) / (total words at that syllable count):

```
syl  total   prol     block    pwr      wwr      comb
  1   9988  3.9%     2.3%     1.9%     1.6%     4.6%
  2   2018  5.8%     3.7%     3.2%     1.0%     9.6%
  3    432  5.6%     3.2%     5.3%     0.5%    12.5%
  4     63  3.2%     6.3%     7.9%     0.0%    14.3%
```

Predictions confirmed:

- **WWR drops monotonically** from 1.6% (1-syl) to 0% (4-syl) — whole-word
  repetition is concentrated at the short, function-word end.
- **Combined events climb** from 4.6% → 14.3% — multi-mechanism
  disfluencies emerge at longer words, consistent with longer motor
  plans giving more opportunities for failure.
- **Block and PWR also climb** with syllable count.
- **Prolongation peaks at 2 syllables** then declines — slightly
  surprising; possibly an artefact of how prolongation onset interacts
  with the first syllable of polysyllabic words.

I'll add this as a new figure (stacked bar or grouped bar) and the
table. This is exactly the analysis Pete asked Jason to take forward
along the PSal lines — the data already supports the framing.

---

## Page 9 — Content vs function word analysis with age effects

**[new]** Two requests, both doable:

1. **Add age effects.** SLASS filenames with the `XyXm` pattern give
   age-at-recording for 1,122 of the 9,961 sessions. I'll bin into
   age bands (e.g. 8–10, 10–12, 12–14, 14–18, 18+) and produce the
   content-vs-function panel per band. Sessions without parseable
   ages will be reported under an "unknown / likely adult" band with
   a caveat that subrange granularity is missing — Pete's suggested
   approach.
2. **Show WWR in the right-hand panel.** The current figure's
   right panel uses 5 stutter types but Pete's right — the WWR
   16.5%-vs-3.4% asymmetry I quote in prose isn't actually visible in
   the bars because WWR is barely represented. Fixing this needs the
   WWR-duration bug fix above (so WWR events show up consistently) and
   a re-render of the figure.

---

## Page 10 — Most frequently stuttered words

**[new + Jason]** Two additions:

1. I can produce a stacked version: top 20 stuttered words coloured by
   stutter type (Prol / Block / PWR / WWR / Combined). My expectation
   matches Pete's: function words at the top will be WWR-dominant
   ("and", "i", "the", "a", "to"); content words further down will mix
   Prol/Block/PWR.
2. Word-type designation (Content / Function) is already in the Jason
   matrices (`Word_Type` column). I'll annotate each bar accordingly so
   the visual cue is explicit, matching the "and = Function" note we
   currently have only at the bottom of the figure caption.

Jason — if you can sanity-check the Word_Type assignments on the top-20
words once the figure exists, that closes this off.

---

## Pages 11–12 — Within-word co-occurrence

**[fix to the write-up]** Pete is right and our results.md is wrong on
ordering. The Stutter_Type field **does encode order**:

| ordered form | n |
|---|---:|
| `Prolongation + PWR` | 233 |
| `Prolongation + Block` | 85 |
| `Block + PWR` | 84 |
| `PWR + Prolongation` | 73 |
| `Prolongation + Block + PWR` | 35 |
| `Prolongation + WWR` | 24 |
| `Block + Prolongation` | 20 |
| `Prolongation + PWR + Block` | 19 |
| `Block + Prolongation + PWR` | 19 |
| `PWR + WWR` | 18 |
| ... | |

Our results.md collapsed these unordered ("PWR + Prolongation 441")
which is what Pete spotted as inconsistent — the dominant form is
**Prolongation-then-PWR (233)**, not PWR-then-Prolongation (73). So the
"sound is prolonged then repeated" description was actually correct for
the dominant ordering, but the *label* on the row was wrong (the label
read "PWR + Prolongation"). Two fixes:

1. Re-issue the table separating each ordered pair / triplet rather
   than collapsing.
2. Re-do the co-occurrence heatmap as a directed transition matrix
   (Prolongation → PWR ≠ PWR → Prolongation) so the ordering is visible.

For Pete's nasal hypothesis (page 11–12, first three lines):
**[Jason]** — Jason, this is the bit where you can lift the manner-class
labels from the IJCLD manner paper and join them in. Three specific
sub-analyses:

1. For Prolongation→PWR events (n=233), what proportion of the
   first phoneme of the affected word is nasal vs other continuant
   (fricative)? Pete's prediction: nasals over-represented because they
   afford both prolongation and obstruction.
2. For Block→Prolongation events (n=85 collapsed across the two ordered
   forms, 20 in `Block+Prol` order, 85 in `Prol+Block` order — sorry,
   I mean: Pete's "block to pro" case is `Block + Prolongation` at
   n=20 ordered): same manner breakdown, expectation again of nasal
   over-representation.
3. For Block→PWR events (n=84 in `Block + PWR`, 17 in `PWR + Block`):
   compare manner-class distribution of the affected word's onset to
   that of pure Blocks and pure PWRs.

If results match the manner paper, we have a clear acoustic-articulatory
mechanistic story to tell in a follow-up.

Also: the Stutter_Type_2 column appears to record the **first** stutter
type of an ordered event (e.g. `Prolongation + PWR` → ST2 =
`Prolongation`). Useful as a convenience field but worth Jason confirming
that interpretation against the annotation manual.

---

## Page 13 — Run lengths / clustering

**[fix to wording]** Agreed — these are fluency-type transitions, not
"word-to-word transitions". I'll rename across the figure titles,
caption text, and results.md.

On the contradiction with Yairi: I recall the Yairi & Ambrose data
showing clustering at the *stuttered-event* level rather than the
fluent-vs-disfluent word level. Our run-length analysis treats any
disfluent word as one unit, which collapses across event types. A
follow-up could:

- separate "block runs" from "prolongation runs" etc. to see whether
  same-type clustering is stronger than mixed-type clustering
  (likely yes per the self-transition data on page 14);
- compare the empirical run-length distribution against a Poisson null
  (independent events) and a contagion null (Yairi).

Jason mentioned writing something on this previously; if that draft
exists it would be useful to recover.

---

## Page 14 — Persistence

**[new]** Pete's question — is block→block persistence inflated by the
fact that the speaker's next disfluent word is on a phoneme that also
elicits a block? — is testable. Plan:

1. For each block event, identify the **next disfluent word** and its
   first phoneme's manner class.
2. Compute P(next = block | first phoneme is obstruent),
   P(next = block | first phoneme is continuant nasal),
   P(next = block | first phoneme is non-nasal continuant), etc.
3. Compare against a control: the same conditional probabilities
   computed on all disfluent words irrespective of preceding type.

If block→block falls when we condition out manner, the persistence is
mostly manner-driven; if it survives conditioning, there is a genuine
within-speaker temporal persistence effect.

For the "half-life" question: I can compute the conditional probability
of a same-type stutter at lag k = 1, 2, 3, ... disfluent events away.
If persistence has a finite half-life we'll see a roughly exponential
decay; if it's a one-shot effect, the autocorrelation drops to
chance after lag 1.

---

## Page 15 — What follows each disfluency type

No action — Pete approved this section.

---

## Page 16 — Stutter-type-to-stutter-type transitions

**[fix to wording]** Renaming as suggested ("stutter type to stutter
type transitions" rather than the word-to-word framing). I'll also lift
the off-diagonal cells out for emphasis since most of the mass is on
the diagonal.

---

## Page 17 — Implications: sustained dysfluency

**[new]** Pete wants more target-and-type information here. After the
page-11/12, page-13 and page-14 follow-ups land, I'll redraft the
implications around (a) sustained block clusters and their manner
profile, (b) Prolongation→PWR as the dominant ordered co-occurrence
and what that means for ASR encoder smoothing, and (c) the corrected
WWR position relative to function words.

---

## Page 18 — General implications

**[fix]** Will be revised once the items above land — particularly the
WWR corrections (Pages 5–6 and 9), the directed co-occurrence story
(Pages 11–12), and the manner-conditioned persistence analysis
(Page 14).

---

# Suggested next-action list (for me)

1. Fix the WWR timestamp parser bug in `generate_eda.py` so WWR
   appears in the duration KDE and the type-by-syllable analysis.
2. Tighten the standardised-set WWR detector (require co-occurring
   stutter-layer mark or short temporal window).
3. Re-render duration KDEs with a clearer "event span vs
   inter-onset duration" framing in the caption.
4. Add stutter-type breakdown to the syllable-count and most-stuttered
   figures.
5. Add age-banded content-vs-function analysis.
6. Re-do co-occurrence as a directed-transition figure with the
   ordered pair / triplet table.
7. Rename "word-to-word" to "stutter-type-to-stutter-type" throughout.
8. Add Speaker-count estimate to the dataset summary row for SLASS
   full archive (≥255 + caveat).

# Action list (for Jason)

1. Confirm whether all WWRs in the matrices are labelled, or whether
   some are tucked under Combined; reconcile with Pete's 9% concern.
2. Manner-class breakdown of the dominant ordered co-occurrences
   (Prolongation → PWR; Block → Prolongation; Block → PWR), as
   discussed against the IJCLD manner paper.
3. Confirm interpretation of the Stutter_Type_2 column (= first
   stutter type in an ordered event).
4. Surface any prior write-up on stutter clustering vs Yairi.

# Action list (for Pete)

No specific blockers remaining. The standardised-WWR-detector
tightening (page 3–4) has been agreed: drop the ortho-layer lookback,
trust the stutter layer, classify `{xN}` as PWR and space-separated
repeats as WWR. Code in `src/data/standardise.py` updated accordingly;
awaits re-run of the standardisation pipeline against the SFS sources.
