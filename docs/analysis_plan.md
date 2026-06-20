# Analysis & Design Plan

Pre-registration-style design document for the stuttered-speech ASR study.
Status: **draft, pre-data** (written before the full benchmark run). The
purpose is to fix the experimental design, primary contrasts, and statistical
model *before* looking at model outputs, so the headline claims are
confound-controlled and not the product of post-hoc choices. The `todo.md`
roadmap lists what *can* be done; this document commits to what *will* be done
for the first paper, and how.

Companion docs: `docs/evaluation_harness.md` (harness design),
`docs/running_models_linux.md` (execution), the manuscript draft.

---

## 1. Venue and framing

Target: **Computational Linguistics** (primary). This is a linguistics-literate,
methodologically exacting venue — not an engineering-benchmark venue. The
framing therefore foregrounds the *linguistic and mechanistic structure* of ASR
failure, not a leaderboard:

- the **structured mapping** between disfluency type and ASR error type
  (`P(error_type | stutter_type)`) — a linguistic regularity, not just a metric;
- the **reference-convention methodology** (surface vs intended) and its effect
  on model rankings — a measurement contribution;
- **mechanistic explanation** (attention / probing / CTC analysis) of *why* the
  encoder–decoder vs CTC gap arises;
- **calibration against a human inter-rater ceiling** (UNWR reliability, PER).

The architectural-recommendation / fine-tuning / augmentation material (todo
Phase 6) is **out of scope for Paper 1** and deferred to Paper 2. (The abstract
and todo Phase 9 currently name Nature Machine Intelligence / TASLP — reconcile
to CL before submission.)

## 2. Scope of Paper 1

In scope: benchmark (H1), co-dependency (H2 descriptive), severity
stratification (H3), phoneme-level human-ceiling calibration (H4), surface-vs-
intended methodology, and the mechanistic probing that explains the
architecture gap. Out of scope (Paper 2): decoder/endpoint tuning, fine-tuning
(LoRA/full/personalised), data augmentation, architectural redesign.

## 3. Hypotheses / research questions

- **H1** ASR error rate is higher on stuttered than fluent speech, and the gap
  varies by architecture (encoder–decoder ≪ CTC expected). *Headline.*
- **H2** The gap is *structured by disfluency type* — `P(error | stutter_type)`
  is non-uniform — and the structure is mechanistically explicable in model
  internals at the word/event level.
- **H3** Error rate scales with stuttering severity (mild/moderate/severe), with
  architecture-dependent robustness.
- **H4** Phoneme-level ASR error (PER) relative to the *human inter-rater
  ceiling* on the same items (UNWR reliability: gold ≈5% PER, naive ≈25%).

H2–H4 are nested under H1.

## 4. Identification strategy and confound control (critical)

The naive contrast "stuttered corpora vs LibriSpeech" is **confounded**:
LibriSpeech is American, adult, *read*; the stuttered data is largely British,
child, *spontaneous*. A raw gap could reflect dialect × age × register, not
stuttering. The design controls this as follows.

- **Primary contrast = within-corpus stuttered vs matched control.** FluencyBank
  **UMD-CMU** provides 15 CWS + 15 matched fluent controls (same cohort,
  protocol, register, recording conditions). This is the cleanest "stuttering,
  holding everything else fixed" comparison and carries the causal claim.
- **External anchors (secondary).** LibriSpeech (fluent American adult read) and
  the public stuttered corpora anchor absolute WER to the literature, but are
  *not* the basis for the causal stutter claim.
- **Confound covariates carried throughout:** dialect (British/American), age
  band, register (read/spontaneous/conversation), corpus. These enter the
  statistical model as fixed effects (§7) rather than being averaged over.
- **Severity (H3)** uses the Jason gold stuttering rate (46 severe / 15 moderate
  / 7 mild); the `slass_full` auto-labels under-detect and are not used as
  severity ground truth.

Corpus roles:

| Corpus | Role |
|---|---|
| FluencyBank UMD-CMU (CWS + controls) | **Primary** within-corpus stutter contrast |
| SLASS (slass_full, British child) | Primary stuttered WER base; H2/H3 engine (Jason labels) |
| UCLASS | Stuttered, British; external |
| FluencyBank Voices-AWS/CWS | Stuttered, American; external |
| LibriSpeech | Fluent American adult read; external ceiling anchor only |
| UNWR reliability | H4 phoneme PER + human ceiling |
| UNWR (adult SSI) | Adult severity context |

## 5. Conditions and references

Every unit scored against **both** references, reported separately:

- **Surface** — disfluencies preserved as produced.
- **Intended** — normalised fluent target.

Rankings can invert between the two; we report both and quantify inversions
(todo Phase 7). Normaliser: Whisper `EnglishTextNormalizer`, applied identically
to reference and hypothesis; filled pauses retained for surface, removed for
intended. Normalisation-sensitivity is a secondary analysis.

## 6. Models

- **Open-source (primary, all probe-able):** whisper-large-v3 / medium / small,
  faster-whisper-large-v3, wav2vec2-large, hubert-large. Greedy decoding,
  temperature 0, language forced to English.
- **Commercial APIs (secondary, governance-restricted — see §9):** Google,
  Azure, Amazon, AssemblyAI, Deepgram — **public corpora only**.

Inference is decoupled from scoring: `run_asr.py` saves raw hypotheses *before*
normalisation, so metrics and references can be added/changed without re-running
models. This is why the full open-source run can proceed in parallel with
finalising this plan.

## 7. Statistical analysis

The data are **clustered and longitudinal** (repeated recordings per speaker,
speaker imbalance, multiple corpora). Per-utterance independence tests are
therefore inappropriate as the primary inference.

- **Primary model:** mixed-effects regression of the error metric with
  **random intercepts for speaker** (and corpus), fixed effects for condition
  (stuttered/fluent), severity, architecture, and the confound covariates
  (dialect, age band, register). For WER, model error counts with an appropriate
  link (e.g. beta/binomial on error proportion) rather than raw WER as Gaussian.
- **WER system comparison:** **MAPSSWE** (NIST matched-pairs) as the
  field-standard significance test for paired system WER, alongside the
  mixed-effects estimates.
- **Uncertainty:** paired bootstrap 95% CIs (10,000 resamples) on the
  stuttered−fluent gap (already in `score.py`).
- **Multiple comparisons:** Bonferroni/Holm across system pairs; report
  **effect sizes (Cohen's d / odds ratios)** beside p-values.
- Speaker-exclusive handling throughout; no speaker spans contrasted groups.

## 8. Metrics (Paper 1)

- WER, CER — micro + macro, both references *(in `score.py`)*.
- **BERTScore** F1 (DeBERTa-xlarge-mnli) — meaning preservation *(to add)*.
- **Hallucination rate** (I/N), per stutter type *(to add)*.
- **Co-dependency** `P(error_type | stutter_type)` — needs word-level
  reference↔hypothesis alignment *(to add)*.
- **PER vs human ceiling** (H4) on UNWR reliability *(to build)*.

## 9. Governance constraints (DPIA)

- The DPIA permits processing **outside the EU only on de-identified, GDPR-
  compliant data**. Speech is inherently identifying biometric data and cannot
  be meaningfully de-identified. Therefore **commercial-API evaluation must not
  include SLASS** (or any non-public identifiable audio); APIs run on the
  already-public corpora (FluencyBank/UCLASS under their terms) only.
- Model outputs and derived data live on the governed store (RDSS) / the local
  working copy; no identifiable audio leaves approved infrastructure.
- Commercial APIs: send only short segments, disable retention where offered.

## 10. Known caveats

- **Training contamination:** LibriSpeech is in model training data (best-case
  fluent ceiling — conservative for the gap). FluencyBank/UCLASS are public and
  may also be in web-scale training (e.g. Whisper); flag when interpreting their
  absolute WER.
- **Reference-label reliability:** the SLASS stutter-type labels (Jason
  matrices) underpin H2; their provenance/agreement is stated as a limitation.
- **Chunking:** CTC models chunk long audio at fixed windows (slightly
  pessimistic at boundaries); the manuscript's VAD-segmentation claim must be
  either implemented (Silero) or removed to match the harness.

## 11. Open design decisions (resolve before freezing results)

Tracked in `todo.md` → "Design decisions / open methodological risks". Each must
be settled before the corresponding result is reported:

1. Confirm UMD-CMU matched-control as the primary contrast; finalise covariate set.
2. Lock the mixed-effects specification (link function, random-effects structure).
3. Elevate or descope H4 (phoneme PER vs human ceiling) — recommend elevate.
4. Confirm commercial-API restriction to public corpora (governance).
5. Resolve manuscript↔harness drift: VAD segmentation, BERTScore, hallucination,
   word-alignment.
6. Confirm Paper 1 / Paper 2 scope boundary.
7. Contamination check for public stuttered corpora.

## 12. Reproducibility

Raw hypotheses (pre-normalisation), manifests, scoring configs, and seeds
retained on the governed store. Code released with the paper. Data availability
per corpus terms (UCLASS public; FluencyBank via TalkBank; SLASS on request).
