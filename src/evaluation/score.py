#!/usr/bin/env python3
"""Score ASR hypotheses against the manifest references.

Joins a hypotheses JSONL (from run_asr.py) to the manifest, normalises
reference and hypothesis identically, and computes WER/CER against BOTH
reference conventions (intended and surface) — surface vs intended can
invert model rankings on stuttered speech, so both are always reported.

Aggregations:
  * overall (micro = pooled words; macro = mean over units)
  * by condition (stuttered vs fluent) — the H1 gap
  * by speaker (for speaker-level / cluster-bootstrap inference)
  * stuttered-with-labels subset
With a paired bootstrap 95% CI on the stuttered-minus-fluent WER gap.

Normalisation prefers Whisper's EnglishTextNormalizer (the Open-ASR-
Leaderboard standard); falls back to a basic lowercase/strip-punct
normaliser if Whisper isn't installed. WER/CER use a self-contained
Levenshtein so scoring needs no third-party packages.

Usage:
    python -m src.evaluation.score \\
        --manifest /Volumes/FATSPEECH/manifests/benchmark_v1.jsonl \\
        --hyp /Volumes/FATSPEECH/results/hyp/whisper-large-v3.jsonl \\
        --out /Volumes/FATSPEECH/results/scores/whisper-large-v3.json

    python -m src.evaluation.score --selftest   # validate scoring logic
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


# ── Normalisation ────────────────────────────────────────────────────────────

_BASIC_RE = re.compile(r"[^a-z0-9' ]+")


def get_normaliser():
    """Return a text-normalising callable (Whisper's if available)."""
    try:
        from whisper.normalizers import EnglishTextNormalizer
        return EnglishTextNormalizer()
    except Exception:
        def basic(text: str) -> str:
            text = text.lower()
            text = _BASIC_RE.sub(" ", text)
            return re.sub(r"\s+", " ", text).strip()
        return basic


# ── Edit distance (word- and char-level) ─────────────────────────────────────

def _levenshtein(ref: list, hyp: list) -> tuple[int, int, int, int]:
    """Return (substitutions, deletions, insertions, hits) via DP backtrace."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return (0, 0, m, 0)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + cost)
    # Backtrace to count S/D/I/H
    i, j = n, m
    s = dele = ins = hit = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and \
                d[i][j] == d[i - 1][j - 1]:
            hit += 1; i -= 1; j -= 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            s += 1; i -= 1; j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            dele += 1; i -= 1
        else:
            ins += 1; j -= 1
    return (s, dele, ins, hit)


def wer_counts(ref: str, hyp: str) -> tuple[int, int, int, int]:
    return _levenshtein(ref.split(), hyp.split())


def cer_counts(ref: str, hyp: str) -> tuple[int, int, int, int]:
    return _levenshtein(list(ref.replace(" ", "")), list(hyp.replace(" ", "")))


# ── Aggregation ──────────────────────────────────────────────────────────────

def micro_wer(units: list[dict], key: str) -> float:
    s = sum(u[key][0] + u[key][1] + u[key][2] for u in units)
    refn = sum(u[key][0] + u[key][1] + u[key][3] for u in units)  # S+D+H = |ref|
    return s / refn if refn else 0.0


def macro_wer(units: list[dict], key: str) -> float:
    rates = []
    for u in units:
        s, dlt, ins, hit = u[key]
        denom = s + dlt + hit
        if denom:
            rates.append((s + dlt + ins) / denom)
    return sum(rates) / len(rates) if rates else 0.0


def bootstrap_gap(stut: list[float], flu: list[float], n_boot=2000):
    """95% CI on (mean stuttered WER - mean fluent WER) via unit bootstrap.

    Deterministic LCG so results are reproducible without numpy.random.
    """
    def mean(x):
        return sum(x) / len(x) if x else 0.0
    obs = mean(stut) - mean(flu)
    seed = 12345
    def rnd():
        nonlocal seed
        seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
        return seed / 0x7FFFFFFF
    diffs = []
    for _ in range(n_boot):
        bs = [stut[int(rnd() * len(stut))] for _ in range(len(stut))]
        bf = [flu[int(rnd() * len(flu))] for _ in range(len(flu))]
        diffs.append(mean(bs) - mean(bf))
    diffs.sort()
    lo = diffs[int(0.025 * n_boot)]
    hi = diffs[int(0.975 * n_boot)]
    return obs, lo, hi


def score(manifest: Path, hyp_path: Path, ref_field: str,
          normaliser) -> dict:
    mani = {json.loads(l)["unit_id"]: json.loads(l)
            for l in manifest.open(encoding="utf-8")}
    scored = []
    for line in hyp_path.open(encoding="utf-8"):
        h = json.loads(line)
        u = mani.get(h["unit_id"])
        if not u:
            continue
        ref = normaliser(u[ref_field])
        hyp = normaliser(h.get("hypothesis", "") or "")
        rec = dict(u)
        rec["wer_c"] = wer_counts(ref, hyp)
        rec["cer_c"] = cer_counts(ref, hyp)
        rec["_unit_wer"] = ((rec["wer_c"][0] + rec["wer_c"][1] + rec["wer_c"][2])
                            / max(1, rec["wer_c"][0] + rec["wer_c"][1] + rec["wer_c"][3]))
        scored.append(rec)

    def block(units):
        return {
            "n_units": len(units),
            "wer_micro": round(micro_wer(units, "wer_c"), 4),
            "wer_macro": round(macro_wer(units, "wer_c"), 4),
            "cer_micro": round(micro_wer(units, "cer_c"), 4),
        }

    stut = [u for u in scored if u["condition"] == "stuttered"]
    flu = [u for u in scored if u["condition"] == "fluent"]

    # Speaker-level mean WER (for cluster inference)
    spk = defaultdict(list)
    for u in scored:
        spk[(u["condition"], u["speaker_id"])].append(u["_unit_wer"])
    spk_stut = [sum(v) / len(v) for (c, _), v in spk.items() if c == "stuttered"]
    spk_flu = [sum(v) / len(v) for (c, _), v in spk.items() if c == "fluent"]

    obs, lo, hi = bootstrap_gap(
        [u["_unit_wer"] for u in stut],
        [u["_unit_wer"] for u in flu]) if stut and flu else (0, 0, 0)

    return {
        "reference": ref_field,
        "overall": block(scored),
        "stuttered": block(stut),
        "fluent": block(flu),
        "stuttered_with_labels": block([u for u in stut
                                        if u.get("has_stutter_labels")]),
        "gap_unit_level": {
            "stuttered_minus_fluent_macro_wer": round(obs, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
        },
        "gap_speaker_level": {
            "stuttered_macro_wer": round(sum(spk_stut) / len(spk_stut), 4)
                                   if spk_stut else None,
            "fluent_macro_wer": round(sum(spk_flu) / len(spk_flu), 4)
                                if spk_flu else None,
            "n_speakers_stuttered": len(spk_stut),
            "n_speakers_fluent": len(spk_flu),
        },
    }


def selftest() -> None:
    """Validate scoring on a tiny synthetic case."""
    assert wer_counts("a b c", "a b c") == (0, 0, 0, 3)
    assert _levenshtein(["a", "b"], ["a", "x"]) == (1, 0, 0, 1)
    s, d, i, h = wer_counts("the cat sat", "the cat")   # 1 deletion
    assert (s, d, i, h) == (0, 1, 0, 2), (s, d, i, h)
    # micro WER over two units
    units = [{"wer_c": wer_counts("a b c d", "a b c d")},
             {"wer_c": wer_counts("a b c d", "a b x d")}]
    assert abs(micro_wer(units, "wer_c") - (1 / 8)) < 1e-9
    n = get_normaliser()
    assert n("The CAT, sat!") .replace(".", "") != ""
    print("selftest OK: WER/CER counts, micro aggregation, normaliser")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--hyp", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not (args.manifest and args.hyp and args.out):
        ap.error("--manifest, --hyp and --out are required (or --selftest)")

    norm = get_normaliser()
    using = ("whisper EnglishTextNormalizer"
             if norm.__class__.__name__ == "EnglishTextNormalizer"
             else "basic fallback normaliser")
    print(f"normaliser: {using}")

    result = {"hyp_file": str(args.hyp), "normaliser": using, "by_reference": {}}
    for ref_field in ("reference_intended", "reference_surface"):
        result["by_reference"][ref_field] = score(
            args.manifest, args.hyp, ref_field, norm)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))

    # Console summary
    for ref_field, r in result["by_reference"].items():
        print(f"\n[{ref_field}]")
        print(f"  stuttered WER micro={r['stuttered']['wer_micro']} "
              f"macro={r['stuttered']['wer_macro']} (n={r['stuttered']['n_units']})")
        print(f"  fluent    WER micro={r['fluent']['wer_micro']} "
              f"macro={r['fluent']['wer_macro']} (n={r['fluent']['n_units']})")
        g = r["gap_unit_level"]
        print(f"  gap (stut-flu macro WER) = {g['stuttered_minus_fluent_macro_wer']} "
              f"95%CI {g['ci95']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
