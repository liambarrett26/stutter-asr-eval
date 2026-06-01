#!/usr/bin/env python3
"""Compute inter-rater agreement on the UNWR children's nonword reading data.

Reads /Volumes/FATSPEECH/unwr_reliability/processed/items.csv (produced by
process_unwr_reliability.py) and emits two outputs:

    agreement_pairwise.csv      One row per (school, child, item, transcriber_a,
                                transcriber_b) with phoneme edit distance,
                                exact-match flag, length deltas.

    agreement_summary.csv       One row per transcriber pair with mean PER
                                (phoneme error rate), exact-match rate,
                                accuracy-against-target, n items compared.

PER is computed as Levenshtein edit distance between phoneme sequences
divided by reference length, where the reference is the other transcriber's
response (for pairwise) or the published target (for accuracy).

Standard library only.

Usage:
    python src/data/unwr_reliability_agreement.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


ITEMS_CSV = Path("/Volumes/FATSPEECH/unwr_reliability/processed/items.csv")
OUT_DIR = ITEMS_CSV.parent

# Transcribers we treat as principal raters (retranscriptions + originals).
# 'Roaa_empty' and 'empty_placeholder' contain placeholder TextGrids with
# no transcribed content; excluded.
RATER_KEEP = {"Clarissa", "Kaho", "Roaa_28Nov",
              "Clarissa_original", "Kaho_original"}


def tokens(s: str) -> list[str]:
    """Split a phoneme sequence into tokens, ignoring empty strings."""
    return [t for t in s.replace(",", " ").split() if t]


def edit_distance(a: list[str], b: list[str]) -> int:
    """Token-level Levenshtein distance."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ai in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, bj in enumerate(b, 1):
            cost = 0 if ai == bj else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def per(ref: list[str], hyp: list[str]) -> float:
    """Phoneme Error Rate = edit_distance / len(ref). 0 if ref empty."""
    return (edit_distance(ref, hyp) / len(ref)) if ref else 0.0


def main() -> None:
    rows = list(csv.DictReader(ITEMS_CSV.open(encoding="utf-8")))
    print(f"Loaded {len(rows):,} item rows")

    # Group by (school, child_name, syllable_length, item_kind, item_idx).
    keyed: dict[tuple, dict[str, dict]] = defaultdict(dict)
    targets: dict[tuple, str] = {}
    for r in rows:
        if r["transcriber"] not in RATER_KEEP:
            continue
        if not r["response"]:
            continue
        key = (r["school"], r["child_name"],
               r["syllable_length"], r["item_kind"], r["item_idx"])
        # Multiple files for the same (transcriber, item) can occur if a
        # rater retranscribed an item more than once; keep the first.
        keyed[key].setdefault(r["transcriber"], r)
        if r["target"] and key not in targets:
            targets[key] = r["target"]

    print(f"Items with at least one principal rater: {len(keyed):,}")
    print(f"Items with a published target available: {len(targets):,}")

    # Pairwise comparison
    pair_rows = []
    pair_summary: dict[tuple, list] = defaultdict(list)
    rater_accuracy: dict[str, list] = defaultdict(list)

    raters = sorted(RATER_KEEP)
    for key, by_rater in keyed.items():
        present = sorted(by_rater.keys())
        target_tokens = tokens(targets.get(key, ""))
        for ra in present:
            if target_tokens:
                acc_per = per(target_tokens,
                              tokens(by_rater[ra]["response"]))
                rater_accuracy[ra].append(acc_per)
        for i, ra in enumerate(present):
            for rb in present[i + 1:]:
                ta = tokens(by_rater[ra]["response"])
                tb = tokens(by_rater[rb]["response"])
                ed = edit_distance(ta, tb)
                pair_per = (ed / len(ta)) if ta else 0.0
                exact = (ta == tb)
                school, child, syl, kind, idx = key
                pair_rows.append({
                    "school": school,
                    "child_name": child,
                    "syllable_length": syl,
                    "item_kind": kind,
                    "item_idx": idx,
                    "transcriber_a": ra,
                    "transcriber_b": rb,
                    "len_a": len(ta),
                    "len_b": len(tb),
                    "edit_distance": ed,
                    "per": f"{pair_per:.3f}",
                    "exact_match": "yes" if exact else "no",
                })
                pair_summary[(ra, rb)].append((pair_per, exact))

    # Write pairwise CSV
    cols = ["school", "child_name", "syllable_length", "item_kind",
            "item_idx", "transcriber_a", "transcriber_b",
            "len_a", "len_b", "edit_distance", "per", "exact_match"]
    out_pw = OUT_DIR / "agreement_pairwise.csv"
    with out_pw.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in pair_rows:
            w.writerow(r)
    print(f"Wrote {len(pair_rows):,} pairwise rows -> {out_pw}")

    # Per-pair and per-rater summaries
    out_sum = OUT_DIR / "agreement_summary.csv"
    with out_sum.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pair_or_rater", "kind",
                    "n", "mean_per", "exact_match_rate"])
        # Pair-level
        for (ra, rb), vals in sorted(pair_summary.items()):
            n = len(vals)
            mean_per = sum(p for p, _ in vals) / n if n else 0
            ex = sum(1 for _, e in vals if e) / n if n else 0
            w.writerow([f"{ra} vs {rb}", "pair", n,
                        f"{mean_per:.3f}", f"{ex:.3f}"])
        # Per-rater accuracy against published target
        for ra, vals in sorted(rater_accuracy.items()):
            n = len(vals)
            mean_per = sum(vals) / n if n else 0
            w.writerow([ra, "accuracy_vs_target",
                        n, f"{mean_per:.3f}", ""])

    # Console summary
    print(f"\nPairwise comparison summary (target-vs-transcriber PER):")
    for ra in sorted(rater_accuracy):
        vals = rater_accuracy[ra]
        if not vals:
            continue
        print(f"  {ra:25s} n={len(vals):>4}  "
              f"mean PER vs target = {sum(vals)/len(vals):.3f}")
    print(f"\nInter-rater pairwise PER:")
    for (ra, rb), vals in sorted(pair_summary.items()):
        n = len(vals)
        mean_per = sum(p for p, _ in vals) / n
        ex = sum(1 for _, e in vals if e) / n
        print(f"  {ra:20s} vs {rb:20s}  n={n:>4}  "
              f"PER={mean_per:.3f}  exact={ex:.3f}")


if __name__ == "__main__":
    main()
