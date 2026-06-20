#!/usr/bin/env python3
"""Stutter-type co-dependency analysis: P(ASR error | stutter type).

This is the H2 engine. For every reference word that carries a gold stutter-type
label (from ``h2_word_events.jsonl``), we determine what the ASR did to it —
Correct / Substituted / Deleted — by aligning the model's session-level
hypothesis to the per-word reference, then cross-tabulate. The result is a
contingency table P(op | stutter_type) per model, with a chi-square test of
independence and a normalised mutual information.

The gold per-word labels live on the *curated* SLASS sessions, which are NOT in
``benchmark_v1`` (different naming). So this needs ASR hypotheses on the H2
sessions specifically. Two modes:

    # 1) build a session-level manifest for run_asr from the word events
    python -m src.evaluation.codependency build-manifest \\
        --h2 /media/liamb/FATSPEECH/manifests/h2_word_events.jsonl \\
        --audio-root-map "/Volumes/ritd-ag-project-rd02dw-lbarr63=/media/liamb/FATSPEECH" \\
        --out /media/liamb/FATSPEECH/manifests/h2_sessions.jsonl
    #    ...then run the models over it (run_asr / run_benchmark) ...

    # 2) score the co-dependency from the hypotheses
    python -m src.evaluation.codependency score \\
        --h2 .../h2_word_events.jsonl --hyp .../hyp/whisper-large-v3.jsonl \\
        --out .../codep/whisper-large-v3.json

    python -m src.evaluation.codependency --selftest

Normalisation note: reference words are normalised token-by-token (so the
word↔stutter-type alignment is preserved); tokens that normalise to empty are
dropped with their label. The hypothesis is normalised as a whole then split.
This keeps the label mapping exact at a small cost in number/contraction
handling — documented as a limitation.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from src.evaluation.score import get_normaliser, align_ref_ops

OPS = ("C", "S", "D")
# Canonical stutter types we report (others fall through as-is).
TYPE_ORDER = ("Fluent", "Block", "Prolongation", "PWR", "WWR", "Combined")


def load_sessions(h2_path: Path) -> dict[str, list[dict]]:
    """Group word events by session, ordered by word_index."""
    sess: dict[str, list[dict]] = defaultdict(list)
    for line in h2_path.open(encoding="utf-8"):
        e = json.loads(line)
        sess[e["session"]].append(e)
    for s in sess.values():
        s.sort(key=lambda e: e.get("word_index", 0))
    return sess


def build_manifest(h2_path: Path, out: Path, root_map: str = "") -> None:
    old = new = None
    if root_map:
        old, new = root_map.split("=", 1)
    sess = load_sessions(h2_path)
    n_exist = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for name, events in sorted(sess.items()):
            words = [e["word"] for e in events if e.get("word")]
            audio = events[0]["audio_path"]
            if old and audio.startswith(old):
                audio = new + audio[len(old):]
            dur = max((e.get("end_s") or 0.0) for e in events)
            if Path(audio).exists():
                n_exist += 1
            fh.write(json.dumps({
                "unit_id": name,
                "dataset": "slass_h2",
                "condition": "stuttered",
                "audio_path": audio,
                "speaker_id": name.split("_")[0],
                "split": "test",
                "reference_intended": " ".join(words),
                "reference_surface": " ".join(words),
                "n_ref_words": len(words),
                "duration_s": round(dur, 3),
                "needs_chunking": bool(dur > 30.0),
                "has_stutter_labels": True,
                "stutter_type_counts": dict(Counter(
                    e.get("stutter_type", "") for e in events)),
            }) + "\n")
    print(f"wrote {len(sess)} session units -> {out} "
          f"({n_exist} with audio present)")


def _mutual_information(table: dict[str, Counter]) -> float:
    """Normalised mutual information between stutter type and op (0..1)."""
    total = sum(sum(c.values()) for c in table.values())
    if not total:
        return 0.0
    p_t = {t: sum(c.values()) / total for t, c in table.items()}
    p_o = {o: sum(table[t][o] for t in table) / total for o in OPS}
    mi = 0.0
    for t, c in table.items():
        for o in OPS:
            p_to = c[o] / total
            if p_to > 0 and p_t[t] > 0 and p_o[o] > 0:
                mi += p_to * math.log(p_to / (p_t[t] * p_o[o]))
    # normalise by min entropy
    h_t = -sum(p * math.log(p) for p in p_t.values() if p > 0)
    h_o = -sum(p * math.log(p) for p in p_o.values() if p > 0)
    denom = min(h_t, h_o)
    return mi / denom if denom > 0 else 0.0


def _chi_square(table: dict[str, Counter]) -> dict:
    types = [t for t in table if sum(table[t].values()) > 0]
    mat = [[table[t][o] for o in OPS] for t in types]
    try:
        from scipy.stats import chi2_contingency
        chi2, p, dof, _ = chi2_contingency(mat)
        return {"chi2": round(float(chi2), 3), "p_value": float(p), "dof": int(dof)}
    except Exception as e:                       # scipy missing or singular table
        return {"chi2": None, "p_value": None, "note": str(e)[:80]}


def score_codependency(h2_path: Path, hyp_path: Path, normaliser) -> dict:
    sess = load_sessions(h2_path)
    hyps = {}
    for line in hyp_path.open(encoding="utf-8"):
        h = json.loads(line)
        hyps[h["unit_id"]] = h.get("hypothesis", "") or ""

    table: dict[str, Counter] = defaultdict(Counter)
    n_sess = n_words = total_ins = 0
    for name, events in sess.items():
        if name not in hyps:
            continue
        ref_words, ref_types = [], []
        for e in events:
            w = normaliser(e.get("word", "") or "").strip()
            if w:                                # keep word↔type alignment
                ref_words.append(w)
                ref_types.append(e.get("stutter_type", "") or "Unknown")
        if not ref_words:
            continue
        hyp_tokens = normaliser(hyps[name]).split()
        ops, ins = align_ref_ops(ref_words, hyp_tokens)
        for op, t in zip(ops, ref_types):
            table[t][op] += 1
        n_sess += 1
        n_words += len(ref_words)
        total_ins += ins

    # Per-type rates: error = (S+D)/(C+S+D), and P(op|type)
    by_type = {}
    for t in sorted(table, key=lambda x: (TYPE_ORDER.index(x)
                                          if x in TYPE_ORDER else 99, x)):
        c = table[t]
        n = sum(c.values())
        by_type[t] = {
            "n": n,
            "error_rate": round((c["S"] + c["D"]) / n, 4) if n else None,
            "P_correct": round(c["C"] / n, 4) if n else None,
            "P_sub": round(c["S"] / n, 4) if n else None,
            "P_del": round(c["D"] / n, 4) if n else None,
            "counts": {o: c[o] for o in OPS},
        }
    return {
        "hyp_file": str(hyp_path),
        "n_sessions_scored": n_sess,
        "n_labelled_ref_words": n_words,
        "total_insertions": total_ins,
        "by_stutter_type": by_type,
        "independence_test": _chi_square(table),
        "normalised_mutual_information": round(_mutual_information(table), 4),
    }


def selftest() -> None:
    norm = get_normaliser()
    # synthetic: session 's1', words the/boy/boy/ran, the two 'boy' are PWR.
    # hyp drops one 'boy' -> for PWR: one Correct, one Deleted (error_rate 0.5).
    import tempfile
    d = Path(tempfile.mkdtemp())
    h2 = d / "h2.jsonl"
    with h2.open("w") as f:
        for i, (w, t) in enumerate([("the", "Fluent"), ("boy", "PWR"),
                                    ("boy", "PWR"), ("ran", "Fluent")]):
            f.write(json.dumps({"session": "s1", "word_index": i, "word": w,
                                "stutter_type": t, "audio_path": "/x.wav",
                                "end_s": 1.0}) + "\n")
    hyp = d / "hyp.jsonl"
    hyp.write_text(json.dumps({"unit_id": "s1", "hypothesis": "the boy ran"}) + "\n")
    r = score_codependency(h2, hyp, norm)
    pwr = r["by_stutter_type"]["PWR"]
    assert pwr["counts"] == {"C": 1, "S": 0, "D": 1}, pwr
    assert pwr["error_rate"] == 0.5, pwr
    assert r["by_stutter_type"]["Fluent"]["counts"] == {"C": 2, "S": 0, "D": 0}
    print("selftest OK: co-dependency alignment + contingency")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?", choices=["build-manifest", "score"])
    ap.add_argument("--h2", type=Path)
    ap.add_argument("--hyp", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--audio-root-map", default="")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if args.mode == "build-manifest":
        if not (args.h2 and args.out):
            ap.error("build-manifest needs --h2 and --out")
        build_manifest(args.h2, args.out, args.audio_root_map)
        return
    if args.mode == "score":
        if not (args.h2 and args.hyp and args.out):
            ap.error("score needs --h2, --hyp and --out")
        r = score_codependency(args.h2, args.hyp, get_normaliser())
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(r, indent=2))
        print(f"sessions={r['n_sessions_scored']} "
              f"labelled_words={r['n_labelled_ref_words']} "
              f"NMI={r['normalised_mutual_information']} "
              f"chi2={r['independence_test'].get('chi2')}")
        print("error_rate by stutter type:")
        for t, v in r["by_stutter_type"].items():
            print(f"  {t:14s} n={v['n']:5d}  err={v['error_rate']}  "
                  f"P(C/S/D)={v['P_correct']}/{v['P_sub']}/{v['P_del']}")
        print(f"wrote {args.out}")
        return
    ap.error("specify a mode (build-manifest | score) or --selftest")


if __name__ == "__main__":
    main()
