#!/usr/bin/env python3
"""
Exploratory Data Analysis of stuttered speech datasets.

Generates individual figures for:
    1. Dataset summary table
    2. Stutter type distributions (Jason ordered + standardised post-fix)
    3. Word duration KDEs (Jason matrices, with WWR; standardised event spans)
    4. Cross-dataset duration comparison
    5. Stuttering rate by syllable count (overall + by stutter type)
    6. Content vs function word analysis (overall + by age band)
    7. Most frequently stuttered words (overall + by stutter type)
    8. Directed co-occurrence heatmap (ordered pairs)
    9. Stutter-type to stutter-type transition matrix
   10. Self-transition rates
   11. Run-length distribution of consecutive disfluent words

Usage:
    python results/eda/generate_eda.py
"""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# ── Config ───────────────────────────────────────────────────────────────────

OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLOURS = {
    "Fluent": "#1f77b4",
    "Prolongation": "#ff7f0e",
    "PWR": "#2ca02c",
    "WWR": "#d62728",
    "Block": "#9467bd",
    "Combined": "#8c564b",
    "Other": "#7f7f7f",
}

TYPE_ORDER = ("Fluent", "Prolongation", "PWR", "WWR", "Block", "Combined")
PURE_STUTTER_TYPES = ("Prolongation", "Block", "PWR", "WWR")

plt.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


# ── Helpers ──────────────────────────────────────────────────────────────────

AGE_RE = re.compile(r"_(\d{1,2})y(\d{1,2})m", re.I)


def age_months_from_filename(filename: str) -> int | None:
    """Return age in months parsed from a SLASS-style filename, or None."""
    m = AGE_RE.search(filename)
    if not m:
        return None
    return int(m.group(1)) * 12 + int(m.group(2))


def age_band(months: int | None) -> str:
    """Bucket an age-in-months into a band label."""
    if months is None:
        return "unknown"
    years = months / 12
    if years < 8:
        return "<8y"
    if years < 10:
        return "8–10y"
    if years < 12:
        return "10–12y"
    if years < 14:
        return "12–14y"
    if years < 18:
        return "14–18y"
    return "18y+"


AGE_BAND_ORDER = ("<8y", "8–10y", "10–12y", "12–14y", "14–18y", "18y+", "unknown")


def simplify_jason(st: str) -> str:
    """Collapse the Jason matrices' Stutter_Type field to a single category."""
    st = st.strip()
    if st == "Fluent" or not st:
        return "Fluent"
    if st == "Unknown":
        return "Unknown"
    pure = {"Block", "Prolongation", "PWR", "WWR"}
    for key in pure:
        others = pure - {key}
        if key in st and not any(o in st for o in others):
            return key
    return "Combined"


def first_type(st: str) -> str | None:
    """Return the first stutter type listed in an ordered Stutter_Type string."""
    st = st.strip()
    if not st or st in ("Fluent", "Unknown"):
        return None
    head = st.split("+")[0].strip()
    if head in PURE_STUTTER_TYPES:
        return head
    return None


# ── Data loading ─────────────────────────────────────────────────────────────

def load_jason_matrices():
    """Load Jason usage matrices with WWR start|end timestamps handled.

    Each row gets:
        word, fluency, stutter_type (raw, ordered),
        stutter_type_simple (collapsed Fluent/Prol/Block/PWR/WWR/Combined),
        first_type (e.g. 'Prolongation' for 'Prolongation + PWR'),
        syllable, word_type, ts_start, ts_end, duration,
        file, age_months, age_band

    For WWR rows the Timestamp field is 'start | end' and duration is
    end-start directly. For other rows the Timestamp is a single onset
    and duration is the gap to the next row's onset (capped at 30s).
    """
    matrix_dir = Path("/Volumes/SPEECH/from_jason/Speech data Jason/output")
    all_words = []

    for f in sorted(matrix_dir.glob("*_Usage_Matrix.csv")):
        # Strip the '_Usage_Matrix' suffix so the age regex finds 'XyXm'
        stem = f.stem.replace("_Usage_Matrix", "")
        age_m = age_months_from_filename(stem)
        rows = []
        with open(f, encoding="latin-1") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                ts_field = (row.get("Timestamp", "") or "").strip()
                ts_start = ts_end = None
                if "|" in ts_field:
                    parts = [p.strip() for p in ts_field.split("|")]
                    try:
                        ts_start = float(parts[0])
                        ts_end = float(parts[1])
                    except (ValueError, IndexError):
                        pass
                elif ts_field:
                    try:
                        ts_start = float(ts_field)
                    except ValueError:
                        pass
                rows.append({
                    "file": stem,
                    "age_months": age_m,
                    "age_band": age_band(age_m),
                    "word": row.get("Word", ""),
                    "fluency": row.get("Fluency", ""),
                    "stutter_type": (row.get("Stutter_Type", "") or "").strip(),
                    "syllable": int(row.get("Syllable", "1") or "1"),
                    "word_type": row.get("Word_Type", ""),
                    "ts_start": ts_start,
                    "ts_end": ts_end,
                    "duration": None,
                })

        # Duration:
        #   - if both ts_start and ts_end are set, use end - start (WWR case)
        #   - otherwise use inter-onset interval to next row's ts_start
        for i, r in enumerate(rows):
            if r["ts_start"] is not None and r["ts_end"] is not None:
                d = r["ts_end"] - r["ts_start"]
                if 0 < d < 30:
                    r["duration"] = d
            elif (r["ts_start"] is not None
                  and i + 1 < len(rows)
                  and rows[i + 1]["ts_start"] is not None):
                d = rows[i + 1]["ts_start"] - r["ts_start"]
                if 0 < d < 30:
                    r["duration"] = d

        for r in rows:
            r["stutter_type_simple"] = simplify_jason(r["stutter_type"])
            r["first_type"] = first_type(r["stutter_type"])
        all_words.extend(rows)

    return all_words


def load_standardised(corpus_dir: Path):
    """Load standardised CSV files into a list of dicts."""
    records = []
    if not corpus_dir.exists():
        return records
    for f in corpus_dir.rglob("*.csv"):
        if f.name.startswith("."):
            continue
        text = f.read_bytes().replace(b"\x00", b"").decode("latin-1")
        for row in csv.DictReader(text.splitlines()):
            dur = None
            if row.get("start_s") and row.get("end_s"):
                try:
                    dur = float(row["end_s"]) - float(row["start_s"])
                except ValueError:
                    pass
            records.append({**row, "duration": dur})
    return records


def count_slass_full_archive_speakers() -> tuple[int, int, int]:
    """Return (n_speakers, n_files_with_id, n_files_total) for the
    SLASS full extracted archive."""
    inv = Path("/Volumes/FATSPEECH/slass/full_archive/inventory.csv")
    if not inv.exists():
        return (0, 0, 0)
    ids = set()
    named = 0
    total = 0
    with inv.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            total += 1
            stem = row["filename"].rsplit(".", 1)[0]
            m = re.match(r"^[CcFfMm_]*(\d{3,4})", stem)
            if m:
                ids.add(m.group(1).zfill(4))
                named += 1
    return (len(ids), named, total)


UNWR_GROUP_COLOURS = {
    "Control": "#1f77b4",
    "Attention": "#ff7f0e",
    "Stutter": "#d62728",
    "Attention+Stutter": "#9467bd",
}


def load_unwr_speakers() -> list[dict]:
    """Load UNWR per-speaker info merging inventory + speakers."""
    inv_p = Path("/Volumes/FATSPEECH/unwr/processed/inventory.csv")
    spk_p = Path("/Volumes/FATSPEECH/unwr/processed/speakers.csv")
    if not inv_p.exists() or not spk_p.exists():
        return []
    spk = {r["pid"]: r for r in csv.DictReader(spk_p.open(encoding="utf-8"))
           if r.get("pid")}
    rows = []
    for r in csv.DictReader(inv_p.open(encoding="utf-8")):
        sp = spk.get(r["pid"], {})
        dur = 0.0
        for col in ("q1_duration_s", "q2_duration_s"):
            try:
                dur += float(r.get(col, "") or 0)
            except ValueError:
                pass
        try:
            ssi_pct = float(sp.get("ssi_pct", "") or "")
        except ValueError:
            ssi_pct = None
        try:
            age = int(sp.get("age", "") or "")
        except ValueError:
            age = None
        rows.append({
            "pid": r["pid"],
            "group": r["group"],
            "total_audio_s": dur,
            "ssi_pct": ssi_pct,
            "age": age,
            "gender": sp.get("gender", ""),
        })
    return rows


# ── KDE plot ─────────────────────────────────────────────────────────────────

def plot_kde(durations_by_type, title, filename, x_max=2.0,
             type_order=TYPE_ORDER):
    fig, ax = plt.subplots(figsize=(10, 6))
    for stype in type_order:
        durs = durations_by_type.get(stype, [])
        durs = [d for d in durs if 0.01 < d < x_max * 1.5]
        if len(durs) < 5:
            continue
        try:
            kde = gaussian_kde(durs, bw_method=0.3)
            x = np.linspace(0, x_max, 500)
            y = kde(x)
            colour = COLOURS.get(stype, "#999999")
            ax.plot(x, y, label=f"{stype} (n={len(durs)})",
                    color=colour, linewidth=2)
            ax.fill_between(x, y, alpha=0.15, color=colour)
        except Exception:
            pass
    ax.set_xlabel("Duration (s)")
    ax.set_ylabel("Probability density")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.set_xlim(0, x_max)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    jason = load_jason_matrices()
    print(f"  Jason matrices: {len(jason):,} rows")
    jason_with_dur = [w for w in jason if w["duration"] is not None]
    print(f"    with duration: {len(jason_with_dur):,}")

    uclass = load_standardised(Path("/Volumes/FATSPEECH/standardised/uclass"))
    print(f"  UCLASS: {len(uclass):,} records")
    fb = load_standardised(Path("/Volumes/FATSPEECH/standardised/fluencybank"))
    print(f"  FluencyBank: {len(fb):,} records")
    slass_std = load_standardised(Path("/Volumes/FATSPEECH/standardised/slass"))
    print(f"  SLASS standardised: {len(slass_std):,} records")

    n_spk, n_named, n_total = count_slass_full_archive_speakers()
    print(f"  SLASS full archive: ≥{n_spk} speakers "
          f"({n_named}/{n_total} files with parseable id)")

    unwr = load_unwr_speakers()
    unwr_hours = sum(s["total_audio_s"] for s in unwr) / 3600
    print(f"  UNWR (adult cohort): {len(unwr)} speakers, "
          f"{unwr_hours:.2f} hours")

    # ── KDE: SLASS Jason (now including WWR) ──
    print("\nKDE plots...")
    jason_dur = defaultdict(list)
    for w in jason_with_dur:
        s = w["stutter_type_simple"]
        if s != "Unknown":
            jason_dur[s].append(w["duration"])

    plot_kde(jason_dur,
             "Word duration by disfluency type — SLASS (Jason matrices)",
             "kde_duration_slass_jason.png", x_max=2.5)

    # Print duration stats for results.md
    print("\nJason duration stats (after WWR fix):")
    for stype in TYPE_ORDER:
        ds = jason_dur.get(stype, [])
        if not ds:
            continue
        ds_sorted = sorted(ds)
        median = ds_sorted[len(ds_sorted) // 2]
        mean = sum(ds_sorted) / len(ds_sorted)
        print(f"  {stype:14s} n={len(ds):>4}  "
              f"mean={mean:.2f}s  median={median:.2f}s")

    # ── KDE: SLASS standardised (event spans) ──
    slass_std_dur = defaultdict(list)
    for r in slass_std:
        if r.get("duration") and 0.01 < r["duration"] < 5:
            st = (r.get("stutter_type") or "fluent").lower()
            # Map standardised labels to display labels
            display = {
                "fluent": "Fluent",
                "block": "Block",
                "prolongation": "Prolongation",
                "pwr": "PWR",
                "wwr": "WWR",
            }.get(st, "Combined" if "+" in st else None)
            if display:
                slass_std_dur[display].append(r["duration"])
    plot_kde(slass_std_dur,
             "Annotation span duration by type — SLASS standardised",
             "kde_duration_slass_standardised.png", x_max=3.0)

    # ── KDE: UCLASS, FluencyBank, cross-dataset ──
    uclass_durs = {"UCLASS (all words)":
                   [r["duration"] for r in uclass
                    if r.get("duration") and 0.01 < r["duration"] < 5]}
    plot_kde(uclass_durs,
             "Word duration distribution — UCLASS",
             "kde_duration_uclass.png", x_max=2.0,
             type_order=("UCLASS (all words)",))

    fb_durs = {"FluencyBank (utterances)":
               [r["duration"] for r in fb
                if r.get("duration") and 0.1 < r["duration"] < 60]}
    plot_kde(fb_durs,
             "Utterance duration distribution — FluencyBank",
             "kde_duration_fluencybank.png", x_max=30.0,
             type_order=("FluencyBank (utterances)",))

    fig, ax = plt.subplots(figsize=(10, 6))
    cross = {
        "SLASS fluent":
            [w["duration"] for w in jason_with_dur
             if w["stutter_type_simple"] == "Fluent"
             and 0.01 < w["duration"] < 3],
        "SLASS stuttered":
            [w["duration"] for w in jason_with_dur
             if w["stutter_type_simple"] not in ("Fluent", "Unknown")
             and 0.01 < w["duration"] < 3],
        "UCLASS (all)":
            [r["duration"] for r in uclass
             if r.get("duration") and 0.01 < r["duration"] < 3],
    }
    cross_colours = {"SLASS fluent": "#1f77b4",
                     "SLASS stuttered": "#d62728",
                     "UCLASS (all)": "#2ca02c"}
    for name, durs in cross.items():
        if len(durs) < 5:
            continue
        kde = gaussian_kde(durs, bw_method=0.3)
        x = np.linspace(0, 2.0, 500)
        ax.plot(x, kde(x), label=f"{name} (n={len(durs)})",
                color=cross_colours[name], linewidth=2)
        ax.fill_between(x, kde(x), alpha=0.1, color=cross_colours[name])
    ax.set_xlabel("Duration (s)")
    ax.set_ylabel("Probability density")
    ax.set_title("Word duration distribution across datasets")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 2.0)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "kde_duration_cross_dataset.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved kde_duration_cross_dataset.png")

    # ── Type distribution: Jason ──
    print("\nType distributions...")
    jason_types = Counter(w["stutter_type_simple"]
                          for w in jason
                          if w["fluency"] == "Stuttered")
    labels = ["Prolongation", "Block", "PWR", "WWR", "Combined"]
    values = [jason_types.get(l, 0) for l in labels]
    total = sum(values)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values,
                  color=[COLOURS[l] for l in labels], edgecolor="white")
    ax.set_ylabel("Count")
    ax.set_title(f"Disfluency type distribution — "
                 f"SLASS Jason matrices (n={total} stuttered words)")
    for bar, val in zip(bars, values):
        pct = val / total * 100 if total else 0
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.01,
                f"{val}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "type_distribution_slass_jason.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved type_distribution_slass_jason.png")

    # ── Type distribution: Standardised (post-fix) ──
    std_types = Counter()
    for r in slass_std:
        st = (r.get("stutter_type") or "fluent").lower().strip()
        if not st or st == "fluent":
            continue
        if st == "unknown":
            continue
        if "+" in st:
            std_types["Combined"] += 1
        elif st in ("block",):
            std_types["Block"] += 1
        elif st == "prolongation":
            std_types["Prolongation"] += 1
        elif st == "pwr":
            std_types["PWR"] += 1
        elif st == "wwr":
            std_types["WWR"] += 1
        else:
            std_types["Other"] += 1

    labels = ["Prolongation", "Block", "PWR", "WWR", "Combined", "Other"]
    values = [std_types.get(l, 0) for l in labels]
    total = sum(values)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values,
                  color=[COLOURS.get(l, "#999") for l in labels],
                  edgecolor="white")
    ax.set_ylabel("Count")
    ax.set_title(f"Disfluency type distribution — "
                 f"SLASS standardised post-fix (n={total} disfluent events)")
    for bar, val in zip(bars, values):
        pct = val / total * 100 if total else 0
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.02,
                f"{val}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "type_distribution_slass_standardised.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved type_distribution_slass_standardised.png")

    # ── Syllable analysis (overall) ──
    print("\nSyllable analyses...")
    fig, ax = plt.subplots(figsize=(7, 5))
    for s in range(1, 6):
        tot = sum(1 for w in jason if w["syllable"] == s)
        stutt = sum(1 for w in jason
                    if w["syllable"] == s and w["fluency"] == "Stuttered")
        if tot > 10:
            rate = stutt / tot * 100
            ax.bar(s, rate, color="#1f77b4", edgecolor="white")
            ax.text(s, rate + 0.3, f"{rate:.1f}%\n(n={tot})",
                    ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Syllable count")
    ax.set_ylabel("Stuttering rate (%)")
    ax.set_title("Stuttering rate by syllable count — SLASS Jason matrices")
    ax.set_xticks(range(1, 5))
    plt.tight_layout()
    plt.savefig(OUT_DIR / "syllable_rate_slass.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved syllable_rate_slass.png")

    # ── Syllable × stutter type (NEW) ──
    types_for_bar = ["Prolongation", "Block", "PWR", "WWR", "Combined"]
    syl_range = range(1, 5)
    matrix = np.zeros((len(types_for_bar), len(syl_range)))
    totals = []
    for j, s in enumerate(syl_range):
        tot = sum(1 for w in jason if w["syllable"] == s)
        totals.append(tot)
        for i, t in enumerate(types_for_bar):
            n = sum(1 for w in jason
                    if w["syllable"] == s
                    and w["stutter_type_simple"] == t)
            matrix[i, j] = (n / tot * 100) if tot else 0
    fig, ax = plt.subplots(figsize=(9, 5.5))
    width = 0.15
    xs = np.arange(len(syl_range))
    for i, t in enumerate(types_for_bar):
        ax.bar(xs + i * width - 2 * width, matrix[i],
               width, label=t,
               color=COLOURS.get(t, "#999"), edgecolor="white")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{s}\n(n={totals[j]})"
                        for j, s in enumerate(syl_range)])
    ax.set_xlabel("Syllable count")
    ax.set_ylabel("Rate within syllable-count bin (%)")
    ax.set_title("Stutter-type rate by syllable count — "
                 "SLASS Jason matrices")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "syllable_rate_by_type_slass.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved syllable_rate_by_type_slass.png")

    # ── Content vs function word analysis (overall) ──
    print("\nWord-type analyses...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, wt in enumerate(["Content", "Function"]):
        tot = sum(1 for w in jason if w["word_type"] == wt)
        stutt = sum(1 for w in jason
                    if w["word_type"] == wt and w["fluency"] == "Stuttered")
        rate = stutt / tot * 100 if tot else 0
        axes[0].bar(wt, rate, color=["#2ca02c", "#ff7f0e"][i],
                    edgecolor="white")
        axes[0].text(i, rate + 0.3, f"{rate:.1f}%\n(n={tot})",
                     ha="center", fontsize=10)
    axes[0].set_ylabel("Stuttering rate (%)")
    axes[0].set_title("Stuttering rate by word type")

    labels = ["Prolongation", "Block", "PWR", "WWR", "Combined"]
    x = np.arange(len(labels))
    width = 0.35
    for j, wt in enumerate(["Content", "Function"]):
        stutt_words = [w for w in jason
                       if w["word_type"] == wt
                       and w["fluency"] == "Stuttered"]
        tot = len(stutt_words)
        vals = [(sum(1 for w in stutt_words
                     if w["stutter_type_simple"] == l) / tot * 100)
                if tot else 0 for l in labels]
        axes[1].bar(x + j * width - width / 2, vals, width,
                    label=wt,
                    color=["#2ca02c", "#ff7f0e"][j], edgecolor="white")
    axes[1].set_ylabel("Proportion of stuttered events (%)")
    axes[1].set_title("Disfluency type by word type")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "word_type_slass.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved word_type_slass.png")

    # ── Word type × age band (NEW) ──
    fig, ax = plt.subplots(figsize=(10, 6))
    bands_with_data = [b for b in AGE_BAND_ORDER
                       if any(w["age_band"] == b for w in jason)]
    width = 0.35
    xs = np.arange(len(bands_with_data))
    for i, wt in enumerate(["Content", "Function"]):
        rates = []
        ns = []
        for b in bands_with_data:
            band_rows = [w for w in jason if w["age_band"] == b]
            tot = sum(1 for w in band_rows if w["word_type"] == wt)
            stutt = sum(1 for w in band_rows
                        if w["word_type"] == wt
                        and w["fluency"] == "Stuttered")
            rates.append(stutt / tot * 100 if tot else 0)
            ns.append(tot)
        bars = ax.bar(xs + i * width - width / 2, rates, width,
                      label=wt,
                      color=["#2ca02c", "#ff7f0e"][i], edgecolor="white")
        for bar, r, n in zip(bars, rates, ns):
            if n > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        r + 0.3, f"{r:.1f}%\nn={n}",
                        ha="center", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(bands_with_data)
    ax.set_xlabel("Age band (at recording)")
    ax.set_ylabel("Stuttering rate (%)")
    ax.set_title("Stuttering rate by word type across age bands — "
                 "SLASS Jason matrices")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "word_type_by_age_slass.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved word_type_by_age_slass.png")

    # ── Most stuttered words (overall) ──
    print("\nMost-stuttered-words analyses...")
    fig, ax = plt.subplots(figsize=(10, 6))
    stuttered = [w["word"].lower() for w in jason
                 if w["fluency"] == "Stuttered" and w["word"]]
    wc = Counter(stuttered).most_common(20)
    words, counts = zip(*wc)
    y_pos = range(len(words))
    ax.barh(y_pos, counts, color="#1f77b4", edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title(f"Most frequently stuttered words — "
                 f"SLASS Jason matrices (n={len(stuttered)} stuttered tokens)")
    for i, c in enumerate(counts):
        ax.text(c + 1, i, str(c), va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "most_stuttered_words_slass.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved most_stuttered_words_slass.png")

    # ── Most stuttered words × stutter type (NEW) ──
    by_word = defaultdict(Counter)
    word_typeof = {}
    for w in jason:
        if w["fluency"] != "Stuttered" or not w["word"]:
            continue
        key = w["word"].lower()
        by_word[key][w["stutter_type_simple"]] += 1
        word_typeof.setdefault(key, w["word_type"])
    top_words = [k for k, _ in
                 sorted(by_word.items(),
                        key=lambda kv: -sum(kv[1].values()))][:20]
    types_stack = ["Prolongation", "Block", "PWR", "WWR", "Combined"]
    fig, ax = plt.subplots(figsize=(10, 6.5))
    y_pos = np.arange(len(top_words))
    left = np.zeros(len(top_words))
    for t in types_stack:
        vals = np.array([by_word[w].get(t, 0) for w in top_words])
        ax.barh(y_pos, vals, left=left, label=t,
                color=COLOURS[t], edgecolor="white")
        left += vals
    labels_with_wt = [f"{w} [{(word_typeof.get(w) or '?')[:1]}]"
                      for w in top_words]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_with_wt)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title("Most frequently stuttered words by disfluency type — "
                 "SLASS Jason matrices (C = Content, F = Function)")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "most_stuttered_words_by_type_slass.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved most_stuttered_words_by_type_slass.png")

    # ── Directed co-occurrence heatmap (NEW: order-preserving) ──
    print("\nCo-occurrence / transition analyses...")
    pairs = Counter()
    for w in jason:
        st = w["stutter_type"]
        if not st or st in ("Fluent", "Unknown"):
            continue
        parts = [p.strip() for p in st.split("+")]
        parts = [p for p in parts if p in PURE_STUTTER_TYPES]
        for i in range(len(parts) - 1):
            pairs[(parts[i], parts[i + 1])] += 1

    types_4 = list(PURE_STUTTER_TYPES)
    cooc = np.zeros((len(types_4), len(types_4)))
    for (a, b), n in pairs.items():
        cooc[types_4.index(a), types_4.index(b)] = n
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cooc, cmap="Reds", aspect="equal")
    ax.set_xticks(range(len(types_4)))
    ax.set_yticks(range(len(types_4)))
    ax.set_xticklabels(types_4)
    ax.set_yticklabels(types_4)
    ax.set_xlabel("Second component")
    ax.set_ylabel("First component")
    ax.set_title("Within-word ordered co-occurrence — "
                 "SLASS Jason matrices")
    for i in range(len(types_4)):
        for j in range(len(types_4)):
            v = int(cooc[i, j])
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > cooc.max() / 2 else "black",
                    fontsize=11)
    fig.colorbar(im, ax=ax, label="Count")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "cooccurrence_heatmap.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved cooccurrence_heatmap.png")

    # ── Stutter-type transition matrices (NEW) ──
    # Build sequences per file, walking rows in file order.
    by_file: dict[str, list] = defaultdict(list)
    for w in jason:
        by_file[w["file"]].append(w["stutter_type_simple"])

    trans = Counter()
    prev_counts = Counter()
    runs = []
    for file_id, seq in by_file.items():
        run = 0
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            if a in ("Unknown",):
                continue
            trans[(a, b)] += 1
            prev_counts[a] += 1
        for x in seq:
            if x in ("Unknown",):
                continue
            if x == "Fluent":
                if run > 0:
                    runs.append(run)
                    run = 0
            else:
                run += 1
        if run > 0:
            runs.append(run)

    state_order = ["Fluent", "Prolongation", "Block", "PWR",
                   "WWR", "Combined"]
    P = np.zeros((len(state_order), len(state_order)))
    for i, a in enumerate(state_order):
        denom = prev_counts.get(a, 0)
        if denom == 0:
            continue
        for j, b in enumerate(state_order):
            P[i, j] = trans.get((a, b), 0) / denom

    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(P, cmap="Blues", aspect="equal", vmin=0, vmax=1)
    ax.set_xticks(range(len(state_order)))
    ax.set_yticks(range(len(state_order)))
    ax.set_xticklabels(state_order, rotation=30, ha="right")
    ax.set_yticklabels(state_order)
    ax.set_xlabel("Next word's fluency state")
    ax.set_ylabel("Current word's fluency state")
    ax.set_title("Stutter-type to stutter-type transition probability — "
                 "SLASS Jason matrices")
    for i in range(len(state_order)):
        for j in range(len(state_order)):
            v = P[i, j]
            if v > 0.005:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 0.5 else "black",
                        fontsize=9)
    fig.colorbar(im, ax=ax, label="P(next | current)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "transition_probability_heatmap.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved transition_probability_heatmap.png")

    # Self-transitions (diagonal of disfluent states)
    disf_states = ["Prolongation", "Block", "PWR", "WWR", "Combined"]
    self_rates = []
    self_ns = []
    for s in disf_states:
        n = prev_counts.get(s, 0)
        self_rates.append((trans.get((s, s), 0) / n * 100) if n else 0)
        self_ns.append(n)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(disf_states, self_rates,
                  color=[COLOURS[s] for s in disf_states],
                  edgecolor="white")
    for bar, r, n in zip(bars, self_rates, self_ns):
        ax.text(bar.get_x() + bar.get_width() / 2,
                r + 0.3, f"{r:.1f}%\nn={n}",
                ha="center", fontsize=9)
    ax.set_ylabel("P(same type on next word) (%)")
    ax.set_title("Disfluency persistence: self-transition rates — "
                 "SLASS Jason matrices")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "self_transition_rates.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved self_transition_rates.png")

    # Stacked bar: P(next = fluent | previous type)
    fig, ax = plt.subplots(figsize=(9, 5))
    prev_states = state_order
    p_fluent = []
    p_disfluent = []
    for s in prev_states:
        n = prev_counts.get(s, 0)
        if n == 0:
            p_fluent.append(0)
            p_disfluent.append(0)
            continue
        pf = trans.get((s, "Fluent"), 0) / n * 100
        p_fluent.append(pf)
        p_disfluent.append(100 - pf)
    xs = np.arange(len(prev_states))
    ax.bar(xs, p_fluent, label="Next = Fluent", color="#1f77b4")
    ax.bar(xs, p_disfluent, bottom=p_fluent,
           label="Next = Disfluent", color="#d62728")
    ax.set_xticks(xs)
    ax.set_xticklabels(prev_states, rotation=30, ha="right")
    ax.set_ylabel("Probability (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Fluency of next word, by current fluency state — "
                 "SLASS Jason matrices")
    ax.legend(loc="lower right")
    for x, pf in zip(xs, p_fluent):
        ax.text(x, pf + 1, f"{pf:.1f}%", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "transition_stacked_bar.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved transition_stacked_bar.png")

    # Disfluent-only transitions (no fluent column or row)
    disf_only = ["Prolongation", "Block", "PWR", "WWR", "Combined"]
    P_disf = np.zeros((len(disf_only), len(disf_only)))
    for i, a in enumerate(disf_only):
        denom = sum(trans.get((a, b), 0) for b in disf_only)
        if denom == 0:
            continue
        for j, b in enumerate(disf_only):
            P_disf[i, j] = trans.get((a, b), 0) / denom
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(P_disf, cmap="Purples", aspect="equal", vmin=0, vmax=1)
    ax.set_xticks(range(len(disf_only)))
    ax.set_yticks(range(len(disf_only)))
    ax.set_xticklabels(disf_only, rotation=30, ha="right")
    ax.set_yticklabels(disf_only)
    ax.set_xlabel("Next disfluent state")
    ax.set_ylabel("Current disfluent state")
    ax.set_title("Disfluent → disfluent transition probability "
                 "(fluent excluded)")
    for i in range(len(disf_only)):
        for j in range(len(disf_only)):
            v = P_disf[i, j]
            if v > 0.005:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 0.5 else "black",
                        fontsize=9)
    fig.colorbar(im, ax=ax, label="P(next | current, both disfluent)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "transition_disfluent_only.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved transition_disfluent_only.png")

    # ── Run lengths of consecutive disfluent words ──
    run_counts = Counter(runs)
    fig, ax = plt.subplots(figsize=(8, 5))
    y = [
        run_counts.get(1, 0),
        run_counts.get(2, 0),
        run_counts.get(3, 0),
        run_counts.get(4, 0),
        sum(v for k, v in run_counts.items() if k >= 5),
    ]
    labels = ["1\n(isolated)", "2", "3", "4", "5+"]
    bars = ax.bar(labels, y, color="#1f77b4", edgecolor="white")
    total = sum(y)
    for bar, val in zip(bars, y):
        pct = val / total * 100 if total else 0
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.01,
                f"{val}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Consecutive disfluent words")
    ax.set_ylabel("Number of runs")
    ax.set_title(f"Run-length distribution of consecutive "
                 f"disfluent words — SLASS Jason matrices (n={total} runs)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "stutter_run_lengths.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved stutter_run_lengths.png")

    # ── UNWR group breakdown (NEW: adult cohort) ──
    if unwr:
        print("\nUNWR group breakdown...")
        groups_order = ["Control", "Attention", "Stutter", "Attention+Stutter"]
        present = [g for g in groups_order
                   if any(s["group"] == g for s in unwr)]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

        # Panel 1: speaker counts per group
        counts = [sum(1 for s in unwr if s["group"] == g) for g in present]
        bars = axes[0].bar(present, counts,
                           color=[UNWR_GROUP_COLOURS.get(g, "#999")
                                  for g in present], edgecolor="white")
        for bar, n in zip(bars, counts):
            axes[0].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.2, str(n),
                         ha="center", fontsize=10)
        axes[0].set_ylabel("Speakers")
        axes[0].set_title("UNWR speakers per group")
        axes[0].tick_params(axis="x", rotation=20)

        # Panel 2: SSI % stuttered syllables by group (strip/box-ish)
        for i, g in enumerate(present):
            vals = [s["ssi_pct"] for s in unwr
                    if s["group"] == g and s["ssi_pct"] is not None]
            if not vals:
                continue
            xs = np.full(len(vals), i) + np.random.uniform(
                -0.15, 0.15, size=len(vals))
            axes[1].scatter(xs, vals, color=UNWR_GROUP_COLOURS.get(g, "#999"),
                            alpha=0.7, edgecolor="white", s=40)
            mean = sum(vals) / len(vals)
            axes[1].plot([i - 0.25, i + 0.25], [mean, mean],
                         color="black", linewidth=2)
        axes[1].set_xticks(range(len(present)))
        axes[1].set_xticklabels(present, rotation=20)
        axes[1].set_ylabel("SSI % stuttered syllables")
        axes[1].set_title("SSI severity by group (adult cohort)")
        axes[1].axhline(3, color="grey", linestyle="--", linewidth=0.8)
        axes[1].text(len(present) - 0.5, 3.1,
                     "mild/moderate threshold (3%)",
                     fontsize=8, color="grey", ha="right")

        # Panel 3: age distribution by group
        for i, g in enumerate(present):
            ages = [s["age"] for s in unwr
                    if s["group"] == g and s["age"] is not None]
            if not ages:
                continue
            xs = np.full(len(ages), i) + np.random.uniform(
                -0.15, 0.15, size=len(ages))
            axes[2].scatter(xs, ages, color=UNWR_GROUP_COLOURS.get(g, "#999"),
                            alpha=0.7, edgecolor="white", s=40)
            mean = sum(ages) / len(ages)
            axes[2].plot([i - 0.25, i + 0.25], [mean, mean],
                         color="black", linewidth=2)
        axes[2].set_xticks(range(len(present)))
        axes[2].set_xticklabels(present, rotation=20)
        axes[2].set_ylabel("Age (years)")
        axes[2].set_title("Age distribution by group")
        axes[2].axhline(18, color="grey", linestyle="--", linewidth=0.8)
        axes[2].text(len(present) - 0.5, 18.5, "18 (adult)",
                     fontsize=8, color="grey", ha="right")

        plt.tight_layout()
        plt.savefig(OUT_DIR / "unwr_groups.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved unwr_groups.png")

    # ── Dataset summary table ──
    print("\nDataset summary...")
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.axis("off")
    slass_full_spk = f"≥{n_spk}*" if n_spk else "—"
    unwr_row_speakers = str(len(unwr)) if unwr else "—"
    unwr_row_hours = f"{unwr_hours:.1f}" if unwr else "—"
    unwr_row_utts = (str(sum(1 for s in unwr if s["total_audio_s"] > 0))
                     if unwr else "—")
    table_data = [
        ["SLASS (curated subset)", "148", "9.1", "62", "28,138", "6,429"],
        ["SLASS (full archive)", "9,780", "329.6", slass_full_spk, "—", "—"],
        ["UCLASS", "304", "10.2", "120", "989", "0"],
        ["FluencyBank", "332", "46.9", "113", "61,033", "0"],
        ["UNWR (adult SSI)", unwr_row_speakers, unwr_row_hours,
         unwr_row_speakers, unwr_row_utts, unwr_row_utts + "**"],
        ["LibriSpeech (test+dev)", "11,126", "21.2", "146",
         "11,126", "11,126"],
    ]
    table = ax.table(
        cellText=table_data,
        colLabels=["Dataset", "Sessions", "Hours", "Speakers",
                   "Words/Utts", "With stutter type"],
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    for j in range(6):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("Dataset summary", fontsize=13, pad=20)
    notes = []
    if n_spk:
        notes.append(
            f"* Minimum speaker estimate for SLASS full archive: "
            f"{n_named}/{n_total} filenames carry a parseable participant "
            f"code; the remaining {n_total - n_named} use opaque schemas "
            f"(e.g. '0050c116') requiring a master-roster join.")
    if unwr:
        notes.append(
            "** UNWR stutter labels are session-level SSI % stuttered "
            "syllables, not per-word annotations. All 58 speakers have an "
            "SSI score; adult cohort, age range 18-60.")
    for i, note in enumerate(notes):
        ax.text(0.5, -0.05 - i * 0.08, note,
                transform=ax.transAxes, ha="center",
                fontsize=8, style="italic")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "dataset_summary.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved dataset_summary.png")

    print(f"\nAll figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
