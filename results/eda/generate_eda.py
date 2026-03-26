#!/usr/bin/env python3
"""
Exploratory Data Analysis of stuttered speech datasets.

Generates individual figures for:
    1. KDE duration plots by stutter type (per dataset + aggregated)
    2. Stutter type distributions
    3. Stuttering rate by syllable count
    4. Content vs function word analysis
    5. Most commonly stuttered words
    6. Cross-dataset duration comparison
    7. Dataset summary

Usage:
    python docs/eda/generate_eda.py
"""

import csv
import statistics
from collections import Counter
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
}

plt.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


# ── Data loading ─────────────────────────────────────────────────────────────

def load_jason_matrices_with_duration():
    """Load Jason usage matrices and compute word durations from timestamps."""
    matrix_dir = Path("/Volumes/SPEECH/from_jason/Speech data Jason/output")
    all_words = []

    for f in sorted(matrix_dir.glob("*_Usage_Matrix.csv")):
        rows = []
        with open(f, encoding="latin-1") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                ts = row.get("Timestamp", "")
                try:
                    ts_val = float(ts) if ts else None
                except ValueError:
                    ts_val = None
                rows.append({
                    "file": f.stem,
                    "word": row.get("Word", ""),
                    "fluency": row.get("Fluency", ""),
                    "stutter_type": row.get("Stutter_Type", ""),
                    "syllable": int(row.get("Syllable", "1") or "1"),
                    "word_type": row.get("Word_Type", ""),
                    "ts": ts_val,
                })

        # Compute duration from consecutive timestamps
        for i in range(len(rows) - 1):
            if rows[i]["ts"] is not None and rows[i + 1]["ts"] is not None:
                dur = rows[i + 1]["ts"] - rows[i]["ts"]
                if 0 < dur < 30:
                    rows[i]["duration"] = dur
                    all_words.append(rows[i])

    return all_words


def load_standardised(corpus_dir: Path):
    """Load standardised CSV files."""
    records = []
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


def simplify_jason(st: str) -> str:
    """Simplify Jason stutter types to standard categories."""
    st = st.strip()
    if st == "Fluent" or not st:
        return "Fluent"
    if st == "Unknown":
        return "Unknown"
    pure_map = {"Block": "Block", "Prolongation": "Prolongation", "PWR": "PWR", "WWR": "WWR"}
    for key, val in pure_map.items():
        others = [k for k in pure_map if k != key]
        if key in st and not any(o in st for o in others):
            return val
    return "Combined"


# ── KDE plot ─────────────────────────────────────────────────────────────────

def plot_kde(durations_by_type, title, filename, x_max=2.0,
             type_order=("Fluent", "Prolongation", "PWR", "WWR", "Block", "Combined")):
    """Plot KDE of durations by type."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for stype in type_order:
        durs = durations_by_type.get(stype, [])
        # Filter to reasonable range for KDE
        durs = [d for d in durs if 0.01 < d < x_max * 1.5]
        if len(durs) < 5:
            continue
        try:
            kde = gaussian_kde(durs, bw_method=0.3)
            x = np.linspace(0, x_max, 500)
            y = kde(x)
            colour = COLOURS.get(stype, "#999999")
            ax.plot(x, y, label=f"{stype} (n={len(durs)})", color=colour, linewidth=2)
            ax.fill_between(x, y, alpha=0.15, color=colour)
        except Exception:
            pass

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Probability Density")
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
    jason = load_jason_matrices_with_duration()
    print(f"  Jason matrices: {len(jason)} words with duration")

    uclass = load_standardised(Path("/Volumes/FATSPEECH/standardised/uclass"))
    print(f"  UCLASS: {len(uclass)} records")

    fb = load_standardised(Path("/Volumes/FATSPEECH/standardised/fluencybank"))
    print(f"  FluencyBank: {len(fb)} records")

    # ── KDE: SLASS (Jason matrices — proper word-level durations with WWR) ──
    print("\nKDE plots...")

    jason_dur = {}
    for w in jason:
        stype = simplify_jason(w["stutter_type"])
        if stype != "Unknown":
            jason_dur.setdefault(stype, []).append(w["duration"])

    plot_kde(jason_dur,
             "Word duration by disfluency type — SLASS",
             "kde_duration_slass.png", x_max=2.0)

    # ── KDE: UCLASS ──
    uclass_durs = {"UCLASS (all words)": [r["duration"] for r in uclass
                    if r.get("duration") and 0.01 < r["duration"] < 5]}
    plot_kde(uclass_durs,
             "Word duration distribution — UCLASS",
             "kde_duration_uclass.png", x_max=2.0,
             type_order=("UCLASS (all words)",))

    # ── KDE: FluencyBank (utterance-level) ──
    fb_durs = {"FluencyBank (utterances)": [r["duration"] for r in fb
                if r.get("duration") and 0.1 < r["duration"] < 60]}
    plot_kde(fb_durs,
             "Utterance duration distribution — FluencyBank",
             "kde_duration_fluencybank.png", x_max=30.0,
             type_order=("FluencyBank (utterances)",))

    # ── KDE: Aggregated (SLASS with all stutter types) ──
    plot_kde(jason_dur,
             "Word duration by disfluency type — Aggregated (SLASS)",
             "kde_duration_aggregated.png", x_max=2.0)

    # ── KDE: Cross-dataset ──
    fig, ax = plt.subplots(figsize=(10, 6))
    cross = {
        "SLASS fluent": [w["duration"] for w in jason
                         if simplify_jason(w["stutter_type"]) == "Fluent" and 0.01 < w["duration"] < 3],
        "SLASS stuttered": [w["duration"] for w in jason
                            if simplify_jason(w["stutter_type"]) != "Fluent"
                            and simplify_jason(w["stutter_type"]) != "Unknown"
                            and 0.01 < w["duration"] < 3],
        "UCLASS (all)": [r["duration"] for r in uclass
                         if r.get("duration") and 0.01 < r["duration"] < 3],
    }
    cross_colours = {"SLASS fluent": "#1f77b4", "SLASS stuttered": "#d62728", "UCLASS (all)": "#2ca02c"}
    for name in ("SLASS fluent", "SLASS stuttered", "UCLASS (all)"):
        durs = cross[name]
        if len(durs) < 5:
            continue
        kde = gaussian_kde(durs, bw_method=0.3)
        x = np.linspace(0, 2.0, 500)
        y = kde(x)
        ax.plot(x, y, label=f"{name} (n={len(durs)})", color=cross_colours[name], linewidth=2)
        ax.fill_between(x, y, alpha=0.1, color=cross_colours[name])
    ax.set_xlabel("Duration (s)")
    ax.set_ylabel("Probability Density")
    ax.set_title("Word duration distribution across datasets")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 2.0)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "kde_duration_cross_dataset.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved kde_duration_cross_dataset.png")

    # ── Stutter type distribution ──
    print("\nStutter type distributions...")

    jason_types = Counter(simplify_jason(w["stutter_type"])
                          for w in jason if w["fluency"] == "Stuttered")
    labels = ["Prolongation", "Block", "PWR", "WWR", "Combined"]
    values = [jason_types.get(l, 0) for l in labels]
    total = sum(values)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=[COLOURS[l] for l in labels], edgecolor="white")
    ax.set_ylabel("Count")
    ax.set_title(f"Disfluency type distribution — SLASS (n={total} stuttered words)")
    for bar, val in zip(bars, values):
        pct = val / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.01,
                f"{val}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "type_distribution_slass.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved type_distribution_slass.png")

    # ── Syllable analysis ──
    print("\nSyllable analysis...")

    fig, ax = plt.subplots(figsize=(7, 5))
    for s in range(1, 6):
        tot = sum(1 for w in jason if w["syllable"] == s)
        stutt = sum(1 for w in jason if w["syllable"] == s and w["fluency"] == "Stuttered")
        if tot > 10:
            rate = stutt / tot * 100
            bar = ax.bar(s, rate, color="#1f77b4", edgecolor="white")
            ax.text(s, rate + 0.3, f"{rate:.1f}%\n(n={tot})", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Syllable count")
    ax.set_ylabel("Stuttering rate (%)")
    ax.set_title("Stuttering rate by syllable count — SLASS")
    ax.set_xticks(range(1, 5))
    plt.tight_layout()
    plt.savefig(OUT_DIR / "syllable_rate_slass.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved syllable_rate_slass.png")

    # ── Word type analysis ──
    print("\nWord type analysis...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, wt in enumerate(["Content", "Function"]):
        tot = sum(1 for w in jason if w["word_type"] == wt)
        stutt = sum(1 for w in jason if w["word_type"] == wt and w["fluency"] == "Stuttered")
        rate = stutt / tot * 100 if tot else 0
        axes[0].bar(wt, rate, color=["#2ca02c", "#ff7f0e"][i], edgecolor="white")
        axes[0].text(i, rate + 0.3, f"{rate:.1f}%\n(n={tot})", ha="center", fontsize=10)
    axes[0].set_ylabel("Stuttering rate (%)")
    axes[0].set_title("Stuttering rate by word type")

    labels = ["Prolongation", "Block", "PWR", "WWR", "Combined"]
    x = np.arange(len(labels))
    width = 0.35
    for j, wt in enumerate(["Content", "Function"]):
        stutt_words = [w for w in jason if w["word_type"] == wt and w["fluency"] == "Stuttered"]
        tot = len(stutt_words)
        vals = [sum(1 for w in stutt_words if simplify_jason(w["stutter_type"]) == l) / tot * 100
                if tot else 0 for l in labels]
        axes[1].bar(x + j * width - width / 2, vals, width,
                    label=wt, color=["#2ca02c", "#ff7f0e"][j], edgecolor="white")
    axes[1].set_ylabel("Proportion (%)")
    axes[1].set_title("Disfluency type by word type")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "word_type_slass.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved word_type_slass.png")

    # ── Most stuttered words ──
    print("\nMost stuttered words...")

    fig, ax = plt.subplots(figsize=(10, 6))
    stuttered = [w["word"].lower() for w in jason if w["fluency"] == "Stuttered" and w["word"]]
    wc = Counter(stuttered).most_common(20)
    words, counts = zip(*wc)
    y_pos = range(len(words))
    ax.barh(y_pos, counts, color="#1f77b4", edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title(f"Most frequently stuttered words — SLASS (n={len(stuttered)})")
    for i, c in enumerate(counts):
        ax.text(c + 1, i, str(c), va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "most_stuttered_words_slass.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved most_stuttered_words_slass.png")

    # ── Dataset summary table ──
    print("\nDataset summary...")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    table_data = [
        ["SLASS (curated)", "148", "9.1", "62", "28,138", "6,429"],
        ["SLASS (full archive)", "9,780", "329.6", "—", "—", "—"],
        ["UCLASS", "304", "10.2", "120", "989", "0"],
        ["FluencyBank", "332", "46.9", "113", "61,033", "0"],
        ["LibriSpeech", "11,126", "21.2", "146", "11,126", "11,126"],
    ]
    table = ax.table(
        cellText=table_data,
        colLabels=["Dataset", "Sessions", "Hours", "Speakers", "Words/Utts", "With stutter type"],
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    for j in range(6):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("Dataset summary", fontsize=13, pad=20)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "dataset_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved dataset_summary.png")

    print(f"\nAll figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
