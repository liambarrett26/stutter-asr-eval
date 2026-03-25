#!/usr/bin/env python3
"""
Download FluencyBank media files from media.talkbank.org.

Their server returns only 11 bytes by default (partial content), so we must
send an explicit Range header to get the full file.

Usage:
    python src/data/download_fluencybank_media.py --cookie "talkbank=s%3A..."

Get the cookie by:
1. Log in at https://media.talkbank.org/fluency/Voices-AWS/interview/
2. Open DevTools > Network tab > click any request > Request Headers > Cookie
"""

import argparse
import re
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError

MEDIA_BASE = "https://media.talkbank.org/fluency"
OUT_ROOT = Path("/Volumes/FATSPEECH/fluencybank/raw")

DOWNLOADS = [
    ("Voices-AWS", "interview"),
    ("Voices-AWS", "reading"),
    ("Voices-CWS", "interview"),
    ("Voices-CWS", "reading"),
    ("Voices-AWC", "interview"),
    ("Voices-AWC", "reading"),
    ("Voices-AWC", "stuttering"),
]


def get_file_list(corpus: str, subdir: str, cookie: str) -> list[str]:
    """Fetch directory listing and extract .mp4 filenames."""
    url = f"{MEDIA_BASE}/{corpus}/{subdir}/"
    req = Request(url, headers={"Cookie": cookie})
    try:
        with urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        names = list(dict.fromkeys(re.findall(r"[\w\d._-]+\.mp4", html)))
        return names
    except Exception as e:
        print(f"  ERROR listing {url}: {e}")
        return []


def download_file(url: str, out_path: Path, cookie: str) -> bool:
    """Download a single file using Range header workaround."""
    req = Request(url, headers={
        "Cookie": cookie,
        "Range": "bytes=0-99999999999",
    })
    try:
        with urlopen(req, timeout=300) as resp:
            with open(out_path, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)  # 1MB chunks
                    if not chunk:
                        break
                    f.write(chunk)
        return out_path.stat().st_size > 1000
    except Exception as e:
        print(f"    ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download FluencyBank media")
    parser.add_argument("--cookie", required=True, help="TalkBank cookie value")
    parser.add_argument("--corpus", help="Download only this corpus (e.g. Voices-AWS)")
    parser.add_argument("--subdir", help="Download only this subdir (e.g. interview)")
    args = parser.parse_args()

    cookie = args.cookie
    downloads = DOWNLOADS
    if args.corpus:
        downloads = [(c, s) for c, s in downloads if c == args.corpus]
    if args.subdir:
        downloads = [(c, s) for c, s in downloads if s == args.subdir]

    total_files = 0
    total_downloaded = 0
    total_skipped = 0
    total_failed = 0

    for corpus, subdir in downloads:
        out_dir = OUT_ROOT / corpus / "media" / subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== {corpus}/{subdir} ===")
        files = get_file_list(corpus, subdir, cookie)
        if not files:
            print("  No files found (auth expired?)")
            continue

        print(f"  {len(files)} files to download")
        total_files += len(files)

        for i, fname in enumerate(files, 1):
            out_path = out_dir / fname
            if out_path.exists() and out_path.stat().st_size > 1000:
                print(f"  [{i}/{len(files)}] {fname}: exists ({out_path.stat().st_size // 1024 // 1024}MB), skip")
                total_skipped += 1
                continue

            url = f"{MEDIA_BASE}/{corpus}/{subdir}/{fname}"
            print(f"  [{i}/{len(files)}] {fname}: downloading...", end="", flush=True)
            t0 = time.time()

            if download_file(url, out_path, cookie):
                size_mb = out_path.stat().st_size / 1024 / 1024
                elapsed = time.time() - t0
                speed = size_mb / elapsed if elapsed > 0 else 0
                print(f" {size_mb:.0f}MB in {elapsed:.0f}s ({speed:.1f}MB/s)")
                total_downloaded += 1
            else:
                print(" FAILED")
                total_failed += 1

    print(f"\n=== SUMMARY ===")
    print(f"Total files: {total_files}")
    print(f"Downloaded: {total_downloaded}")
    print(f"Skipped (existing): {total_skipped}")
    print(f"Failed: {total_failed}")


if __name__ == "__main__":
    main()
