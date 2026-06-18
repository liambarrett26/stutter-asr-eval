#!/usr/bin/env bash
#
# Migrate the project data from the portable FATSPEECH drive to the UCL RDSS
# governed store (approved by the UCL DPO, June 2026).
#
#   SRC = /Volumes/FATSPEECH                         (exFAT external, unencrypted)
#   DST = /Volumes/ritd-ag-project-rd02dw-lbarr63    (RDSS, SMB network share)
#
# The move is staged: copy -> verify -> (regenerate manifests) -> later
# secure-delete the source once the Linux GPU inference runs are complete.
# This script does the COPY and VERIFY only. It never deletes anything.
#
# Design notes
# ------------
# * Network share: rsync 3.x with --partial makes the transfer resumable at
#   the byte level; re-running picks up where an interrupted run stopped.
# * --no-perms/--no-owner/--no-group/--omit-dir-times: POSIX ownership and
#   FAT/SMB timestamps don't map cleanly across exFAT -> SMB and otherwise
#   spew errors. We compare by content (size, and --checksum in verify mode).
# * COPYFILE_DISABLE=1 stops macOS writing new ._AppleDouble sidecars onto the
#   share; existing ._ forks and other macOS cruft are excluded outright.
# * uchg immutable flags on the speech_sfs originals do NOT block reading, so
#   the copy is unaffected; they're only cleared at the secure-delete stage.
#
# Usage:
#   scripts/migrate_to_rdss.sh copy            # mirror SRC -> DST (resumable)
#   scripts/migrate_to_rdss.sh copy <dir>      # one top-level dir only
#   scripts/migrate_to_rdss.sh verify          # checksum-compare every file
#   scripts/migrate_to_rdss.sh reconcile       # quick file-count + byte totals
#
set -euo pipefail

SRC="/Volumes/FATSPEECH"
DST="/Volumes/ritd-ag-project-rd02dw-lbarr63"
RSYNC="/opt/homebrew/bin/rsync"          # modern rsync 3.x (not system openrsync)
LOG_DIR="${SRC}/scripts/migration_logs"

# Project data to move (everything; raw SFS + ingress zips included).
DIRS=(
  fluencybank librispeech manifests scripts slass slass_extended
  source_archives speech_sfs standardised store_metadata
  uclass unwr unwr_reliability
)

# macOS / drive cruft never copied to the governed store.
EXCLUDES=(
  --exclude='._*'
  --exclude='.DS_Store'
  --exclude='.Spotlight-V100'
  --exclude='.Trashes'
  --exclude='.fseventsd'
  --exclude='.TemporaryItems'
  --exclude='System Volume Information'
  --exclude='$RECYCLE.BIN'
  --exclude='~$*'                 # Office lock files (seen in speech_sfs)
  --exclude='migration_logs'      # don't copy our own logs
)

COMMON_FLAGS=(
  -rtv --partial --human-readable --info=progress2,stats2
  --no-perms --no-owner --no-group --omit-dir-times
  "${EXCLUDES[@]}"
)

export COPYFILE_DISABLE=1

mkdir -p "$LOG_DIR"
ts() { date +%Y%m%d_%H%M%S; }

require_mounts() {
  [[ -d "$SRC" ]] || { echo "SRC not mounted: $SRC" >&2; exit 1; }
  [[ -d "$DST" ]] || { echo "DST not mounted: $DST" >&2; exit 1; }
}

do_copy() {
  require_mounts
  local only="${1:-}"
  local targets=("${DIRS[@]}")
  [[ -n "$only" ]] && targets=("$only")
  for d in "${targets[@]}"; do
    [[ -d "$SRC/$d" ]] || { echo "skip (missing): $d"; continue; }
    echo "=== copy $d -> RDSS  ($(ts)) ==="
    mkdir -p "$DST/$d"
    "$RSYNC" "${COMMON_FLAGS[@]}" "$SRC/$d/" "$DST/$d/" \
      2>&1 | tee "$LOG_DIR/copy_${d}_$(ts).log"
  done
  # Root-level docs (README.md etc.) that live at the FATSPEECH top level, not
  # inside a listed subdir. -d = top level only, don't recurse.
  if [[ -z "$only" ]]; then
    echo "=== copy root-level docs -> RDSS  ($(ts)) ==="
    "$RSYNC" -dtv --no-perms --no-owner --no-group --omit-dir-times \
      "${EXCLUDES[@]}" --include='*.md' --exclude='*' \
      "$SRC/" "$DST/" 2>&1 | tee "$LOG_DIR/copy_root_$(ts).log"
  fi
  echo "=== copy pass complete ($(ts)) ==="
}

# Byte-for-byte verification: --checksum forces a full content hash compare,
# --dry-run + -i means it only REPORTS differences (transfers nothing). A clean
# run lists no files (only the run header/footer) => the copy is identical.
do_verify() {
  require_mounts
  local log="$LOG_DIR/verify_$(ts).log"
  echo "checksum-verifying all dirs -> $log"
  for d in "${DIRS[@]}"; do
    [[ -d "$SRC/$d" ]] || continue
    echo "=== verify $d ===" | tee -a "$log"
    "$RSYNC" -rin --checksum --no-perms --no-owner --no-group --omit-dir-times \
      "${EXCLUDES[@]}" "$SRC/$d/" "$DST/$d/" 2>&1 | tee -a "$log"
  done
  echo
  echo "Lines above starting with '>f' or 'cf' = files that DIFFER or are MISSING."
  echo "A clean verify prints only the '=== verify <dir> ===' headers."
}

# Cheaper sanity check: per-dir file count and total bytes on each side.
do_reconcile() {
  require_mounts
  printf '%-18s %12s %12s %14s %14s\n' DIR SRC_FILES DST_FILES SRC_BYTES DST_BYTES
  for d in "${DIRS[@]}"; do
    [[ -d "$SRC/$d" ]] || continue
    local sf df sb db
    sf=$(find "$SRC/$d" -type f ! -name '._*' ! -name '.DS_Store' | wc -l | tr -d ' ')
    df=$(find "$DST/$d" -type f ! -name '._*' ! -name '.DS_Store' 2>/dev/null | wc -l | tr -d ' ')
    sb=$(find "$SRC/$d" -type f ! -name '._*' ! -name '.DS_Store' -print0 | xargs -0 stat -f%z 2>/dev/null | awk '{s+=$1} END{print s+0}')
    db=$(find "$DST/$d" -type f ! -name '._*' ! -name '.DS_Store' -print0 2>/dev/null | xargs -0 stat -f%z 2>/dev/null | awk '{s+=$1} END{print s+0}')
    printf '%-18s %12s %12s %14s %14s\n' "$d" "$sf" "$df" "$sb" "$db"
  done
}

cmd="${1:-}"
case "$cmd" in
  copy)      do_copy "${2:-}";;
  verify)    do_verify;;
  reconcile) do_reconcile;;
  *) echo "usage: $0 {copy [dir] | verify | reconcile}" >&2; exit 2;;
esac
