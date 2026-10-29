#!/usr/bin/env bash
# Pull the pipeline inputs from Google Drive onto this machine.
#
# Lightning AI (where everything used to run) is gone, so Drive is now the only
# copy of the v4 run. This fetches just what the Stage 4 clustering needs —
# about 400 MB — rather than the whole bucket, which is tens of GB of tiles.
#
#   ./motifs/fetch_from_drive.sh              # embeddings + master table (~400 MB)
#   ./motifs/fetch_from_drive.sh --graphs     # also the per-image graphs (slow)
#
# Everything lands under $CELLPOSE_LOCAL_ROOT (default ~/cellpose_work), which
# is the same layout every script already expects.

set -euo pipefail

ROOT="${CELLPOSE_LOCAL_ROOT:-$HOME/cellpose_work}"
export RCLONE_CONFIG="${RCLONE_CONFIG:-$HOME/rclone-config/rclone.conf}"
DEST="gdrive:Fusion AI/Prof Huang Project/Cellpose feature extractions"
FLAGS=(--transfers=8 --checkers=16 --progress)

command -v rclone >/dev/null || {
    echo "rclone not installed:  brew install rclone" >&2; exit 1; }
[ -f "$RCLONE_CONFIG" ] || {
    echo "no rclone config at $RCLONE_CONFIG" >&2; exit 1; }

echo "Fetching into $ROOT"
mkdir -p "$ROOT/v4/stage3"

# The embedding is the one file Stage 4 cannot be rebuilt without.
echo "==> v4/stage3 (embeddings + encoder, ~400 MB)"
rclone copy "$DEST/v4/stage3" "$ROOT/v4/stage3" "${FLAGS[@]}" \
    --include "embeddings.parquet" --include "encoder*.pt" --include "config.json"

echo "==> master_table.csv + normalization_stats.json"
rclone copy "$DEST" "$ROOT" "${FLAGS[@]}" \
    --include "master_table.csv" --include "normalization_stats.json"

if [ "${1:-}" = "--graphs" ]; then
    echo "==> graphs/ (per-image .pt; needed only for spatial validation)"
    rclone copy "$DEST/graphs" "$ROOT/graphs" "${FLAGS[@]}"
fi

echo
echo "Done. Local tree:"
du -sh "$ROOT"/* 2>/dev/null || true
