#!/bin/bash
# Submit both methods for every scene in ~/reefrender/scenes (or the scenes named as arguments).
#   bash ~/reefrender/jasmin/submit_all.sh                      # all scenes
#   bash ~/reefrender/jasmin/submit_all.sh valerio_GX020083_loop
set -euo pipefail
WORK=${REEFRENDER:-$HOME/reefrender}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$WORK/logs" && cd "$WORK/logs"
SCENES=("$@")
[ ${#SCENES[@]} -gt 0 ] || SCENES=($(ls "$WORK/scenes"))
for s in "${SCENES[@]}"; do
  sbatch "$HERE/train_watersplatting.sbatch" "$s"
  sbatch "$HERE/train_i2nerf.sbatch" "$s"
done
squeue -u "$USER"
