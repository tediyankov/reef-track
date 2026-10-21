#!/usr/bin/env bash
# Re-run tracking on a GPU from the clicks saved in tracking/tracklet.json.
# Typical use: click the fish in the GUI on your laptop (fast, interactive), copy the run's
# tracking/tracklet.json to JASMIN, then replay with the stronger SAM 3 model at full speed.
#   sbatch scripts/jasmin/replay_tracking.slurm data/runs/<run> [config.yaml]
#SBATCH --job-name=reef-track
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
RUN_DIR="${1:?run dir}"
CFG="${2:-}"
source scripts/jasmin/env.sh
reeftrack replay "$RUN_DIR" ${CFG:+-c "$CFG"}
