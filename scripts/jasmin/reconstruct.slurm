#!/usr/bin/env bash
# 3D reef reconstruction on a JASMIN ORCHID GPU node.
#   sbatch scripts/jasmin/reconstruct.slurm data/runs/<run> [configs/runs/<run>.yaml]
# Check current ORCHID partition/account/QoS names in the JASMIN docs if submission fails.
#SBATCH --job-name=reef-recon
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
RUN_DIR="${1:?run dir}"
CFG="${2:-}"
source scripts/jasmin/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
reeftrack reconstruct "$RUN_DIR" ${CFG:+-c "$CFG"}
reeftrack fuse "$RUN_DIR" ${CFG:+-c "$CFG"} || echo "no tracklet yet - skipped fusion"
