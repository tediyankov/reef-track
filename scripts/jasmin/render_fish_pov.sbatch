#!/bin/bash
# Fish-POV swim-through render of a trained WaterSplatting scene (a few seconds, renders in ~1 min).
#   sbatch ~/reefrender/jasmin/render_fish_pov.sbatch valerio_GX020083_loop
# Needs ~/reefrender/scenes/<scene>/fish_pov/fish_pov_path.json (from the laptop:
#   python -m reeftrack.render.fish_pov data/runs/<scene> --export data/render_exports/<scene>)
# Output: ~/reefrender/outputs/watersplatting/<model>/fish_pov/fish_pov_{rgb,rgb_clear}.mp4
# MODEL=<name> picks another trained model of the scene (e.g. valerio_GX020083_loop_L_30000steps).
#SBATCH --job-name=fishpov
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.log
set -eo pipefail   # no -u: conda's activate hooks read unset variables

SCENE=${1:?usage: sbatch render_fish_pov.sbatch <scene name>}
WORK=${REEFRENDER:-$HOME/reefrender}
MODEL=${MODEL:-$SCENE}
OUT=$WORK/outputs/watersplatting/$MODEL
source "${REEFRENDER_CONDA:-$WORK/miniforge3}/etc/profile.d/conda.sh"
export TMPDIR=$WORK/tmp XDG_CACHE_HOME=$WORK/cache TORCH_EXTENSIONS_DIR=$WORK/cache/torch_extensions
conda activate "${REEFRENDER_CONDA:-$WORK/miniforge3}/envs/watersplatting"
nvidia-smi --query-gpu=name,memory.total --format=csv
cd "$WORK/code/water-splatting"   # config.yml paths are relative to where training ran
python -u "$WORK/jasmin/render_fish_pov.py" "$OUT/water-splatting/run/config.yml" "$WORK/scenes/$SCENE" \
  "$WORK/scenes/$SCENE/fish_pov/fish_pov_path.json" "$OUT/fish_pov"
