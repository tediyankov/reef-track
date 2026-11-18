#!/bin/bash
# Train I2-NeRF on one exported scene, then render + score its held-out (test) views.
#   sbatch ~/reefrender/jasmin/train_i2nerf.sbatch valerio_GX020083_loop
#   I2_MODE=ff sbatch --export=ALL ~/reefrender/jasmin/train_i2nerf.sbatch <scene>   # fallback mode
#SBATCH --job-name=i2nerf
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.log
set -eo pipefail   # no -u: conda's activate hooks read unset variables

SCENE=${1:?usage: sbatch train_i2nerf.sbatch <scene name>}
WORK=${REEFRENDER:-$HOME/reefrender}
DATA=$WORK/scenes/$SCENE
source "${REEFRENDER_CONDA:-$WORK/miniforge3}/etc/profile.d/conda.sh"
export TMPDIR=$WORK/tmp XDG_CACHE_HOME=$WORK/cache TORCH_EXTENSIONS_DIR=$WORK/cache/torch_extensions
mkdir -p "$TMPDIR"
conda activate "${REEFRENDER_CONDA:-$WORK/miniforge3}/envs/i2nerf"
export WANDB_MODE=disabled   # train.py calls wandb.login(), which would wait for a prompt
nvidia-smi --query-gpu=name,memory.total --format=csv
cd "$WORK/code/I2-NeRF"

# I2_MODE=unbounded (default, any camera path) | ff (authors' forward-facing setup, transects)
MODE=${I2_MODE:-unbounded}
if [ "$MODE" = ff ]; then
  EXP=reef_ff/$SCENE
  ARGS=(--gin_configs=configs/reef_uw_ff.gin)
else
  EXP=reef/$SCENE
  # near / far of the reef in I2-NeRF's normalised units, measured when the scene was exported
  NEAR=$(python -c "import json; print(json.load(open('$DATA/export.json'))['i2nerf']['near'])")
  FAR=$(python -c "import json; print(json.load(open('$DATA/export.json'))['i2nerf']['far'])")
  ARGS=(--gin_configs=configs/reef_uw.gin
        --gin_bindings="Config.near = $NEAR" --gin_bindings="Config.far = $FAR")
  echo "near=$NEAR far=$FAR"
fi
ARGS+=(--gin_bindings="Config.data_dir = '$DATA'" --gin_bindings="Config.exp_name = '$EXP'")
OUT=$WORK/outputs/i2nerf_$MODE/$SCENE

python train.py "${ARGS[@]}"
python eval.py "${ARGS[@]}"          # writes exp/<EXP>/test_preds/color_NNN.png + metrics

mkdir -p "$OUT"
cp -r "exp/$EXP/test_preds" "$OUT/"
echo "done: $OUT/test_preds"
