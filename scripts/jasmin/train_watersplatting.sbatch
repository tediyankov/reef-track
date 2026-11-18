#!/bin/bash
# Train WaterSplatting on one exported scene, then render its held-out (test) views.
#   sbatch ~/reefrender/jasmin/train_watersplatting.sbatch valerio_GX020083_loop
#   WS_STEPS=30000 sbatch ~/reefrender/jasmin/train_watersplatting.sbatch <scene>   # longer training
# Outputs go to outputs/watersplatting/<scene> (or <scene>_<N>steps when WS_STEPS is set).
#SBATCH --job-name=ws
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.log
set -eo pipefail   # no -u: conda's activate hooks read unset variables

SCENE=${1:?usage: sbatch train_watersplatting.sbatch <scene name>}
WORK=${REEFRENDER:-$HOME/reefrender}
DATA=$WORK/scenes/$SCENE
OUT=$WORK/outputs/watersplatting
source "${REEFRENDER_CONDA:-$WORK/miniforge3}/etc/profile.d/conda.sh"
export TMPDIR=$WORK/tmp XDG_CACHE_HOME=$WORK/cache TORCH_EXTENSIONS_DIR=$WORK/cache/torch_extensions
mkdir -p "$TMPDIR"
conda activate "${REEFRENDER_CONDA:-$WORK/miniforge3}/envs/watersplatting"
nvidia-smi --query-gpu=name,memory.total --format=csv
cd "$WORK/code/water-splatting"

# test split, as written by `reeftrack export-colmap`: interval (every 8th) or filename (train_/eval_)
EVAL_MODE=$(python -c "import json; print(json.load(open('$DATA/export.json')).get('eval_mode', 'interval'))")
echo "eval mode: $EVAL_MODE"

# training length: the method's default is 15k steps. WS_STEPS=N trains N steps, densifies until
# 2/3 of N (default: 10k of 15k); learning rates still finish decaying at 15k, then stay at their
# final values (fine-tuning)
NAME=$SCENE
STEP_ARGS=()
if [ -n "$WS_STEPS" ]; then
  NAME=${SCENE}_${WS_STEPS}steps
  STEP_ARGS=(--max-num-iterations "$WS_STEPS" --pipeline.model.num-steps "$WS_STEPS"
             --pipeline.model.stop-split-at $((WS_STEPS * 2 / 3)))
fi
echo "run: $NAME ${STEP_ARGS[*]}"

# no live viewer on a batch node
ns-train water-splatting "${STEP_ARGS[@]}" \
  --output-dir "$OUT" --experiment-name "$NAME" --timestamp run \
  --vis tensorboard --viewer.quit-on-train-completion True \
  colmap --data "$DATA" --colmap-path sparse/0 --images-path images --downscale-factor 1 \
  --eval-mode "$EVAL_MODE"

CFG=$OUT/$NAME/water-splatting/run/config.yml
# held-out views: with water (rgb) and with the water removed (rgb_clear)
ns-render dataset --load-config "$CFG" --split test --rendered-output-names rgb rgb_clear \
  --image-format png --output-path "$OUT/$NAME/renders"
ns-eval --load-config "$CFG" --output-path "$OUT/$NAME/metrics.json"
echo "done: $OUT/$NAME/renders"
