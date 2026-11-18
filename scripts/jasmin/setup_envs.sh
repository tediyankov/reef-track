#!/bin/bash
# One-off setup on JASMIN: Miniforge + two conda environments + both methods' code.
#   watersplatting : WaterSplatting (3DV 2025) on nerfstudio 1.1.4, PyTorch 2.1.2 / CUDA 11.8
#   i2nerf         : I2-NeRF (NeurIPS 2025, ZipNeRF-based), PyTorch 2.1.2 / CUDA 11.8
#
# Run it on any JASMIN sci server or on gpuhost001 (no GPU needed to *build*: the A100's
# architecture is given explicitly). It takes ~45-60 min (tiny-cuda-nn compiles slowly).
# Re-running is safe: finished steps are skipped.
#
#   bash ~/reefrender/jasmin/setup_envs.sh
set -eo pipefail   # no -u: conda's (de)activate hooks read unset variables such as ZSH_VERSION

WORK=${REEFRENDER:-$HOME/reefrender}
CODE=$WORK/code
CONDA_DIR=${REEFRENDER_CONDA:-$WORK/miniforge3}
# everything big lives in $WORK, nothing in the (small) home directory: conda + envs, pip and
# conda download caches, and the temporary files of pip / compilers
export TMPDIR=$WORK/tmp TEMP=$WORK/tmp TMP=$WORK/tmp
export PIP_CACHE_DIR=$WORK/cache/pip CONDA_PKGS_DIRS=$WORK/cache/conda_pkgs XDG_CACHE_HOME=$WORK/cache
mkdir -p "$CODE" "$TMPDIR" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS"
export TCNN_CUDA_ARCHITECTURES=80        # A100 (ORCHID)
export TORCH_CUDA_ARCH_LIST="8.0"

# ---- conda (its own Miniforge in $WORK; independent of any conda you already have) -----------
if [ ! -x "$CONDA_DIR/bin/conda" ]; then
  echo "== installing Miniforge in $CONDA_DIR"
  cd "$TMPDIR"
  curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash Miniforge3-Linux-x86_64.sh -b -p "$CONDA_DIR"
fi
source "$CONDA_DIR/etc/profile.d/conda.sh"
# environments by full path: a ~/.condarc 'envs_dirs' (e.g. ~/envs) can then never redirect them
WS_ENV=$CONDA_DIR/envs/watersplatting
I2_ENV=$CONDA_DIR/envs/i2nerf
export CONDA_ENVS_DIRS=$CONDA_DIR/envs

# ---- WaterSplatting ----------------------------------------------------------------------------
if [ ! -d "$WS_ENV/conda-meta" ]; then
  echo "== creating env: $WS_ENV"
  conda create -y -p "$WS_ENV" python=3.8
fi
conda activate "$WS_ENV"
export CUDA_HOME=$CONDA_PREFIX
if ! python -c "import water_splatting" 2>/dev/null; then
  python -m pip install --upgrade pip
  pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
  conda install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit
  pip install ninja "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
  pip install nerfstudio==1.1.4
  [ -d "$CODE/water-splatting" ] || git clone --recursive https://github.com/water-splatting/water-splatting.git "$CODE/water-splatting"
  (cd "$CODE/water-splatting" && git submodule update --init --recursive && pip install --no-use-pep517 -e .)
  ns-install-cli || true
fi
python -c "import torch, water_splatting, nerfstudio; print('watersplatting OK, torch', torch.__version__)"
conda deactivate

# ---- I2-NeRF -----------------------------------------------------------------------------------
if [ ! -d "$I2_ENV/conda-meta" ]; then
  echo "== creating env: $I2_ENV"
  conda create -y -p "$I2_ENV" python=3.9
fi
conda activate "$I2_ENV"
export CUDA_HOME=$CONDA_PREFIX
[ -d "$CODE/I2-NeRF" ] || git clone https://github.com/ShuhongLL/I2-NeRF.git "$CODE/I2-NeRF"
if ! python -c "import torch; import _cuda_backend" 2>/dev/null; then
  python -m pip install --upgrade pip
  pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
  conda install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit
  # their requirements leave torch unpinned: keep the CUDA 11.8 build installed above. Also skip
  # transformers: their code never imports it, and current versions break torch 2.1 (torch.onnx
  # imports transformers when it is installed and fails on torch.utils._pytree)
  grep -viE '^(torch|torchvision|transformers)([=<> ]|$)' "$CODE/I2-NeRF/requirements.txt" > "$TMPDIR/i2nerf_reqs.txt"
  pip install -r "$TMPDIR/i2nerf_reqs.txt"
  # imported by their code but missing from requirements.txt (torch pinned so pip can't swap it)
  pip install torchmetrics torch==2.1.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
  pip install torch_scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
  # their build file asks for C++14 on Linux, but PyTorch 2.x headers need C++17
  sed -i 's/-std=c++14/-std=c++17/g' "$CODE/I2-NeRF/extensions/cuda/setup.py"
  # build inside this env (it has torch); pip's default isolated build env does not
  (cd "$CODE/I2-NeRF" && pip install --no-build-isolation ./extensions/cuda)
fi
cp "$(dirname "$0")/i2nerf_reef.gin" "$CODE/I2-NeRF/configs/reef_uw.gin"
cp "$(dirname "$0")/i2nerf_reef_ff.gin" "$CODE/I2-NeRF/configs/reef_uw_ff.gin"
# fetch the LPIPS metric's backbone now (GPU nodes may have no internet); jobs use the same cache
python -c "from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as L; L(normalize=True)"
python -c "import torch, torchmetrics; print('i2nerf OK, torch', torch.__version__)"
conda deactivate

echo "== setup finished. Next: submit the jobs (see docs/rendering_jasmin.md, step 4)."
