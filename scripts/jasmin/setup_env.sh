#!/usr/bin/env bash
# One-off environment setup on JASMIN (run on a sci server, e.g. sci-vm-01, not the login node).
#
#   bash scripts/jasmin/setup_env.sh /gws/nopw/j04/<your-gws>/reef-track-cache
#
# Installs Miniforge (provides `mamba`) in $HOME if missing, creates the env from
# environment.yml, and points the Hugging Face cache at a group workspace (model weights are
# several GB; $HOME quota is small).
set -euo pipefail

CACHE_DIR="${1:?usage: setup_env.sh <cache dir on a group workspace>}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_PREFIX="$HOME/envs/reef-track"

if ! command -v mamba >/dev/null 2>&1; then
  if [[ ! -x "$HOME/miniforge3/bin/mamba" ]]; then
    curl -fsSL -o /tmp/miniforge.sh \
      "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
  fi
  export PATH="$HOME/miniforge3/bin:$PATH"
fi
eval "$(mamba shell hook --shell bash)"

if [[ ! -d "$ENV_PREFIX" ]]; then
  mamba env create --prefix "$ENV_PREFIX" -f "$REPO/environment.yml" -y
fi
mamba activate "$ENV_PREFIX"
pip install -e "$REPO" --no-deps
bash "$REPO/scripts/fetch_wat3r.sh"

mkdir -p "$CACHE_DIR"
cat > "$REPO/scripts/jasmin/env.sh" <<EOF
# sourced by the SLURM scripts
export PATH="\$HOME/miniforge3/bin:\$PATH"
eval "\$(mamba shell hook --shell bash)"
mamba activate "$ENV_PREFIX"
export HF_HOME="$CACHE_DIR/huggingface"
EOF
echo "Done. For SAM 3: request access at https://huggingface.co/facebook/sam3, then run 'hf auth login' once."
