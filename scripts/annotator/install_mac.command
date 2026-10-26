#!/bin/bash
# One-time setup of the reef-track annotation tool on a Mac. Run it from Terminal:
#     bash <drag this file here>
# (macOS blocks double-clicking downloaded scripts; alternatively, after the warning use
#  System Settings -> Privacy & Security -> Open Anyway.)
#
# What it does (safe to run again, e.g. after downloading a newer version of reef-track):
#   1. installs Miniforge (a small Python distribution) in ~/miniforge3 if no conda/mamba is found
#   2. creates/updates the environment ~/envs/reeftrack-annotate (~1.7 GB)
#   3. puts a double-clickable "Annotate fish videos" launcher on the Desktop
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
ENV="$HOME/envs/reeftrack-annotate"
echo "reef-track annotator setup"
echo "  code:        $REPO"
echo "  environment: $ENV"
echo

# --- 1. a conda/mamba installation ------------------------------------------------------
CONDA=""
for c in mamba "$HOME/miniforge3/bin/mamba" conda "$HOME/miniforge3/bin/conda" /opt/anaconda3/bin/conda \
         "$HOME/anaconda3/bin/conda" "$HOME/miniconda3/bin/conda"; do
  if command -v "$c" >/dev/null 2>&1; then CONDA="$(command -v "$c")"; break; fi
done
if [[ -z "$CONDA" ]]; then
  echo "Installing Miniforge (one-off, ~100 MB download)..."
  ARCH="$(uname -m)"  # arm64 (Apple Silicon) or x86_64 (Intel)
  curl -fsSL -o /tmp/miniforge.sh \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-${ARCH}.sh"
  bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
  CONDA="$HOME/miniforge3/bin/mamba"
fi
echo "Using $CONDA"

# --- 2. the environment -------------------------------------------------------------------
if [[ -d "$ENV" ]]; then
  "$CONDA" env update --prefix "$ENV" -f "$REPO/environment-annotate.yml" --prune
else
  "$CONDA" env create --prefix "$ENV" -f "$REPO/environment-annotate.yml" -y
fi
"$ENV/bin/python" -m pip install -q -e "$REPO" --no-deps

# --- 3. Desktop launcher ------------------------------------------------------------------
LAUNCHER="$HOME/Desktop/Annotate fish videos.command"
cat > "$LAUNCHER" <<EOF
#!/bin/bash
# Opens the reef-track annotation tool. Keep this window open while you annotate.
export PATH="$ENV/bin:\$PATH"
cd "$REPO"
exec "$ENV/bin/reeftrack" annotate
EOF
chmod +x "$LAUNCHER"

echo
echo "Done. Double-click 'Annotate fish videos' on your Desktop to start."
echo "(You can close this window.)"
