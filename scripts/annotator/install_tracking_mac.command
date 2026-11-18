#!/bin/bash
# One-time setup of reef-track WITH MASK TRACKING (SAM) on a Mac. Run it from Terminal:
#     bash <drag this file here>
# (macOS blocks double-clicking downloaded scripts; alternatively, after the warning use
#  System Settings -> Privacy & Security -> Open Anyway.)
#
# What it does (safe to run again, e.g. after downloading a newer version of reef-track):
#   1. installs Miniforge (a small Python distribution) in ~/miniforge3 if no conda/mamba is found
#   2. creates/updates the full environment ~/envs/reeftrack-full (~6 GB: PyTorch + SAM)
#   3. puts a double-clickable "Track fish videos" launcher on the Desktop: the same window as
#      "Annotate fish videos", plus mask tracking. Masks are saved per annotator in the video's
#      annotation folder (tracking_<name>/), next to behaviours_<name>.csv
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
ENV="$HOME/envs/reeftrack-full"
echo "reef-track setup (annotation + mask tracking)"
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
  "$CONDA" env update --prefix "$ENV" -f "$REPO/environment.yml" --prune
else
  "$CONDA" env create --prefix "$ENV" -f "$REPO/environment.yml" -y
fi
"$ENV/bin/python" -m pip install -q -e "$REPO" --no-deps

# --- 3. Desktop launcher ------------------------------------------------------------------
LAUNCHER="$HOME/Desktop/Track fish videos.command"
cat > "$LAUNCHER" <<EOF
#!/bin/bash
# Opens the reef-track annotation tool with mask tracking. Keep this window open while you work.
export PATH="$ENV/bin:\$PATH"
cd "$REPO"
exec "$ENV/bin/reeftrack" annotate
EOF
chmod +x "$LAUNCHER"

echo
echo "Done. Double-click 'Track fish videos' on your Desktop to start."
echo "(The first time you track a fish, the tracking model downloads once: 0.2-3.4 GB.)"
echo "(You can close this window.)"
