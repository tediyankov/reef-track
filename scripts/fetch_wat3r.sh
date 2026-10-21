#!/usr/bin/env bash
# Vendor the Wat3R model code (Apache-2.0) at a pinned commit into third_party/Wat3R.
# Weights (~4 GB) are downloaded separately by Hugging Face on first use into ~/.cache/huggingface
# (set HF_HOME to put them elsewhere, e.g. a JASMIN group workspace).
set -euo pipefail

REPO_URL="https://huggingface.co/spaces/lsxi77777/Wat3R"
COMMIT="1b285b084f946c4cd849fe8004c25c44453f442a"
DEST="$(cd "$(dirname "$0")/.." && pwd)/third_party/Wat3R"

if [[ -d "$DEST/.git" ]]; then
  echo "Wat3R already present at $DEST"
else
  GIT_LFS_SKIP_SMUDGE=1 git clone --quiet "$REPO_URL" "$DEST"
fi
git -C "$DEST" -c advice.detachedHead=false checkout --quiet "$COMMIT"
echo "Wat3R code @ ${COMMIT:0:8} -> $DEST"
