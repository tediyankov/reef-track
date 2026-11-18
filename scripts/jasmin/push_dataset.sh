#!/bin/bash
# Copy a dataset folder (e.g. from RFS) to JASMIN, keeping its name:
#   bash scripts/jasmin/push_dataset.sh /Volumes/RFS/triggerfish_individual_id/Datasets/background_experiment
#   -> /gws/ssde/j25b/iecdt/tyankov/triggerfish-id/data/background_experiment/
# Safe to re-run: only new or changed files are sent. Nothing on JASMIN is deleted.
# Needs your JASMIN key loaded (ssh-add). Override the user / destination with env vars:
#   JASMIN_USER=tyankov JASMIN_DEST=/gws/.../data bash scripts/jasmin/push_dataset.sh <folder>
set -eo pipefail
SRC=${1:?usage: push_dataset.sh <dataset folder>}
SRC=${SRC%/}
USER_=${JASMIN_USER:-tyankov}
DEST=${JASMIN_DEST:-/gws/ssde/j25b/iecdt/tyankov/triggerfish-id/data}
[ -d "$SRC" ] || { echo "no such folder: $SRC"; exit 1; }
echo "copying $SRC -> $USER_@xfer-vm-01.jasmin.ac.uk:$DEST/$(basename "$SRC")/"
rsync -av --progress --exclude '.DS_Store' --exclude '._*' --exclude '*.partial' \
  "$SRC" "$USER_@xfer-vm-01.jasmin.ac.uk:$DEST/"
echo "done. On JASMIN: ls $DEST/$(basename "$SRC")"
