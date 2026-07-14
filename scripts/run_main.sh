#!/usr/bin/env bash
# Launches main.py through IsaacLab's own launcher, with src/ on PYTHONPATH so its
# absolute imports (isaacsim_interface.*, rdt_inference.*) resolve without needing
# `pip install -e .`.
#
# Run this standalone, in the `isaaclab` conda env:
#
#   conda activate isaaclab
#   ./scripts/run_main.sh
#
# ISAACLAB_PATH must point at your local IsaacLab checkout (the directory containing
# isaaclab.sh); override it if yours isn't at the default below.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ISAACLAB_PATH="${ISAACLAB_PATH:-$HOME/lab/IsaacLab}"

if [ ! -x "$ISAACLAB_PATH/isaaclab.sh" ]; then
    echo "error: isaaclab.sh not found at $ISAACLAB_PATH/isaaclab.sh" >&2
    echo "       set ISAACLAB_PATH to your IsaacLab checkout, e.g.:" >&2
    echo "       ISAACLAB_PATH=/path/to/IsaacLab ./scripts/run_main.sh" >&2
    exit 1
fi

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

exec "$ISAACLAB_PATH/isaaclab.sh" -p "$REPO_ROOT/src/isaacsim_inference/main.py" "$@"
