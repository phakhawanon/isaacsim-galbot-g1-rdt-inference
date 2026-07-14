#!/usr/bin/env bash
# Launches rdt_server.py as a module (required for its relative `from .rdt_ipc import`
# to resolve) without needing `pip install -e .` -- just puts src/ on PYTHONPATH.
#
# Run this standalone, in the `rdt` conda env -- NOT through isaaclab.sh, and NOT in
# the `isaaclab` conda env:
#
#   conda activate rdt
#   ./scripts/start_rdt_server.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

exec python -m rdt_inference.rdt_server "$@"
