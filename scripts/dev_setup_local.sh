#!/usr/bin/env bash
# Installs TabPFN, TabDPT, carte-ai and phmd as editable packages from local
# submodule checkouts, overriding the git-installed versions.
#
# Run after `uv sync` whenever you need local editable installs:
#   uv sync
#   bash scripts/dev_setup_local.sh
#
# Note: running `uv sync` again will reinstall from git, requiring another
# run of this script to restore editable installs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

for pkg in TabPFN TabDPT carte-ai phmd; do
    path="$PROJECT_ROOT/local_packages/$pkg"
    if [ ! -d "$path" ]; then
        echo "SKIP  $pkg  (not found at $path)"
        continue
    fi
    echo "Installing $pkg from $path"
    uv pip install -e "$path"
done

echo ""
echo "Done. Local editable installs are active."
echo "Re-run this script after any 'uv sync' to restore them."
