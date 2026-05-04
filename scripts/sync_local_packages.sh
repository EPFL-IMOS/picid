#!/usr/bin/env bash
# Sync the default environment, then overlay editable local package checkouts.
set -euo pipefail

cd "$(dirname "$0")/.."

missing=0
for path in \
    local_packages/TabPFN \
    local_packages/TabDPT \
    local_packages/carte-ai \
    local_packages/phmd
do
    if [ ! -d "$path" ]; then
        echo "Missing $path"
        missing=1
    fi
done

if [ "$missing" -ne 0 ]; then
    echo "Create or clone the missing checkouts under ./local_packages, then rerun this script."
    exit 1
fi

uv sync
uv pip install -r requirements-local.txt
