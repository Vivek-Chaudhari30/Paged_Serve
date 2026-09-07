#!/bin/bash
# Download the ShareGPT v3 dataset used by the benchmark sweep.
# Idempotent: exits immediately if the file is already present, so re-running
# this script never re-downloads 673 MB.
#
# Usage:
#   bash scripts/fetch_dataset.sh                    # writes data/sharegpt_v3.json
#   bash scripts/fetch_dataset.sh /path/to/dest.json # custom path

set -euo pipefail

DEST="${1:-data/sharegpt_v3.json}"

if [ -f "${DEST}" ]; then
    echo "already present: ${DEST} ($(du -sh "${DEST}" | cut -f1))"
    exit 0
fi

mkdir -p "$(dirname "${DEST}")"

# ShareGPT v3 cleaned — same copy used by the vLLM and SGLang benchmarks.
URL="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"

echo "downloading ShareGPT v3 -> ${DEST}"
if command -v curl >/dev/null 2>&1; then
    curl -L --progress-bar "${URL}" -o "${DEST}"
elif command -v wget >/dev/null 2>&1; then
    wget -q --show-progress -O "${DEST}" "${URL}"
else
    echo "error: neither curl nor wget is available" >&2
    exit 1
fi

echo "done: ${DEST} ($(du -sh "${DEST}" | cut -f1))"
