#!/usr/bin/env bash
# Fetch the benchmark workload.
#
# ShareGPT V3 (unfiltered, cleaned split) is the standard prompt set for serving
# benchmarks, and it is what bench/loadgen.py's loader was written against: a
# JSON list of conversations, each with a `conversations` array of
# {"from": ..., "value": ...} turns. The loader takes the first human turn.
#
# Why this dataset and not a synthetic one: its length distribution is
# heavy-tailed, and variance is precisely what a paged allocator exploits. A
# uniform-length workload would understate the result this project exists to
# measure. bench/loadgen.py's built-in synthetic prompts remain the *controlled*
# arm for isolating single variables -- they are not the headline.
#
# The file is ~642 MiB and lands in data/, which is gitignored: ShareGPT is
# scraped conversation data of uncertain provenance, so it is cited in the
# methodology rather than redistributed from this repo.
#
# Usage:
#   scripts/fetch_dataset.sh            # -> data/sharegpt_v3.json
#   DATA_DIR=/scratch/$USER scripts/fetch_dataset.sh

set -euo pipefail

DATA_DIR="${DATA_DIR:-data}"
DEST="${DATA_DIR}/sharegpt_v3.json"
URL="${SHAREGPT_URL:-https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json}"

if [ -f "${DEST}" ]; then
    echo "already present: ${DEST} ($(du -h "${DEST}" | cut -f1))"
else
    echo "==> downloading ShareGPT V3 to ${DEST}"
    mkdir -p "${DATA_DIR}"
    # --create-dirs and a temp name so an interrupted download never leaves a
    # truncated file that later parses as valid JSON right up to the cut.
    curl -L --fail --create-dirs -o "${DEST}.partial" "${URL}"
    mv "${DEST}.partial" "${DEST}"
fi

# Parse it here rather than discovering a bad file mid-sweep, when the GPU is
# already on the clock.
python - "${DEST}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
entries = json.loads(path.read_text())
usable = sum(
    1
    for e in entries
    if any(t.get("from") == "human" and t.get("value") for t in e.get("conversations") or [])
)
print(f"parsed {len(entries)} conversations, {usable} with a usable human turn")
if usable == 0:
    raise SystemExit("no usable prompts: the loader would raise on this file")
PY

echo "==> dataset ready: ${DEST}"
