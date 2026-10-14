#!/usr/bin/env bash
# Regenerate every chart from committed raw results (FINAL-COMPLETION.md W3).
# Charts with missing inputs are skipped with a message, never faked.
set -euo pipefail
cd "$(dirname "$0")/.."
RESULTS="${1:-results}"
OUT="${2:-docs/charts}"
if [ ! -d "$RESULTS" ]; then
  echo "No $RESULTS/ directory: run a sweep first (python -m bench.sweep --help)." >&2
  exit 1
fi
python -m bench.plot --results "$RESULTS" --out "$OUT"
