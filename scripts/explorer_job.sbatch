#!/bin/bash
#SBATCH --job-name=pagedserve-bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# Benchmark job for Northeastern Explorer. This is the ONLY environment whose
# numbers go in the README: it is the one place with a pinned, unshared A100
# (AGENTS.md section 4).
#
# Partition and constraint names vary by cluster and change over time. Confirm
# them with `sinfo -o "%P %G %N"` before the first submission rather than
# trusting the defaults above.
#
# Usage:
#   sbatch scripts/explorer_job.sbatch
#   MODEL=meta-llama/Llama-3.2-1B-Instruct CONCURRENCIES="1 8 32" \
#       sbatch scripts/explorer_job.sbatch
#
# Every run writes its own JSON under results/. Those files are the evidence
# for every number this project reports, so they are committed (AGENTS.md
# section 6). Nothing here prints a speedup: ratios are computed later, by
# bench/plot.py, from committed files.

set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
CONCURRENCIES="${CONCURRENCIES:-1 2 4 8 16 32 64}"
NUM_REQUESTS="${NUM_REQUESTS:-512}"
MAX_TOKENS="${MAX_TOKENS:-128}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-32}"
REPEATS="${REPEATS:-3}"
DATASET="${DATASET:-}"
RESULT_DIR="${RESULT_DIR:-results/$(date -u +%Y%m%dT%H%M%SZ)-${SLURM_JOB_ID:-local}}"

module load cuda/12.1 2>/dev/null || echo "note: no module system, using the ambient CUDA"

echo "==> environment"
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# Pin clocks. Thermal throttling between runs silently invents double-digit
# percentage differences, which is how a benchmark ends up measuring the
# cooling system. Needs privileges we may not have; failing is not fatal, but
# it MUST be recorded, because an unpinned run is weaker evidence.
CLOCKS_PINNED=false
if nvidia-smi -pm 1 >/dev/null 2>&1 && nvidia-smi -lgc "$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1)" >/dev/null 2>&1; then
    CLOCKS_PINNED=true
    echo "clocks pinned"
else
    echo "WARNING: could not pin clocks; run-to-run spread will be wider."
fi

echo "==> install"
python -m venv .venv-slurm 2>/dev/null || true
source .venv-slurm/bin/activate 2>/dev/null || true
pip install -q -e ".[engine,baseline,dev]"
python -c "import pagedserve; print('pagedserve', pagedserve.__version__)"

mkdir -p "${RESULT_DIR}"
echo "clocks_pinned=${CLOCKS_PINNED}" > "${RESULT_DIR}/run_notes.txt"
echo "model=${MODEL}" >> "${RESULT_DIR}/run_notes.txt"

DATASET_ARG=""
if [ -n "${DATASET}" ]; then
    DATASET_ARG="--dataset ${DATASET}"
fi

# Three repeats per point, because a single run is an anecdote. The median and
# the run-to-run spread are computed downstream from these files; this script
# only collects them.
echo "==> sweep: baselines"
for mode in sequential static; do
    for c in ${CONCURRENCIES}; do
        for run in $(seq 1 "${REPEATS}"); do
            out="${RESULT_DIR}/hf-${mode}-c${c}-run${run}.json"
            echo "--- hf/${mode} concurrency=${c} run=${run}"
            python bench/loadgen.py \
                --backend hf \
                --model "${MODEL}" \
                --baseline-mode "${mode}" \
                --mode closed \
                --concurrency "${c}" \
                --num-requests "${NUM_REQUESTS}" \
                --max-tokens "${MAX_TOKENS}" \
                --warmup-requests "${WARMUP_REQUESTS}" \
                --seed "${run}" \
                ${DATASET_ARG} \
                --output "${out}"
        done
    done
done

echo "==> done. results in ${RESULT_DIR}"
echo "Commit them: they are the denominator of every claim this project makes."
