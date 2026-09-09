#!/bin/bash
#SBATCH --job-name=pagedserve-bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
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
#   SECTIONS="baselines arms" sbatch scripts/explorer_job.sbatch
#   MODEL=Qwen/Qwen2.5-0.5B-Instruct CONCURRENCIES="1 8 32" \
#       sbatch scripts/explorer_job.sbatch
#
#   Run only the HF baselines:
#   SECTIONS=baselines sbatch scripts/explorer_job.sbatch
#
#   Run only the pagedserve ablation arms:
#   SECTIONS="arms" sbatch ...
#
# Every run writes its own JSON under results/. Those files are the evidence
# for every number this project reports, so they are committed (AGENTS.md
# section 6). Nothing here prints a speedup: ratios are computed later, by
# bench/plot.py, from committed files.
#
# ---------------------------------------------------------------------------
# Rehearsing it off the GPU
#
# An allocation is the worst place to discover an unbound variable. Two escape
# hatches exist so the whole control flow can be exercised on a laptop:
#
#   DRY_RUN=1 scripts/explorer_job.sbatch
#       Prints every command it would run, expands the full
#       SECTIONS x CONCURRENCIES x REPEATS matrix, and touches nothing --
#       no install, no download, no result files.
#
#   BACKEND=mock SECTIONS=baselines CONCURRENCIES="1 2" NUM_REQUESTS=4 \
#       REPEATS=1 DATASET=/path/to/tiny.json RESULT_DIR=/tmp/rehearsal \
#       SKIP_INSTALL=1 scripts/explorer_job.sbatch
#       Actually runs, on CPU, in seconds. Proves the JSON shape.
#
# CPU timings are NOT evidence and must never be committed to results/. Point
# RESULT_DIR outside the repo for any rehearsal.
# ---------------------------------------------------------------------------

set -euo pipefail

# Relative paths below (scripts/fetch_dataset.sh, results/) are repo-root
# relative. SLURM starts the job in the submitting directory, which is usually
# but not always the repo root, and a rehearsal can be launched from anywhere.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DRY_RUN="${DRY_RUN:-0}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"

# Echo instead of execute under DRY_RUN. %q quoting means the printed line is
# one you can paste back into a shell -- an approximation of the command is not
# much use for finding a quoting bug.
run_cmd() {
    if [ "${DRY_RUN}" = "1" ]; then
        printf 'DRY_RUN:'
        printf ' %q' "$@"
        printf '\n'
        return 0
    fi
    "$@"
}

# Qwen2.5-0.5B-Instruct is ungated, so no HF token is needed in any of the three
# environments (AGENTS.md section 7). A gated default would fail at download on
# a compute node with no interactive way to authenticate -- at which point the
# allocation is already spent.
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
CONCURRENCIES="${CONCURRENCIES:-1 2 4 8 16 32 64}"
RATES="${RATES:-2 4 8 16}"
NUM_REQUESTS="${NUM_REQUESTS:-256}"
MAX_TOKENS="${MAX_TOKENS:-128}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-32}"
REPEATS="${REPEATS:-3}"
# Which blocks of work to run. Trim this when the allocation is short: the
# concurrency arms are the headline, everything else is supporting evidence.
SECTIONS="${SECTIONS:-baselines arms prefix poisson}"
PREFIX_WORDS="${PREFIX_WORDS:-400}"
PREFIX_FRACTIONS="${PREFIX_FRACTIONS:-0.0 0.25 0.5 0.75 0.9}"
# Left unset by default so the engine profiles the card. Set it for any run that
# must not OOM: profile_num_blocks currently measures activation memory as zero,
# so its estimate is optimistic (AGENTS.md section 7, known gap).
NUM_BLOCKS="${NUM_BLOCKS:-}"
RESULT_DIR="${RESULT_DIR:-results/$(date -u +%Y%m%dT%H%M%SZ)-${SLURM_JOB_ID:-local}}"
# The baseline arm's backend. Only ever overridden to `mock` for a rehearsal:
# the mock needs no model, no tokenizer and no GPU, so it exercises the
# generator, the metrics and the JSON writer and nothing else.
BACKEND="${BACKEND:-hf}"

module load cuda/12.1 2>/dev/null || echo "note: no module system, using the ambient CUDA"

echo "==> environment"
# Guarded rather than assumed: the script must survive a machine with no
# nvidia-smi, or `set -e` turns a rehearsal into an exit before the first line
# of real work. On the cluster this branch is never taken, and if it ever were,
# a run with no recorded hardware is a run whose numbers are void (section 6).
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    GPU_PRESENT=true
else
    echo "WARNING: no nvidia-smi on this machine. Nothing produced here is evidence."
    GPU_PRESENT=false
fi

# Pin clocks. Thermal throttling between runs silently invents double-digit
# percentage differences, which is how a benchmark ends up measuring the
# cooling system. Needs privileges we may not have; failing is not fatal, but
# it MUST be recorded, because an unpinned run is weaker evidence.
CLOCKS_PINNED=false
if [ "${GPU_PRESENT}" = "true" ] &&
    nvidia-smi -pm 1 >/dev/null 2>&1 &&
    nvidia-smi -lgc "$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1)" >/dev/null 2>&1; then
    CLOCKS_PINNED=true
    echo "clocks pinned"
else
    echo "WARNING: could not pin clocks; run-to-run spread will be wider."
fi

echo "==> install"
if [ "${DRY_RUN}" = "1" ] || [ "${SKIP_INSTALL}" = "1" ]; then
    echo "skipping install (DRY_RUN=${DRY_RUN} SKIP_INSTALL=${SKIP_INSTALL})"
elif [ -n "${VIRTUAL_ENV:-}" ]; then
    # Already inside an environment someone chose deliberately. Building a
    # second one over the top of it installs into the wrong interpreter.
    echo "using the active virtualenv: ${VIRTUAL_ENV}"
    pip install -q -e ".[engine,baseline,dev]"
else
    python -m venv .venv-slurm 2>/dev/null || true
    # shellcheck source=/dev/null
    source .venv-slurm/bin/activate 2>/dev/null || true
    pip install -q -e ".[engine,baseline,dev]"
fi
run_cmd python -c "import pagedserve; print('pagedserve', pagedserve.__version__)"

echo "==> dataset"
DATASET="${DATASET:-}"
if [ -z "${DATASET}" ]; then
    run_cmd scripts/fetch_dataset.sh
    DATASET="${DATA_DIR:-data}/sharegpt_v3.json"
fi
echo "using dataset: ${DATASET}"

run_cmd mkdir -p "${RESULT_DIR}"
[ "${DRY_RUN}" = "1" ] || {
    echo "clocks_pinned=${CLOCKS_PINNED}"
    echo "gpu_present=${GPU_PRESENT}"
    echo "backend=${BACKEND}"
    echo "model=${MODEL}"
    echo "dataset=${DATASET}"
    echo "sections=${SECTIONS}"
    echo "num_blocks=${NUM_BLOCKS:-profiled}"
} > "${RESULT_DIR}/run_notes.txt"

# `python -m` rather than `python bench/loadgen.py`: the script form only
# resolves `bench.metrics` if the repo root happens to be on sys.path, which
# depends on how the editable install landed. The module form works either way,
# and an ImportError at hour zero of an allocation is an expensive way to learn
# the difference.
LOADGEN=(python -m bench.loadgen)

COMMON=(--model "${MODEL}" --dataset "${DATASET}"
        --num-requests "${NUM_REQUESTS}" --max-tokens "${MAX_TOKENS}"
        --warmup-requests "${WARMUP_REQUESTS}")
if [ -n "${NUM_BLOCKS}" ]; then
    COMMON+=(--num-blocks "${NUM_BLOCKS}")
fi
# Unset on the cluster, where the engine resolves CUDA on its own. Set it for a
# laptop rehearsal: a Mac otherwise resolves to MPS, and profile_num_blocks
# refuses to size a cache on any device that cannot report free memory.
if [ -n "${DEVICE:-}" ]; then
    COMMON+=(--device "${DEVICE}")
fi

has_section() { [[ " ${SECTIONS} " == *" $1 "* ]]; }

# Three repeats per point, because a single run is an anecdote. The median and
# the run-to-run spread are computed downstream from these files; this script
# only collects them.

if has_section baselines; then
    echo "==> baselines"
    # Sequential runs one request at a time by construction, so sweeping
    # concurrency against it would buy seven identical results at seven times
    # the cost. It is the floor, measured once.
    for run in $(seq 1 "${REPEATS}"); do
        echo "--- hf/sequential run=${run}"
        run_cmd "${LOADGEN[@]}" --backend "${BACKEND}" --baseline-mode sequential \
            --mode closed --concurrency 1 --seed "${run}" "${COMMON[@]}" \
            --output "${RESULT_DIR}/hf-sequential-c1-run${run}.json"
    done

    # Static batching is the fair floor and the headline comparison.
    for c in ${CONCURRENCIES}; do
        for run in $(seq 1 "${REPEATS}"); do
            echo "--- hf/static concurrency=${c} run=${run}"
            run_cmd "${LOADGEN[@]}" --backend "${BACKEND}" --baseline-mode static \
                --mode closed --concurrency "${c}" --seed "${run}" "${COMMON[@]}" \
                --output "${RESULT_DIR}/hf-static-c${c}-run${run}.json"
        done
    done
fi

if has_section arms; then
    # The three ablation arms. Each adds exactly one variable to the one above
    # it, which is what turns "we are faster than HuggingFace" into a claim
    # about *which* mechanism produced the gain.
    #
    #   contiguous  --no-paging       the Phase 1 naive floor, our own code
    #   paged       --static-batching paging alone, no scheduler
    #   continuous  (default)         paging plus iteration-level scheduling
    echo "==> pagedserve arms"
    for arm in contiguous paged continuous; do
        case "${arm}" in
            contiguous) ARM_FLAGS=(--no-paging) ;;
            paged)      ARM_FLAGS=(--static-batching) ;;
            # Empty by definition -- continuous batching is the default, so the
            # arm that exercises it passes no flag at all. Expanded below as
            # ${ARM_FLAGS[@]+...}: under `set -u`, bash before 4.4 calls an
            # empty array unbound and kills the job. macOS ships bash 3.2, so
            # rehearsing this locally is the only way to see it.
            continuous) ARM_FLAGS=() ;;
        esac
        for c in ${CONCURRENCIES}; do
            for run in $(seq 1 "${REPEATS}"); do
                echo "--- pagedserve/${arm} concurrency=${c} run=${run}"
                run_cmd "${LOADGEN[@]}" --backend pagedserve ${ARM_FLAGS[@]+"${ARM_FLAGS[@]}"} \
                    --mode closed --concurrency "${c}" --seed "${run}" "${COMMON[@]}" \
                    --output "${RESULT_DIR}/ps-${arm}-c${c}-run${run}.json" \
                    || echo "FAILED: ps-${arm}-c${c}-run${run} (continuing)"
            done
        done
    done
fi

if has_section prefix; then
    # Phase 5's claim is a *paired* number: same workload, same seed, cache on
    # versus off. ShareGPT conversations share no opening, so the shared prefix
    # has to be injected or there is nothing to hit on -- and the honest way to
    # report it is as a curve against the shared fraction, not a single number.
    echo "==> prefix caching"
    for frac in ${PREFIX_FRACTIONS}; do
        for cache in off on; do
            CACHE_FLAGS=()
            [ "${cache}" = "on" ] && CACHE_FLAGS=(--enable-prefix-caching)
            for run in $(seq 1 "${REPEATS}"); do
                echo "--- prefix frac=${frac} cache=${cache} run=${run}"
                # Poisson: reuse needs staggered arrivals. Requests admitted in
                # the same step have no computed KV to share, so a closed-loop
                # burst reports a 0% hit rate and that is correct behaviour, not
                # a broken cache.
                run_cmd "${LOADGEN[@]}" --backend pagedserve ${CACHE_FLAGS[@]+"${CACHE_FLAGS[@]}"} \
                    --mode poisson --rate 8 --seed "${run}" \
                    --shared-prefix-words "${PREFIX_WORDS}" \
                    --shared-prefix-fraction "${frac}" \
                    "${COMMON[@]}" \
                    --output "${RESULT_DIR}/prefix-f${frac}-${cache}-run${run}.json" \
                    || echo "FAILED: prefix-f${frac}-${cache}-run${run} (continuing)"
            done
        done
    done
fi

if has_section poisson; then
    # Open-loop headline. Closed-loop bursts flatter the system: a saturated
    # server slows the arrival rate down, so offered load silently drops to
    # whatever the server can handle. Real traffic arrives whether or not you
    # are keeping up, which is why queueing delay belongs in the number.
    echo "==> poisson (open loop)"
    for rate in ${RATES}; do
        for run in $(seq 1 "${REPEATS}"); do
            echo "--- poisson rate=${rate} run=${run}"
            run_cmd "${LOADGEN[@]}" --backend pagedserve \
                --mode poisson --rate "${rate}" --seed "${run}" "${COMMON[@]}" \
                --output "${RESULT_DIR}/ps-poisson-r${rate}-run${run}.json" \
                || echo "FAILED: ps-poisson-r${rate}-run${run} (continuing)"
        done
    done
fi

if [ "${DRY_RUN}" = "1" ]; then
    echo "==> dry run complete. Nothing was executed and no files were written."
    echo "Count the DRY_RUN: lines above against SECTIONS x CONCURRENCIES x REPEATS."
    exit 0
fi

WRITTEN=$(find "${RESULT_DIR}" -name '*.json' | wc -l | tr -d ' ')
echo "==> done. ${WRITTEN} result files in ${RESULT_DIR}"
if [ "${GPU_PRESENT}" != "true" ]; then
    # A rehearsal must not be mistaken later for the run that closed a phase.
    echo "WARNING: this ran without a GPU. These files are NOT evidence (AGENTS.md"
    echo "         section 4) and must not be committed to results/."
    exit 0
fi
echo "Commit them: they are the denominator of every claim this project makes."
