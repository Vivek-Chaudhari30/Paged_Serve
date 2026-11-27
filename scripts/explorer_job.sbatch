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
#   On a plain SSH GPU box with no Slurm, run it with bash. The #SBATCH lines
#   are comments there; CUDA_VISIBLE_DEVICES picks the card (default 0). See
#   docs/lab-runbook.md:
#   SKIP_INSTALL=1 bash scripts/explorer_job.sbatch 2>&1 | tee run.log
#
# Before any benchmark, the job builds the CUDA extension and runs the
# correctness gate (pytest -m gpu, then the golden suite on cuda), and aborts if
# either fails. SKIP_GATE=1 skips both, for a caller that already ran them.
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
#
# Three more sections are opt-in, because together they roughly double the run
# and an AWS box that hits its dead-man's switch mid-sweep loses everything:
#   microbench  bench/kernel_bench.py: the decode kernel alone, us and GB/s
#   ablations   bench/sweep.py for ${ABLATION_SWEEPS} (block size, preemption
#               by prompt length, kernel arms, chunked prefill)
#   http        PagedServe's server, then vLLM if available, driven over HTTP
#               by the same client (bench/http_backend.py)
# Run them as a second session: SECTIONS="microbench ablations http".
SECTIONS="${SECTIONS:-baselines arms prefix poisson}"
ABLATION_SWEEPS="${ABLATION_SWEEPS:-block-size preemption kernel chunked-prefill}"
# The http section's servers. Both get the same dtype, sequence limit, batch
# limit, block size and -- when NUM_BLOCKS is set -- the same KV capacity, so
# the comparison is between the systems rather than their defaults.
HTTP_PORT="${HTTP_PORT:-8011}"
HTTP_DTYPE="${HTTP_DTYPE:-bfloat16}"
HTTP_MAX_SEQ_LEN="${HTTP_MAX_SEQ_LEN:-2048}"
HTTP_MAX_NUM_SEQS="${HTTP_MAX_NUM_SEQS:-32}"
# vLLM lives in its own virtualenv: it pins its own torch, and installing it
# into this one would replace the torch the CUDA extension was built against.
# Point VLLM_PYTHON at an existing one, or set INSTALL_VLLM=1 to build
# .venv-vllm here (VLLM_VERSION pins it; record it either way). Unset both and
# the vLLM half is skipped with a message.
VLLM_PYTHON="${VLLM_PYTHON:-}"
INSTALL_VLLM="${INSTALL_VLLM:-0}"
VLLM_VERSION="${VLLM_VERSION:-}"
# PagedServe's prefix cache is off unless asked for, so vLLM's is turned off to
# match. Flag spellings move between vLLM releases: if the server refuses to
# start, adjust this rather than the comparison.
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---no-enable-prefix-caching}"
PREFIX_WORDS="${PREFIX_WORDS:-400}"
PREFIX_FRACTIONS="${PREFIX_FRACTIONS:-0.0 0.25 0.5 0.75 0.9}"
# Fixed by default. profile_num_blocks measures activation memory as zero, so
# its estimate is optimistic (AGENTS.md section 7, known gap), and on a shared
# machine the free memory it sizes against can shrink mid-run. 8192 blocks of 16
# tokens is ~1.6 GB of KV for the 0.5B model and ~16 GB for an 8B one, and it
# gives vLLM the same capacity in the http section. NUM_BLOCKS= (empty) profiles.
NUM_BLOCKS="${NUM_BLOCKS-8192}"
SKIP_GATE="${SKIP_GATE:-0}"
# A GPU that already has work on it measures the other job as much as this
# one. Refused unless this is set, and recorded either way.
ALLOW_SHARED_GPU="${ALLOW_SHARED_GPU:-0}"
# Slurm sets this to the allocated card. On a plain SSH box an unset value means
# every GPU, so one is chosen explicitly: a run must name the card it measured.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
GPU_INDEX="${CUDA_VISIBLE_DEVICES%%,*}"
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
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Someone else's job on this card turns every number into a measurement of
# contention. Lab machines are shared, so check before spending hours. Other
# users' processes are not always listed, so memory already in use counts too
# (a display server holds a few hundred MB, hence the threshold).
GPU_SHARED=false
if [ "${GPU_PRESENT}" = "true" ]; then
    OTHER_APPS=$(nvidia-smi -i "${GPU_INDEX}" --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader 2>/dev/null || true)
    USED_MIB=$(nvidia-smi -i "${GPU_INDEX}" --query-gpu=memory.used --format=csv,noheader,nounits \
        2>/dev/null | head -1 | tr -d ' ')
    case "${USED_MIB}" in '' | *[!0-9]*) USED_MIB=0 ;; esac  # "[N/A]" on some cards
    if [ -n "${OTHER_APPS}" ] || [ "${USED_MIB:-0}" -gt 2048 ]; then
        GPU_SHARED=true
        echo "GPU ${GPU_INDEX} is already in use (${USED_MIB:-?} MiB used):"
        echo "${OTHER_APPS:-  (processes not visible to this user)}"
        if [ "${ALLOW_SHARED_GPU}" != "1" ] && [ "${DRY_RUN}" != "1" ]; then
            echo "ABORT: pick a free card with CUDA_VISIBLE_DEVICES, or set ALLOW_SHARED_GPU=1" >&2
            echo "       for a run that is not evidence." >&2
            exit 1
        fi
        echo "WARNING: continuing on a shared GPU (ALLOW_SHARED_GPU=1); these numbers are not evidence."
    fi
fi

# Pin clocks. Thermal throttling between runs silently invents double-digit
# percentage differences, which is how a benchmark ends up measuring the
# cooling system. Needs privileges we may not have; failing is not fatal, but
# it MUST be recorded, because an unpinned run is weaker evidence.
CLOCKS_PINNED=false
if [ "${GPU_PRESENT}" = "true" ] &&
    nvidia-smi -i "${GPU_INDEX}" -pm 1 >/dev/null 2>&1 &&
    nvidia-smi -i "${GPU_INDEX}" -lgc "$(nvidia-smi -i "${GPU_INDEX}" --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1)" >/dev/null 2>&1; then
    CLOCKS_PINNED=true
    echo "clocks pinned"
else
    echo "WARNING: could not pin clocks; run-to-run spread will be wider."
fi

echo "==> install"
# pyproject requires 3.11. A cluster's or a conda base's default python is often
# older, and pip's refusal to install comes after the queue wait, not before it.
if ! python -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
    echo "ABORT: $(python --version 2>&1) is too old; pagedserve needs Python 3.11+." >&2
    echo "       e.g. conda create -n pagedserve python=3.11 && conda activate pagedserve" >&2
    exit 1
fi
# An active virtualenv or conda env was chosen deliberately: install into it.
# Building a fresh venv on top would install a second torch, and the CUDA
# extension built against the first one would then fail to load in the second.
ACTIVE_ENV="${VIRTUAL_ENV:-${CONDA_PREFIX:-}}"
if [ "${DRY_RUN}" = "1" ] || [ "${SKIP_INSTALL}" = "1" ]; then
    echo "skipping install (DRY_RUN=${DRY_RUN} SKIP_INSTALL=${SKIP_INSTALL})"
elif [ -n "${ACTIVE_ENV}" ]; then
    echo "using the active environment: ${ACTIVE_ENV}"
    if [ "${CONDA_DEFAULT_ENV:-}" = "base" ] && [ -z "${VIRTUAL_ENV:-}" ]; then
        echo "WARNING: installing into conda base; a dedicated env is safer."
    fi
    python -m pip install -q -e ".[engine,baseline,dev]"
else
    python -m venv .venv-slurm 2>/dev/null || true
    # shellcheck source=/dev/null
    source .venv-slurm/bin/activate 2>/dev/null || true
    python -m pip install -q -e ".[engine,baseline,dev]"
fi
run_cmd python -c "import pagedserve; print('pagedserve', pagedserve.__version__)"

echo "==> CUDA extension and correctness gate"
# Measuring a broken engine produces numbers that are confident and wrong, which
# is worse than no numbers. Same two-part gate as scripts/aws_bench.sh: -m gpu
# alone never runs the golden suite, which is device-parametric (AGENTS.md 2.2).
if [ "${SKIP_GATE}" = "1" ]; then
    echo "skipping build and gate (SKIP_GATE=1): the caller has already run them"
elif [ "${GPU_PRESENT}" != "true" ] && [ "${DRY_RUN}" != "1" ]; then
    echo "no GPU: skipping the CUDA build and the gate"
else
    # No arch flag: torch targets the visible card (sm_80 A100, sm_86 A6000).
    run_cmd python setup.py build_ext --inplace ||
        { echo "ABORT: the CUDA extension did not build. Report the full error above." >&2; exit 1; }
    # build_ext skips the extension, without failing, when it finds no nvcc.
    run_cmd python -c "import sys; from pagedserve import extension; ext = extension.load(); \
print('extension loaded' if ext else f'extension MISSING: {extension.unavailable_reason()}'); \
sys.exit(ext is None)" ||
        { echo "ABORT: no CUDA extension (is nvcc on PATH? see above)." >&2; exit 1; }
    run_cmd python -m pytest -m gpu -q ||
        { echo "ABORT: pytest -m gpu failed. Not benchmarking a failing engine." >&2; exit 1; }
    run_cmd env PAGEDSERVE_TEST_DEVICE=cuda PAGEDSERVE_TEST_DTYPE=float16 python -m pytest -q ||
        { echo "ABORT: the golden gate failed on cuda. That is a bug report (AGENTS.md 2.2)." >&2; exit 1; }
fi

echo "==> dataset"
DATASET="${DATASET:-}"
if [ -z "${DATASET}" ]; then
    run_cmd bash scripts/fetch_dataset.sh
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
    echo "host=$(hostname)"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
    echo "gpu_shared_at_start=${GPU_SHARED}"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "git_dirty=$([ -n "$(git status --porcelain 2>/dev/null)" ] && echo true || echo false)"
} > "${RESULT_DIR}/run_notes.txt"

# Background processes this job owns: a sampler of the card, and the http
# section's server. Killed on every exit path, so an abort or a Ctrl-C cannot
# leave a server holding the port and the GPU for the next person.
GPU_MONITOR_PID=""
HTTP_PID=""
cleanup() {
    # `|| true`: under set -e a failed kill (the process already exited) as the
    # last command of an && list would turn a finished job into exit status 1.
    if [ -n "${HTTP_PID}" ]; then kill "${HTTP_PID}" 2>/dev/null || true; fi
    if [ -n "${GPU_MONITOR_PID}" ]; then kill "${GPU_MONITOR_PID}" 2>/dev/null || true; fi
}
trap cleanup EXIT
# One sample every 30 s for the whole run. If another job lands on the card
# mid-sweep, memory.used and utilization jump outside our runs, and this file is
# the only way to find out afterwards which results it contaminated.
if [ "${GPU_PRESENT}" = "true" ] && [ "${DRY_RUN}" != "1" ]; then
    nvidia-smi -i "${GPU_INDEX}" \
        --query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,clocks.sm \
        --format=csv -l 30 > "${RESULT_DIR}/gpu_monitor.csv" 2>&1 &
    GPU_MONITOR_PID=$!
fi

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

if has_section microbench; then
    # The decode kernel in isolation: no MLP, no sampler, no scheduler. The
    # effective bandwidth it reports, against the card's peak, is what the K6
    # optimisations are measured against.
    echo "==> kernel microbenchmark"
    run_cmd python -m bench.kernel_bench --model "${MODEL}" --dtype float16 \
        --output "${RESULT_DIR}/kernel/microbench.json" \
        || echo "FAILED: kernel microbenchmark (continuing)"
fi

if has_section ablations; then
    # bench/sweep.py owns these grids; this section only hands it the shared
    # workload. Each sweep writes into its own directory and tags every run
    # with its name, so bench/plot.py keeps them out of the concurrency charts.
    SWEEP_COMMON=(--model "${MODEL}" --dataset "${DATASET}"
                  --num-requests "${NUM_REQUESTS}" --max-tokens "${MAX_TOKENS}"
                  --warmup-requests "${WARMUP_REQUESTS}" --repeats "${REPEATS}")
    if [ -n "${NUM_BLOCKS}" ]; then
        SWEEP_COMMON+=(--num-blocks "${NUM_BLOCKS}")
    fi
    if [ -n "${DEVICE:-}" ]; then
        SWEEP_COMMON+=(--extra "--device ${DEVICE}")
    fi
    for sweep in ${ABLATION_SWEEPS}; do
        echo "==> ablation sweep: ${sweep}"
        # shellcheck disable=SC2086
        run_cmd python -m bench.sweep "${sweep}" "${SWEEP_COMMON[@]}" \
            --concurrency ${CONCURRENCIES} --out "${RESULT_DIR}/${sweep}" \
            || echo "FAILED: some ${sweep} runs (continuing; failed runs leave no file)"
    done
fi

# Start a server in the background, wait for /health, run the http sweep
# against it, stop it. A server that never comes up is a failed section, not a
# failed job: the other sections' results are still worth keeping.
port_in_use() {
    python -c 'import socket, sys
s = socket.socket()
s.settimeout(1)
sys.exit(0 if s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)' "$1"
}

http_sweep() {
    local arm="$1" log="$2"
    shift 2
    # On a shared machine the port may already belong to someone else's
    # server. Its /health would answer the readiness check below, and the sweep
    # would then benchmark their server under our arm's name.
    if [ "${DRY_RUN}" != "1" ] && port_in_use "${HTTP_PORT}"; then
        echo "FAILED: ${arm}: port ${HTTP_PORT} is already in use by another process; set HTTP_PORT"
        return 0
    fi
    if [ "${DRY_RUN}" = "1" ]; then
        run_cmd "$@"
    else
        "$@" > "${log}" 2>&1 &
        HTTP_PID=$!
    fi
    if [ "${DRY_RUN}" != "1" ]; then
        local up=false
        for _ in $(seq 1 180); do
            # Ours must still be alive: a server that died on startup cannot
            # be the one answering.
            kill -0 "${HTTP_PID}" 2>/dev/null || break
            if curl -sf "http://127.0.0.1:${HTTP_PORT}/health" >/dev/null 2>&1; then
                up=true
                break
            fi
            sleep 5
        done
        if [ "${up}" != "true" ]; then
            echo "FAILED: ${arm} server did not come up; see ${log}"
            kill "${HTTP_PID}" 2>/dev/null || true
            wait "${HTTP_PID}" 2>/dev/null || true
            HTTP_PID=""
            return 0
        fi
    fi
    # shellcheck disable=SC2086
    run_cmd python -m bench.sweep http --model "${MODEL}" --dataset "${DATASET}" \
        --num-requests "${NUM_REQUESTS}" --max-tokens "${MAX_TOKENS}" \
        --warmup-requests "${WARMUP_REQUESTS}" --repeats "${REPEATS}" \
        --concurrency ${CONCURRENCIES} --base-url "http://127.0.0.1:${HTTP_PORT}/v1" \
        --http-arm "${arm}" --out "${RESULT_DIR}/http/${arm}" \
        || echo "FAILED: some ${arm} runs (continuing)"
    if [ "${DRY_RUN}" != "1" ]; then
        kill "${HTTP_PID}" 2>/dev/null || true
        wait "${HTTP_PID}" 2>/dev/null || true
        HTTP_PID=""
    fi
}

if has_section http; then
    echo "==> http: PagedServe server"
    run_cmd mkdir -p "${RESULT_DIR}/http"
    PS_SERVER=(python -m pagedserve.server --model "${MODEL}" --port "${HTTP_PORT}"
               --dtype "${HTTP_DTYPE}" --max-seq-len "${HTTP_MAX_SEQ_LEN}"
               --max-num-seqs "${HTTP_MAX_NUM_SEQS}" --block-size 16)
    if [ -n "${NUM_BLOCKS}" ]; then
        PS_SERVER+=(--num-blocks "${NUM_BLOCKS}")
    fi
    if [ -n "${DEVICE:-}" ]; then
        PS_SERVER+=(--device "${DEVICE}")
    fi
    http_sweep ps-http "${RESULT_DIR}/http/ps-server.log" "${PS_SERVER[@]}"

    echo "==> http: vLLM server"
    if [ -z "${VLLM_PYTHON}" ] && [ "${INSTALL_VLLM}" = "1" ]; then
        run_cmd python -m venv .venv-vllm
        run_cmd .venv-vllm/bin/pip install -q "vllm${VLLM_VERSION:+==${VLLM_VERSION}}"
        VLLM_PYTHON=.venv-vllm/bin/python
    fi
    if [ -n "${VLLM_PYTHON}" ]; then
        [ "${DRY_RUN}" = "1" ] || "${VLLM_PYTHON}" -c "import vllm; print('vllm', vllm.__version__)" \
            | tee -a "${RESULT_DIR}/run_notes.txt" || true
        # --host: vLLM listens on every interface by default, which on a lab
        # network exposes the server to every other machine. PagedServe's
        # server binds localhost; this matches it.
        VLLM_SERVER=("${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server
                     --host 127.0.0.1
                     --model "${MODEL}" --port "${HTTP_PORT}" --dtype "${HTTP_DTYPE}"
                     --max-model-len "${HTTP_MAX_SEQ_LEN}"
                     --max-num-seqs "${HTTP_MAX_NUM_SEQS}" --block-size 16)
        if [ -n "${NUM_BLOCKS}" ]; then
            VLLM_SERVER+=(--num-gpu-blocks-override "${NUM_BLOCKS}")
        fi
        # shellcheck disable=SC2206
        VLLM_SERVER+=(${VLLM_EXTRA_ARGS})
        http_sweep vllm-http "${RESULT_DIR}/http/vllm-server.log" "${VLLM_SERVER[@]}"
    else
        echo "skipping vLLM: set VLLM_PYTHON or INSTALL_VLLM=1"
    fi
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
