#!/usr/bin/env bash
# Bring PagedServe up from nothing in an ephemeral notebook (Kaggle or Colab).
#
# Kaggle and Colab sessions keep no filesystem between runs, so this script
# assumes an empty machine every time: clone, install, verify, run. It is also
# the fastest way to find out whether the CUDA extension still compiles on a
# card you do not own.
#
# What a notebook GPU is and is not for (AGENTS.md section 4): it is for GPU
# smoke tests, "does the kernel build", and "does a real model emit correct
# tokens". It is NOT for any number that gets reported. T4 and P100 instances
# are shared and unpinnable, so their timings are not reproducible evidence.
#
# Usage, from a notebook cell:
#   !bash <(curl -sSL https://raw.githubusercontent.com/Vivek-Chaudhari30/Paged_Serve/main/scripts/kaggle_bootstrap.sh)
# or, from a checkout:
#   !bash scripts/kaggle_bootstrap.sh

set -euo pipefail

# This script installs into the ambient Python and clones into the working
# directory, which is correct for a throwaway notebook VM and destructive on a
# development machine. Refuse to run outside a recognised ephemeral environment
# unless the caller explicitly says otherwise.
if [ -z "${ALLOW_NON_EPHEMERAL:-}" ] && [ ! -d /kaggle ] \
   && [ -z "${COLAB_RELEASE_TAG:-}" ] && [ -z "${COLAB_GPU:-}" ]; then
    cat >&2 <<'WARN'
refusing to run: this does not look like a Kaggle or Colab session.

It pip-installs into whatever `python` resolves to and clones the repo into the
current directory, neither of which you want on a machine you keep. If you
really mean it:

    ALLOW_NON_EPHEMERAL=1 bash scripts/kaggle_bootstrap.sh
WARN
    exit 1
fi

REPO_URL="${REPO_URL:-https://github.com/Vivek-Chaudhari30/Paged_Serve.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
# Kaggle mounts /kaggle/working; Colab and a plain shell do not, so fall back
# to the current directory rather than failing on a path that does not exist.
WORKDIR="${WORKDIR:-/kaggle/working}"
if [ ! -d "${WORKDIR}" ]; then
    WORKDIR="$(pwd)"
fi
CHECKOUT="${CHECKOUT:-${WORKDIR}/Paged_Serve}"

echo "==> environment"
nvidia-smi || echo "WARNING: no nvidia-smi; this session has no GPU."
python --version

echo "==> checkout"
if [ -d "${CHECKOUT}/.git" ]; then
    # An explicit refspec is required. A shallow clone made with --branch main
    # configures a refspec for main ALONE, so `fetch origin <other-branch>`
    # lands in FETCH_HEAD and creates no ref to check out -- the checkout then
    # fails with a bare "pathspec did not match", which reads like a typo
    # rather than a missing ref.
    git -C "${CHECKOUT}" fetch --depth 1 origin \
        "+refs/heads/${REPO_BRANCH}:refs/remotes/origin/${REPO_BRANCH}"
    git -C "${CHECKOUT}" checkout -f -B "${REPO_BRANCH}" \
        "refs/remotes/origin/${REPO_BRANCH}"
    git -C "${CHECKOUT}" reset --hard "refs/remotes/origin/${REPO_BRANCH}"
    git -C "${CHECKOUT}" clean -fd
else
    git clone --depth 1 --branch "${REPO_BRANCH}" "${REPO_URL}" "${CHECKOUT}"
fi
cd "${CHECKOUT}"
echo "checked out $(git rev-parse --abbrev-ref HEAD) at $(git rev-parse --short HEAD)"

# A bootstrap fetched from one branch but cloning another is silent and
# expensive: the run looks healthy while testing entirely different code.
if [ ! -f scripts/gpu_smoke.py ]; then
    cat >&2 <<WRONGBRANCH

ERROR: this checkout has no scripts/gpu_smoke.py, so REPO_BRANCH=${REPO_BRANCH}
does not contain the code you meant to test. Everything below would have
exercised a different commit and reported healthy.

Re-run with the branch set explicitly, e.g.

    REPO_BRANCH=phase-3/continuous-batching bash <(curl -sSL <raw-url>)

WRONGBRANCH
    exit 1
fi

echo "==> install"
# torch is preinstalled in these images and reinstalling it is a slow way to
# break the CUDA build, so only the extras that are actually missing go in.
pip install -q -e ".[baseline,dev]"

echo "==> verify"
python - <<'PY'
import torch
print("torch     ", torch.__version__)
print("cuda      ", torch.version.cuda)
print("gpu       ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print("compute   ", f"{cap[0]}.{cap[1]}")
    # Native bf16 starts at Ampere. is_bf16_supported() counts emulation and
    # returns True on a T4, which is correct and slow.
    print("bf16 native", cap[0] >= 8)
    print("bf16 reported", torch.cuda.is_bf16_supported())
PY
python -c "import pagedserve; print('pagedserve', pagedserve.__version__)"
python -c "import transformers; print('transformers', transformers.__version__)"
pytest -q -m "not gpu"

echo "==> GPU smoke checks (the CUDA paths that have never run)"
python scripts/gpu_smoke.py || echo "SOME GPU CHECKS FAILED - see above, this is the useful output"

echo "==> golden test on GPU, in this device's native dtype"
PAGEDSERVE_TEST_DEVICE=cuda \
PAGEDSERVE_TEST_DTYPE=$(python -c "import torch;print('bfloat16' if torch.cuda.is_bf16_supported() else 'float16')") \
    pytest tests/test_golden.py -q || echo "GOLDEN TEST FAILED ON GPU - this is a bug report"

echo "==> smoke run (mock backend, proves the harness works here)"
python bench/loadgen.py --backend mock --mode closed --concurrency 8 \
    --num-requests 32 --max-tokens 16

cat <<'EOS'

==> ready.

Next, against a real model:
  python bench/loadgen.py --backend hf --model <model-id> \
      --baseline-mode static --mode closed --concurrency 32 \
      --num-requests 256 --max-tokens 128 \
      --output results/scratch/kaggle-smoke.json

Note the results/scratch/ path: it is gitignored. Notebook timings are smoke
tests, not evidence, and must not land in results/ alongside real measurements.
EOS
