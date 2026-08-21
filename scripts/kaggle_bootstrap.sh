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
    git -C "${CHECKOUT}" fetch --depth 1 origin "${REPO_BRANCH}"
    git -C "${CHECKOUT}" checkout -f "${REPO_BRANCH}"
    git -C "${CHECKOUT}" reset --hard "origin/${REPO_BRANCH}"
else
    git clone --depth 1 --branch "${REPO_BRANCH}" "${REPO_URL}" "${CHECKOUT}"
fi
cd "${CHECKOUT}"

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
print("bf16      ", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
python -c "import pagedserve; print('pagedserve', pagedserve.__version__)"
pytest -q -m "not gpu"

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
