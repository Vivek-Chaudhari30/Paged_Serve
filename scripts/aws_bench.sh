#!/usr/bin/env bash
#
# Launch one GPU instance, run the benchmark sweep on it, bring the results
# home, and destroy the instance.
#
# The design constraint is that this must be safe by construction rather than by
# the operator remembering to do the right thing. A forgotten g5.xlarge costs
# about a dollar an hour, forever, silently. So:
#
#   * the instance is created with --instance-initiated-shutdown-behavior
#     terminate, so a `shutdown` from inside it actually destroys it rather than
#     merely stopping it (a stopped instance still bills for its EBS volume, and
#     looks "off" in the console);
#   * its user-data runs `shutdown -h +540` before anything else, so it destroys
#     itself in nine hours no matter what happens to your laptop, your Wi-Fi, or
#     your attention;
#   * every exit path from this script -- success, failure, or Ctrl-C -- runs a
#     teardown and then polls until AWS confirms the state is `terminated`;
#   * results are copied back to this machine BEFORE teardown, so a failed
#     `git push` cannot destroy an eighteen-dollar measurement.
#
# Nothing here hardcodes a credential, an account id, an instance id, or an AMI
# id. Region, key, and security group come from the environment.
#
# ---------------------------------------------------------------------------
# Usage
#
#   scripts/aws_bench.sh --dry-run              # print every AWS call, run none
#   scripts/aws_bench.sh --branch main          # the real thing (asks first)
#   scripts/aws_bench.sh --status               # what is running right now
#   scripts/aws_bench.sh --teardown-only i-0abc # kill a box from any device
#
# Required environment (see docs/aws-runbook.md):
#   AWS_KEY_NAME       name of the EC2 key pair, e.g. pagedserve
#   SSH_KEY_PATH       path to its private key, e.g. ~/.ssh/pagedserve.pem
#   AWS_SECURITY_GROUP security group id allowing SSH from your IP only
#
# Optional:
#   AWS_REGION         default us-east-1
#   INSTANCE_TYPE      default g5.xlarge
#   VOLUME_GB          default 60
#   REPO_URL           default the public GitHub repo
#   DLAMI_SSM_PARAM    SSM path to the Deep Learning AMI id
#   SWEEP_ENV          extra env passed to the sweep, e.g. "SECTIONS=baselines"
#   PUSH_RESULTS       1 to also `git push` the results branch from the box
# ---------------------------------------------------------------------------

set -euo pipefail

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

AWS_REGION="${AWS_REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g5.xlarge}"
VOLUME_GB="${VOLUME_GB:-60}"
REPO_URL="${REPO_URL:-https://github.com/Vivek-Chaudhari30/Paged_Serve.git}"
SSH_USER="${SSH_USER:-ubuntu}"
# Resolved through SSM rather than pinned: AMI ids are per-region and change
# every time AWS rebuilds the image. A hardcoded one is wrong in every region
# but the one it was copied from, and stale in that one within weeks.
DLAMI_SSM_PARAM="${DLAMI_SSM_PARAM:-/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id}"
# Published on-demand price for the default type in us-east-1 at the time of
# writing. Shown to make the cost of a mistake visible before you confirm, not
# as a billing source of truth -- check the EC2 pricing page for your region.
HOURLY_USD="${HOURLY_USD:-1.006}"
# Nine hours. Long enough for a full sweep, short enough that a forgotten box
# costs a night's sleep and not a month's credits.
DEADMAN_MINUTES="${DEADMAN_MINUTES:-540}"
RESULTS_LOCAL_DIR="${RESULTS_LOCAL_DIR:-results}"
REMOTE_DIR="/home/${SSH_USER}/pagedserve"

# KV blocks for the sweep, passed explicitly rather than profiled.
#
# profile_num_blocks measures activation memory as zero unless a caller supplies
# run_max_shape_forward, so its estimate is optimistic -- on a 14.6 GiB T4 it
# handed 13.1 GB to the KV cache. An OOM two hours into a metered sweep costs
# the whole session, so the sweep gets a number it cannot exceed.
#
# The default is deliberately conservative. For Qwen2.5-0.5B at block_size 16 in
# fp16 one block is 196,608 bytes, so 8192 blocks is about 1.5 GiB of a 24 GiB
# A10G -- roughly 131k cached tokens, far more than the sweep's concurrency and
# sequence lengths need. Raise it if a run reports preemptions you did not want;
# do not raise it to "use the card up".
NUM_BLOCKS="${NUM_BLOCKS:-8192}"

DRY_RUN=0
ASSUME_YES=0
BRANCH="main"
MODE="launch"
TEARDOWN_TARGET=""
INSTANCE_ID=""
PUBLIC_IP=""
TEARDOWN_DONE=0

# --------------------------------------------------------------------------
# Plumbing
# --------------------------------------------------------------------------

log() { printf '\n== %s\n' "$*" >&2; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
    cat >&2 <<'USAGE'
scripts/aws_bench.sh -- launch a GPU box, run the sweep, bring the results home,
and destroy the box. Every exit path tears down and confirms `terminated`.

  --dry-run              print every AWS and SSH command, execute none
  --branch <name>        branch to clone on the box (default: main)
  --yes                  skip the typed confirmation
  --status               list every instance in the region that is still billing
  --teardown-only <id>   terminate one instance and confirm it is gone
  --help

Required environment:
  AWS_KEY_NAME        EC2 key pair name
  SSH_KEY_PATH        path to its private key
  AWS_SECURITY_GROUP  security group id (SSH from your IP only)

Optional:
  AWS_REGION (us-east-1)  INSTANCE_TYPE (g5.xlarge)  VOLUME_GB (60)
  NUM_BLOCKS              KV blocks for the sweep; see the comment in this file
  SWEEP_ENV               extra env for the sweep, e.g. "SECTIONS=baselines"
  PUSH_RESULTS=1          also git push a results branch from the box

Full walkthrough, including account setup: docs/aws-runbook.md
USAGE
    exit 0
}

# Every AWS call goes through here so --dry-run can print the whole plan without
# an account, without quota, and without spending anything.
aws_do() {
    if [ "${DRY_RUN}" = "1" ]; then
        printf 'DRY_RUN:'
        printf ' %q' aws "$@"
        printf '\n' >&2
        return 0
    fi
    aws "$@"
}

# Reads are safe to execute even in a dry run -- they cost nothing and change
# nothing -- but must not be *required*, because a dry run has to work with no
# credentials at all.
aws_read() {
    if [ "${DRY_RUN}" = "1" ]; then
        printf 'DRY_RUN(read):'
        printf ' %q' aws "$@"
        printf '\n' >&2
        return 0
    fi
    aws "$@"
}

remote() {
    if [ "${DRY_RUN}" = "1" ]; then
        printf 'DRY_RUN(ssh): %s\n' "$*" >&2
        return 0
    fi
    # BatchMode: fail rather than sit at an interactive prompt. An unattended
    # script blocked on a yes/no question is a script that is still billing.
    ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
        -o ConnectTimeout=15 -i "${SSH_KEY_PATH}" \
        "${SSH_USER}@${PUBLIC_IP}" "$@"
}

require_env() {
    local name="$1"
    if [ -z "${!name:-}" ]; then
        die "${name} is not set. See docs/aws-runbook.md."
    fi
}

# --------------------------------------------------------------------------
# Teardown -- the most important function in the file
# --------------------------------------------------------------------------

teardown() {
    local id="${1:-${INSTANCE_ID}}"
    [ -n "${id}" ] || return 0
    [ "${TEARDOWN_DONE}" = "1" ] && return 0
    TEARDOWN_DONE=1

    log "terminating ${id}"
    aws_do ec2 terminate-instances --region "${AWS_REGION}" --instance-ids "${id}" \
        --output text >/dev/null 2>&1 ||
        printf 'WARNING: terminate-instances call failed. CHECK THE CONSOLE.\n' >&2

    if [ "${DRY_RUN}" = "1" ]; then
        printf 'DRY_RUN: would poll until state == terminated\n' >&2
        return 0
    fi

    # Poll rather than trust the API's acknowledgement. "I thought I shut it
    # down" is the single most common way people lose their credits.
    local state="" i=0
    while [ "${i}" -lt 40 ]; do
        state=$(aws ec2 describe-instances --region "${AWS_REGION}" \
            --instance-ids "${id}" \
            --query 'Reservations[].Instances[].State.Name' \
            --output text 2>/dev/null || echo "unknown")
        case "${state}" in
            terminated|shutting-down) break ;;
        esac
        i=$((i + 1))
        sleep 10
    done

    printf '\n'
    printf 'instance:  %s\n' "${id}"
    printf 'region:    %s\n' "${AWS_REGION}"
    if [ "${state}" != "terminated" ] && [ "${state}" != "shutting-down" ]; then
        printf 'WARNING: state is "%s", not terminated. OPEN THE CONSOLE AND CHECK.\n' "${state}"
    fi
    # Deliberately the last line of output, whatever happened above.
    printf 'state: %s\n' "${state}"
}

on_exit() {
    local rc=$?
    if [ -n "${INSTANCE_ID}" ] && [ "${TEARDOWN_DONE}" != "1" ]; then
        if [ "${rc}" != "0" ]; then
            printf '\nRun failed (exit %s). Tearing down so it stops billing.\n' "${rc}" >&2
        fi
        teardown "${INSTANCE_ID}"
    fi
    exit "${rc}"
}
trap on_exit EXIT
trap 'exit 130' INT TERM

# --------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --yes|-y) ASSUME_YES=1 ;;
        --branch) BRANCH="${2:?--branch needs a value}"; shift ;;
        --status) MODE="status" ;;
        --teardown-only) MODE="teardown"; TEARDOWN_TARGET="${2:?--teardown-only needs an instance id}"; shift ;;
        --help|-h) usage ;;
        *) die "unknown argument: $1 (try --help)" ;;
    esac
    shift
done

command -v aws >/dev/null 2>&1 || die "the AWS CLI is not installed. brew install awscli"

# --------------------------------------------------------------------------
# --status and --teardown-only
# --------------------------------------------------------------------------

if [ "${MODE}" = "status" ]; then
    log "instances in ${AWS_REGION} that are not terminated"
    aws_read ec2 describe-instances --region "${AWS_REGION}" \
        --filters "Name=instance-state-name,Values=pending,running,stopping,stopped" \
        --query 'Reservations[].Instances[].{id:InstanceId,type:InstanceType,state:State.Name,launched:LaunchTime,ip:PublicIpAddress}' \
        --output table
    printf '\nAnything listed above is billing. Kill it with:\n' >&2
    printf '  scripts/aws_bench.sh --teardown-only <id>\n' >&2
    exit 0
fi

if [ "${MODE}" = "teardown" ]; then
    teardown "${TEARDOWN_TARGET}"
    exit 0
fi

# --------------------------------------------------------------------------
# Launch
# --------------------------------------------------------------------------

if [ "${DRY_RUN}" != "1" ]; then
    require_env AWS_KEY_NAME
    require_env SSH_KEY_PATH
    require_env AWS_SECURITY_GROUP
    [ -f "${SSH_KEY_PATH}" ] || die "SSH_KEY_PATH does not exist: ${SSH_KEY_PATH}"
fi
AWS_KEY_NAME="${AWS_KEY_NAME:-<AWS_KEY_NAME>}"
SSH_KEY_PATH="${SSH_KEY_PATH:-<SSH_KEY_PATH>}"
AWS_SECURITY_GROUP="${AWS_SECURITY_GROUP:-<AWS_SECURITY_GROUP>}"

log "plan"
cat >&2 <<PLAN
  region          ${AWS_REGION}
  instance type   ${INSTANCE_TYPE}
  root volume     ${VOLUME_GB} GB gp3
  branch          ${BRANCH}
  key pair        ${AWS_KEY_NAME}
  security group  ${AWS_SECURITY_GROUP}
  dead-man switch shutdown -h +${DEADMAN_MINUTES} (self-destructs)
  approx cost     \$${HOURLY_USD}/hour on demand

  A full sweep is a few hours. Budget roughly \$15-20 for this run.
PLAN

if [ "${ASSUME_YES}" != "1" ] && [ "${DRY_RUN}" != "1" ]; then
    printf '\nType LAUNCH to continue: ' >&2
    read -r confirmation
    [ "${confirmation}" = "LAUNCH" ] || die "not confirmed; nothing was launched"
fi

log "resolving the Deep Learning AMI"
if [ "${DRY_RUN}" = "1" ]; then
    aws_read ssm get-parameters --region "${AWS_REGION}" --names "${DLAMI_SSM_PARAM}" \
        --query 'Parameters[0].Value' --output text
    AMI_ID="<resolved-ami-id>"
else
    AMI_ID=$(aws ssm get-parameters --region "${AWS_REGION}" --names "${DLAMI_SSM_PARAM}" \
        --query 'Parameters[0].Value' --output text)
    [ -n "${AMI_ID}" ] && [ "${AMI_ID}" != "None" ] ||
        die "could not resolve an AMI from ${DLAMI_SSM_PARAM}. See docs/aws-runbook.md."
    printf 'AMI: %s\n' "${AMI_ID}" >&2
fi

# The dead-man switch is the first thing the machine does, before any of our
# code runs, so a failure in our code cannot leave it alive.
USER_DATA=$(cat <<EOF
#!/bin/bash
shutdown -h +${DEADMAN_MINUTES}
EOF
)

log "launching"
if [ "${DRY_RUN}" = "1" ]; then
    aws_do ec2 run-instances \
        --region "${AWS_REGION}" \
        --image-id "${AMI_ID}" \
        --instance-type "${INSTANCE_TYPE}" \
        --key-name "${AWS_KEY_NAME}" \
        --security-group-ids "${AWS_SECURITY_GROUP}" \
        --instance-initiated-shutdown-behavior terminate \
        --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=${VOLUME_GB},VolumeType=gp3,DeleteOnTermination=true}" \
        --user-data "${USER_DATA}" \
        --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=pagedserve-bench}]' \
        --query 'Instances[0].InstanceId' --output text
    INSTANCE_ID="<instance-id>"
    PUBLIC_IP="<public-ip>"
else
    INSTANCE_ID=$(aws ec2 run-instances \
        --region "${AWS_REGION}" \
        --image-id "${AMI_ID}" \
        --instance-type "${INSTANCE_TYPE}" \
        --key-name "${AWS_KEY_NAME}" \
        --security-group-ids "${AWS_SECURITY_GROUP}" \
        --instance-initiated-shutdown-behavior terminate \
        --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=${VOLUME_GB},VolumeType=gp3,DeleteOnTermination=true}" \
        --user-data "${USER_DATA}" \
        --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=pagedserve-bench}]' \
        --query 'Instances[0].InstanceId' --output text)
    [ -n "${INSTANCE_ID}" ] || die "run-instances returned no instance id"
    printf 'instance: %s\n' "${INSTANCE_ID}" >&2
    printf 'If this script dies from here on, kill it with:\n' >&2
    printf '  scripts/aws_bench.sh --teardown-only %s\n' "${INSTANCE_ID}" >&2

    log "waiting for the instance to run"
    aws ec2 wait instance-running --region "${AWS_REGION}" --instance-ids "${INSTANCE_ID}"
    PUBLIC_IP=$(aws ec2 describe-instances --region "${AWS_REGION}" \
        --instance-ids "${INSTANCE_ID}" \
        --query 'Reservations[].Instances[].PublicIpAddress' --output text)
    [ -n "${PUBLIC_IP}" ] || die "instance has no public IP"
    printf 'ip: %s\n' "${PUBLIC_IP}" >&2

    log "waiting for SSH"
    # instance-running means the hypervisor started it, not that sshd is up.
    ssh_ready=0
    for _ in $(seq 1 60); do
        if ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
            -o ConnectTimeout=10 -i "${SSH_KEY_PATH}" \
            "${SSH_USER}@${PUBLIC_IP}" true 2>/dev/null; then
            ssh_ready=1
            break
        fi
        sleep 10
    done
    [ "${ssh_ready}" = "1" ] || die "SSH never came up. Is the security group allowing your IP?"
fi

# --------------------------------------------------------------------------
# On the box
# --------------------------------------------------------------------------

log "recording the environment"
remote "nvidia-smi; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; uname -a"

log "cloning ${BRANCH}"
remote "rm -rf ${REMOTE_DIR} && git clone --branch ${BRANCH} --depth 50 ${REPO_URL} ${REMOTE_DIR}"

log "installing"
remote "cd ${REMOTE_DIR} && python3 -m venv .venv && . .venv/bin/activate && pip install -q --upgrade pip && pip install -q -e '.[engine,baseline,dev]'"

log "building the CUDA extension"
# Explicitly, never through pip: pip builds in an isolated environment with no
# torch, so the extension silently would not build (AGENTS.md section 7).
remote "cd ${REMOTE_DIR} && . .venv/bin/activate && python setup.py build_ext --inplace"

log "writing the environment record"
remote "cd ${REMOTE_DIR} && . .venv/bin/activate && {
    echo 'instance_type=${INSTANCE_TYPE}'
    echo \"ami=${AMI_ID}\"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    python -c 'import torch;print(\"torch\",torch.__version__);print(\"cuda\",torch.version.cuda);print(\"capability\",torch.cuda.get_device_capability())'
} > environment.txt && cat environment.txt"

log "correctness gate"
# Two commands, because they cover different things and neither is sufficient.
#
#   -m gpu   the ~10 tests that cannot run without CUDA at all: the real
#            profile_num_blocks measurement and the extension build canary.
#
#   the full suite with PAGEDSERVE_TEST_DEVICE=cuda   the golden gate. It is
#            device-parametric rather than gpu-marked, so `-m gpu` does NOT
#            include it. A green `-m gpu` alone would mean the commit gate never
#            ran on this card, and we would benchmark an unverified engine.
#
# Either failing aborts the run. Measuring a broken engine produces numbers that
# are confident and wrong, which is worse than no numbers.
remote "cd ${REMOTE_DIR} && . .venv/bin/activate && pytest -m gpu -q" ||
    die "pytest -m gpu failed. Not benchmarking an engine that fails its own tests."
remote "cd ${REMOTE_DIR} && . .venv/bin/activate && PAGEDSERVE_TEST_DEVICE=cuda PAGEDSERVE_TEST_DTYPE=float16 pytest -q" ||
    die "the golden gate failed on CUDA. This is a bug report, not an obstacle."

log "fetching the dataset"
remote "cd ${REMOTE_DIR} && scripts/fetch_dataset.sh"

log "smoke sweep"
# Tiny, first, to prove the pipeline writes valid JSON before an hour is spent
# discovering that it does not.
remote "cd ${REMOTE_DIR} && . .venv/bin/activate && \
    SECTIONS=baselines CONCURRENCIES='1 8' NUM_REQUESTS=32 REPEATS=1 \
    RESULT_DIR=smoke SKIP_INSTALL=1 bash scripts/explorer_job.sbatch" ||
    die "the smoke sweep failed. Fix it before spending an hour on the full one."

log "full sweep (nohup, teed to sweep.log)"
# nohup so a dropped SSH connection does not kill hours of work. --num-blocks is
# passed explicitly because profile_num_blocks is optimistic without a profiling
# forward pass, and an OOM two hours in costs the whole session.
remote "cd ${REMOTE_DIR} && . .venv/bin/activate && \
    nohup env ${SWEEP_ENV:-} NUM_BLOCKS=${NUM_BLOCKS} SKIP_INSTALL=1 \
    bash scripts/explorer_job.sbatch > sweep.log 2>&1 &
    echo started"

log "waiting for the sweep (tail sweep.log to watch)"
if [ "${DRY_RUN}" != "1" ]; then
    while remote "pgrep -f explorer_job.sbatch >/dev/null" 2>/dev/null; do
        remote "tail -n 2 sweep.log" || true
        sleep 120
    done
fi

# Results come home BEFORE teardown. A push that fails after the instance is
# gone has destroyed the measurement; a copy that succeeded has not.
log "copying results back to ${RESULTS_LOCAL_DIR}/"
if [ "${DRY_RUN}" = "1" ]; then
    printf 'DRY_RUN(scp): results/ and sweep.log and environment.txt -> %s/\n' \
        "${RESULTS_LOCAL_DIR}" >&2
else
    mkdir -p "${RESULTS_LOCAL_DIR}"
    scp -r -o StrictHostKeyChecking=accept-new -i "${SSH_KEY_PATH}" \
        "${SSH_USER}@${PUBLIC_IP}:${REMOTE_DIR}/results/*" "${RESULTS_LOCAL_DIR}/" ||
        printf 'WARNING: no results were copied back.\n' >&2
    scp -o StrictHostKeyChecking=accept-new -i "${SSH_KEY_PATH}" \
        "${SSH_USER}@${PUBLIC_IP}:${REMOTE_DIR}/sweep.log" \
        "${SSH_USER}@${PUBLIC_IP}:${REMOTE_DIR}/environment.txt" \
        "${RESULTS_LOCAL_DIR}/" || true
fi

if [ "${PUSH_RESULTS:-0}" = "1" ]; then
    log "pushing the results branch from the box"
    # Optional and never the only copy. Pushing from EC2 needs a credential on
    # the box; this script never puts one there, so configure a deploy key
    # yourself if you want this path. It is expected to fail otherwise, and that
    # failure must not fail the run -- the results are already local.
    remote "cd ${REMOTE_DIR} && git checkout -b results/$(date -u +%Y%m%d) && \
        git add results/ && git -c user.name=aws-bench -c user.email=aws-bench@local \
        commit -m 'Add sweep results' && git push -u origin HEAD" ||
        printf 'WARNING: push failed. The results are already local; push them from here.\n' >&2
fi

log "done"
# teardown runs from the EXIT trap, and prints the confirmed state last.
