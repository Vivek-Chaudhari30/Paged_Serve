# Running the benchmark on AWS — a runbook for someone who has never used AWS

This walks the whole thing: account, permissions, quota, keys, launch, sweep, teardown.
Every step says **why** it exists, because the steps you do not understand are the ones you
skip, and the one people skip is the one that costs money.

If you only remember one thing: **an instance bills for every second it exists, whether or
not you are using it.** Everything below is arranged around making that impossible to forget.

---

## 0. The mental model, in four sentences

AWS rents you computers by the hour. You pick a machine type — `g5.xlarge` is one NVIDIA
A10G with 24 GB — start it, SSH in like any Linux box, do your work, and **destroy it**. The
disk (an **EBS volume**) can outlive the machine, so a large dataset does not need
re-downloading, but a volume left behind also bills. "Terminate" destroys; "stop" only
pauses, and a stopped instance still bills for its disk while looking switched off in the
console.

---

## 1. Do this today: the quota request gates everything

A brand-new AWS account is allowed **zero** vCPUs of GPU instance. You cannot launch a
`g5.xlarge` until you ask permission, and approval takes **1–3 days**. Start it before you
do anything else in this document; everything in Wave 1 of `FINAL-COMPLETION.md` can be
done on your laptop while you wait.

1. Create an account at `aws.amazon.com`. Stay on the **Free plan**; do not upgrade.
2. **Turn on MFA for the root account.** Console → your name (top right) → Security
   credentials → assign MFA device.
   *Why:* the root account can do anything, including running up an unbounded bill. A
   compromised GPU account mining crypto on someone else's credits is a routine event, not
   a hypothetical.
3. **Set a budget before you launch anything.** Billing and Cost Management → Budgets →
   Create budget → Cost budget → **$80/month** → alert at **50%, 80%, 100%** to your email.
   *Why:* this is the seatbelt. It does not stop spending; it tells you that spending is
   happening, which is the part you cannot otherwise see until the invoice.
4. **Request the GPU quota.** Service Quotas → AWS services → **Amazon EC2** → search
   **"Running On-Demand G and VT instances"** → Request increase → **8 vCPUs**.
   - `g5.xlarge` uses 4 vCPUs; 8 leaves headroom to relaunch before an old box is fully
     released.
   - Justification, written plainly: *"Machine learning research — benchmarking a custom
     LLM inference engine on a single A10G GPU. Short, intermittent sessions."*
   - **Pick a region and stay in it.** `us-east-1` (N. Virginia) is usually cheapest and has
     the most capacity. Quotas, key pairs, security groups, and AMI ids are all *per-region*;
     a quota approved in one region does nothing in another.

---

## 2. Local setup

```bash
brew install awscli
aws configure
```

`aws configure` asks for an **Access Key ID**, a **Secret Access Key**, a **default region**
(use the one from step 1.4), and an output format (`json`).

To get the keys: Console → IAM → Users → Create user → attach **`AmazonEC2FullAccess`** →
Security credentials → Create access key → "Command Line Interface".

> **Never create access keys on the root account, and never paste a key into a chat, a file
> in this repo, or a commit.** `aws configure` writes them to `~/.aws/credentials`, outside
> the repo. If a key ever leaks, deactivate it in IAM immediately — rotating is cheap,
> cleaning up after a leaked key is not.

The launch script also needs to read one SSM parameter (to resolve the current Deep Learning
AMI id). `AmazonEC2FullAccess` does not include that; add **`AmazonSSMReadOnlyAccess`** to
the same user, or pass an AMI id yourself with `DLAMI_SSM_PARAM` unset and the id hardcoded
in your shell.

### The key pair

Console → EC2 → Key pairs → Create key pair → name it `pagedserve`, type **ED25519**, format
**.pem**. The private key downloads once and can never be downloaded again.

```bash
mv ~/Downloads/pagedserve.pem ~/.ssh/
chmod 400 ~/.ssh/pagedserve.pem
```

`chmod 400` is not optional — SSH refuses to use a private key that other users can read.

### The security group

Console → EC2 → Security groups → Create security group. One inbound rule:

| Type | Port | Source |
|---|---|---|
| SSH | 22 | **My IP** |

**Not `0.0.0.0/0`.** An open SSH port on a GPU box is found by automated scanners within
hours. Note the group id (`sg-…`) — the script needs it.

Your home IP changes. If SSH stops connecting later, edit the rule's source back to "My IP"
before assuming the instance is broken.

---

## 3. Environment for the script

```bash
export AWS_REGION=us-east-1
export AWS_KEY_NAME=pagedserve
export SSH_KEY_PATH=~/.ssh/pagedserve.pem
export AWS_SECURITY_GROUP=sg-0123456789abcdef0
```

Put these in your shell profile so you cannot forget one mid-session. They contain no
secrets — the private key stays on disk, and the key *name* is not sensitive.

---

## 4. Rehearse before you spend

```bash
scripts/aws_bench.sh --dry-run
```

This prints every AWS call and every remote command it would run, and executes none of
them. It needs no quota, no credentials, and no money. Read the output once. You are looking
for: the right region, the right instance type, `--instance-initiated-shutdown-behavior
terminate`, the `shutdown -h +540` in the user-data, and a `terminating` step at the end.

---

## 5. The real run

```bash
scripts/aws_bench.sh --branch main
```

It prints the plan and the hourly cost, then waits for you to type `LAUNCH`. What it does:

1. Resolves the current **Deep Learning AMI (Ubuntu)** — it ships NVIDIA drivers, CUDA and
   PyTorch already installed. Building that stack yourself burns an hour of paid GPU time.
2. Launches one `g5.xlarge` with a 60 GB gp3 root volume, shutdown behaviour **terminate**,
   and a `shutdown -h +540` dead-man's switch in its user-data. The switch runs before any
   of our code, so a failure in our code cannot leave the box alive.
3. Waits for SSH, clones the branch, installs, and builds the CUDA extension **explicitly**:
   ```bash
   python setup.py build_ext --inplace
   ```
   Never through `pip install` — pip builds in an isolated environment with no torch, so the
   extension silently would not build.
4. Records `nvidia-smi`, driver, CUDA, torch, and compute capability into `environment.txt`.
   A result that does not name its hardware is not a result (`AGENTS.md` §6).
5. **Runs the correctness gate and aborts the whole run if it fails.** Two commands, because
   neither alone is sufficient:
   - `pytest -m gpu` — the ~10 tests that cannot run without CUDA at all.
   - `PAGEDSERVE_TEST_DEVICE=cuda PAGEDSERVE_TEST_DTYPE=float16 pytest` — the golden gate,
     which is device-parametric and **is not selected by `-m gpu`**. A green `-m gpu` alone
     means the commit gate never ran on this card.

   *Why abort:* measuring a broken engine produces numbers that are confident and wrong,
   which is worse than having no numbers.
6. Fetches the dataset, runs a **tiny smoke sweep** to prove the pipeline writes valid JSON,
   and only then runs the full sweep under `nohup`, teed to `sweep.log`, so a dropped SSH
   connection does not kill hours of work.
7. **Copies the results back to your laptop before teardown.** A `git push` that fails after
   the instance is gone has destroyed the measurement; a copy that already succeeded has not.
8. Terminates the instance and polls until AWS confirms the state. The last line of output
   is that state.

### Watching it from elsewhere

The sweep takes hours; you do not need to sit with it.

```bash
ssh -i ~/.ssh/pagedserve.pem ubuntu@<ip> tail -50 pagedserve/sweep.log
ssh -i ~/.ssh/pagedserve.pem ubuntu@<ip> 'ls pagedserve/results/*/ | wc -l'
```

---

## 6. Shutdown discipline

The script tears down on every exit path, including failure and Ctrl-C. Verify anyway.

```bash
scripts/aws_bench.sh --status                  # anything listed is billing
scripts/aws_bench.sh --teardown-only i-0abc…   # kill one, from any device
```

Then **open the console and look with your own eyes.** "I thought I shut it down" is the
single most common way people lose their credits.

Check two other places once a month:
- **EC2 → Volumes.** A volume detached from a terminated instance still bills. The script
  sets `DeleteOnTermination=true`, so this should stay empty; if it does not, delete them.
- **EC2 → Elastic IPs.** An allocated-but-unattached elastic IP bills by the hour. This
  setup allocates none, so this should also stay empty.

---

## 7. Budget

The plan in `FINAL-COMPLETION.md` fits the remaining project inside roughly $83 of $100
credits. The first sweep is about $18 of that.

**Spot instances** cost roughly a third of on-demand and AWS can reclaim them with two
minutes' notice. That trade is good for kernel development, where losing the box costs you a
`git push`, and bad for a long unattended sweep, where it costs you the run. This script
launches on-demand only; use spot deliberately, by hand, for development.

---

## 8. When it goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `VcpuLimitExceeded` on launch | Quota not approved yet, or approved in another region | Check Service Quotas **in this region** (§1.4) |
| SSH never comes up | Security group does not allow your current IP | Edit the inbound rule's source back to "My IP" |
| `Permissions 0644 … are too open` | Private key is world-readable | `chmod 400 ~/.ssh/pagedserve.pem` |
| `could not resolve an AMI` | The user has no SSM read permission | Attach `AmazonSSMReadOnlyAccess` (§2) |
| Extension fails to compile | This box is `sm_86`; it has only ever been built on `sm_75` | **Stop and report the full error.** Do not work around it |
| Golden gate fails on CUDA | A real bug | **Stop.** A failing golden test is a bug report, not an obstacle (`AGENTS.md` §2.2) |
| Sweep OOMs partway | KV cache sized optimistically | Lower `NUM_BLOCKS`; never leave it to profiling on a metered run |

---

## 9. What has and has not been verified

`scripts/aws_bench.sh` has been checked for shell syntax, and its `--dry-run`, `--status`,
and `--teardown-only` paths have been walked end to end on a laptop.

**The live path has never been executed.** No AWS account has run it, because the GPU quota
has not been granted yet. Read the `--dry-run` output before the first real launch, and
watch the first run rather than leaving it unattended.
