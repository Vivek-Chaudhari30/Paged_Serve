# Running the benchmark on a shared lab GPU box

For a plain Linux machine you SSH into, with an NVIDIA card and no Slurm, such as a lab's
RTX A6000 machines. The job is the same `scripts/explorer_job.sbatch` that runs on Explorer;
under `bash` its `#SBATCH` lines are comments.

What is different from a cluster or a rented box, and what the script now does about each:

| Risk on a shared box | What handles it |
|---|---|
| Someone else's job on the same card | The job refuses to start on a GPU with processes or more than 2 GB already in use (`ALLOW_SHARED_GPU=1` overrides, and marks the run as not evidence). It samples the card every 30 s into `gpu_monitor.csv`, so a job that lands mid-sweep shows up afterwards. |
| A multi-GPU machine | `CUDA_VISIBLE_DEVICES` defaults to `0`; set it to the free card. It is written to `run_notes.txt`. |
| Another server on the HTTP port | The `http` section checks the port is free before starting each server. Change it with `HTTP_PORT`. |
| conda `(base)` as the shell's Python | An active conda env or virtualenv is used as-is. The script no longer builds a second venv on top of it, which would install a second torch and break the CUDA extension. |
| Python older than 3.11 | The job stops at once with the fix, instead of failing inside pip. |
| An engine that fails its own tests | The job builds the CUDA extension, checks it loads, runs `pytest -m gpu` and the golden suite on cuda, and aborts if any of them fail. |
| Cache sized from free memory someone else can take | `NUM_BLOCKS` defaults to 8192. `NUM_BLOCKS=` (empty) profiles instead. |
| The SSH session dropping | Not something a script can fix: run it inside `tmux`. |

Each result directory's `run_notes.txt` records the host, the card, the commit, and whether
the tree had uncommitted changes.

## Steps

Pick one free machine and use it for every run. A6000 (`sm_86`) and RTX 6000 Ada (`sm_89`)
machines are different hardware; never mix them in one result set. Tell the lab you are using
the card.

```bash
ssh <you>@<machine>
```
```bash
tmux new -s pagedserve
```
```bash
nvidia-smi
```

Create a Python 3.11 env, then clone and install. Put the Hugging Face cache somewhere with
space; a home directory quota is easy to fill with model weights.

```bash
conda create -y -n pagedserve python=3.11 && conda activate pagedserve
```
```bash
git clone https://github.com/Vivek-Chaudhari30/Paged_Serve.git && cd Paged_Serve
```
```bash
pip install -e ".[engine,baseline,dev]"
```
```bash
export HF_HOME=/path/with/space/hf
```

The extension needs `nvcc`, matching torch's CUDA major version. Check both:

```bash
nvcc --version; python -c "import torch; print(torch.version.cuda, torch.cuda.get_device_name())"
```

If `nvcc` is missing, install the toolkit into the env with conda (`conda install -c nvidia
cuda-toolkit=<torch's version>`).

### A short smoke run first

It builds the extension, runs the gate, and does a handful of runs. Time it: its log is how
you size the full run.

```bash
SKIP_INSTALL=1 SECTIONS=arms CONCURRENCIES="1 8" NUM_REQUESTS=16 REPEATS=1 RESULT_DIR=/tmp/ps-smoke bash scripts/explorer_job.sbatch 2>&1 | tee smoke.log
```

If the build or the golden gate fails, stop and report the full error. The extension has
never been compiled for `sm_86` before.

### vLLM, in its own env

vLLM pins its own torch, so it must not share the env the extension was built in.

```bash
python -m venv .venv-vllm && .venv-vllm/bin/pip install vllm
```

### The full run

Everything in one go, about 12–16 hours. The gate runs again at the start; pass `SKIP_GATE=1`
if it already passed in the smoke run on this exact commit.

```bash
SKIP_INSTALL=1 SECTIONS="baselines arms prefix poisson microbench ablations http" VLLM_PYTHON=$PWD/.venv-vllm/bin/python bash scripts/explorer_job.sbatch 2>&1 | tee run.log
```

Detach with `Ctrl-b d`; reattach with `tmux attach -t pagedserve`. Results land in
`results/<timestamp>-local/`.

After it finishes, look at `gpu_monitor.csv` for memory or utilization outside your own runs
before trusting any number from that window.

## Where these numbers can go

`AGENTS.md` section 4 names the Explorer A100 as the source of README numbers. A6000 results
are real measurements of different hardware: either change that rule deliberately, or use
this machine for development and rehearsal and keep the README for the A100 run. Do not mix
the two in one result set.
