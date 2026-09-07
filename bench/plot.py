"""Regenerate every README chart from committed result JSON files.

Usage
-----
::

    python -m bench.plot --results results/<dir> --out figures/

Each chart is skipped (with a message) when its required inputs are absent.
A partial sweep — only some arms or concurrency levels present — still plots
whatever it has.

matplotlib is a dev dependency (``pip install -e '.[dev]'``).  Never import
this module from runtime code; it has no place in the serve path.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# Canonical arm order for legend stability across runs.
ARM_ORDER: list[str] = [
    "hf-sequential",
    "hf-static",
    "ps-contiguous",
    "ps-paged",
    "ps-continuous",
]

_ARM_COLORS: dict[str, str] = {
    "hf-sequential": "#1f77b4",
    "hf-static": "#ff7f0e",
    "ps-contiguous": "#2ca02c",
    "ps-paged": "#d62728",
    "ps-continuous": "#9467bd",
}

_ARM_LABELS: dict[str, str] = {
    "hf-sequential": "HF sequential",
    "hf-static": "HF static-batch",
    "ps-contiguous": "PS contiguous",
    "ps-paged": "PS paged",
    "ps-continuous": "PS continuous",
}

# Each entry: (summary-key-suffix, legend-label, linestyle).
_PCTILE_STYLES: list[tuple[str, str, str]] = [
    ("p50", "P50", "-"),
    ("p95", "P95", "--"),
    ("p99", "P99", ":"),
]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class RunRecord:
    """One committed result JSON file, parsed into the scalar fields needed for plotting.

    Only the precomputed summary statistics are stored here; the per-request
    timings in ``requests`` are not re-read, because the summary was already
    computed when the file was written.
    """

    path: Path
    arm: str
    gpu: str
    model: str
    dataset: str
    concurrency: int | None
    max_tokens: int | None
    # workload.prefix_fraction if the sweep set it; None otherwise.
    prefix_fraction: float | None
    # True when workload.prefix_cache is a non-empty dict.
    prefix_cache_on: bool
    # workload.prefix_cache.hit_rate, or None.
    prefix_hit_rate: float | None
    # summary.output_throughput (tokens / s).
    output_throughput: float | None
    ttft_p50: float | None
    ttft_p95: float | None
    ttft_p99: float | None
    itl_p50: float | None
    itl_p95: float | None
    itl_p99: float | None
    # summary.kv_utilization or workload.kv_stats.mean_utilization.
    kv_utilization: float | None


# ---------------------------------------------------------------------------
# Arm inference
# ---------------------------------------------------------------------------


def _infer_arm(result: dict[str, Any]) -> str | None:
    """Return the ablation arm label, or None when it cannot be determined.

    Checks ``workload.arm`` first — an explicit label that sweep.py can write.
    Falls back to inference from ``config`` fields for result files that pre-date
    the field.  Returns None when the arm is genuinely ambiguous; the caller logs
    and skips the file.
    """
    arm: str | None = (result.get("workload") or {}).get("arm")
    if arm:
        return arm

    config = result.get("config") or {}
    backend = config.get("backend")

    if backend == "hf":
        bc = config.get("backend_config") or {}
        return "hf-sequential" if (bc.get("max_batch_size") or 1) <= 1 else "hf-static"

    if backend == "pagedserve":
        engine_cfg = config.get("engine") or {}
        attn = engine_cfg.get("attn_backend") or engine_cfg.get("attention_backend") or ""
        if attn == "contiguous":
            return "ps-contiguous"
        # Distinguish paged-static from paged-continuous.
        # StaticEngineBackend completes one batch per engine.step(), so the
        # number of steps is close to the number of batches.  PagedServeBackend
        # calls step() many times per batch, so steps >> num_batches.
        workload = result.get("workload") or {}
        sched = workload.get("scheduler") or {}
        batching = workload.get("batching") or {}
        steps: int = sched.get("steps") or 0
        num_batches: int = batching.get("num_batches") or 0
        if num_batches > 0 and 0 < steps <= num_batches * 2:
            return "ps-paged"
        return "ps-continuous"

    return None


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------


def _parse_result(path: Path) -> RunRecord | None:
    """Read one result JSON and return a RunRecord, or None when the file is unusable.

    A file is unusable when it lacks a GPU label (AGENTS.md §6: a chart that
    cannot state its hardware is not allowed to render) or a model name.
    """
    try:
        raw: dict[str, Any] = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        logger.warning("skipping %s: JSON parse error: %s", path, exc)
        return None

    env = raw.get("environment") or {}
    gpu: str | None = env.get("gpu")
    if not gpu:
        logger.warning(
            "skipping %s: environment.gpu is absent — AGENTS.md §6 requires every "
            "chart to state its hardware",
            path,
        )
        return None

    config = raw.get("config") or {}
    engine_cfg = config.get("engine") or {}
    backend_cfg = config.get("backend_config") or {}
    model: str | None = engine_cfg.get("model") or backend_cfg.get("model") or config.get("model")
    if not model:
        logger.warning("skipping %s: model name not found in config", path)
        return None

    arm = _infer_arm(raw)
    if not arm:
        logger.warning(
            "skipping %s: cannot determine arm from config — "
            "add workload.arm explicitly to avoid this",
            path,
        )
        return None

    workload = raw.get("workload") or {}
    summary = raw.get("summary") or {}
    pc = workload.get("prefix_cache") or {}
    kv_stats = workload.get("kv_stats") or {}

    kv_util: float | None = (
        summary.get("kv_utilization")
        or kv_stats.get("mean_utilization")
        or workload.get("kv_utilization")
    )

    return RunRecord(
        path=path,
        arm=arm,
        gpu=str(gpu),
        model=str(model),
        dataset=str(workload.get("dataset") or "unknown"),
        concurrency=workload.get("concurrency"),
        max_tokens=workload.get("max_tokens"),
        prefix_fraction=workload.get("prefix_fraction"),
        prefix_cache_on=bool(pc),
        prefix_hit_rate=pc.get("hit_rate"),
        output_throughput=summary.get("output_throughput"),
        ttft_p50=summary.get("ttft_p50"),
        ttft_p95=summary.get("ttft_p95"),
        ttft_p99=summary.get("ttft_p99"),
        itl_p50=summary.get("itl_p50"),
        itl_p95=summary.get("itl_p95"),
        itl_p99=summary.get("itl_p99"),
        kv_utilization=kv_util,
    )


# ---------------------------------------------------------------------------
# Plotting primitives
# ---------------------------------------------------------------------------


def _hw_title(gpu: str, model: str, dataset: str) -> str:
    return f"{gpu} · {model} · {dataset}"


def _hw_safe(gpu: str, model: str, dataset: str) -> str:
    """Filesystem-safe directory name for a (gpu, model, dataset) triple."""
    return "_".join(part.replace(" ", "_").replace("/", "_") for part in (gpu, model, dataset))


def _median_band(
    ys_by_x: dict[float, list[float]],
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Reduce ``{x: [y, y, ...]}`` to ``(xs, medians, mins, maxs)`` for plotting.

    AGENTS.md §6 rule 4: report the median, never a bare mean.  Spread is
    min/max across repeats at each x, drawn as a shaded band.  All four return
    lists are always the same length; x values whose value list is empty are
    silently skipped.
    """
    result_xs: list[float] = []
    medians: list[float] = []
    lo: list[float] = []
    hi: list[float] = []
    for x in sorted(ys_by_x):
        vals = [v for v in ys_by_x[x] if v is not None]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        result_xs.append(x)
        medians.append(float(np.median(arr)))
        lo.append(float(arr.min()))
        hi.append(float(arr.max()))
    return result_xs, medians, lo, hi


# ---------------------------------------------------------------------------
# Chart 1 — output-token throughput vs concurrency, one line per arm
# ---------------------------------------------------------------------------


def plot_throughput_vs_concurrency(
    records: list[RunRecord],
    out_dir: Path,
    hw_label: str,
) -> bool:
    usable = [r for r in records if r.concurrency is not None and r.output_throughput is not None]
    if not usable:
        print(
            f"[throughput-vs-concurrency] skipping '{hw_label}': "
            "need workload.concurrency + summary.output_throughput"
        )
        return False

    by_arm: dict[str, dict[float, list[float]]] = {}
    for r in usable:
        by_arm.setdefault(r.arm, {}).setdefault(float(r.concurrency), []).append(
            r.output_throughput  # type: ignore[arg-type]
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    for arm in ARM_ORDER:
        ys_by_x = by_arm.get(arm)
        if not ys_by_x:
            continue
        xs, meds, lo, hi = _median_band(ys_by_x)
        color = _ARM_COLORS.get(arm, "gray")
        ax.plot(xs, meds, marker="o", color=color, label=_ARM_LABELS.get(arm, arm))
        if len(xs) > 1:
            ax.fill_between(xs, lo, hi, color=color, alpha=0.15)

    ax.set_xlabel("Concurrency (in-flight requests)")
    ax.set_ylabel("Output throughput (tokens / s)")
    ax.set_title(f"Output-token throughput vs concurrency\n{hw_label}")
    ax.legend()
    ax.grid(visible=True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    dest = out_dir / "throughput_vs_concurrency.png"
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    print(f"[throughput-vs-concurrency] wrote {dest}")
    return True


# ---------------------------------------------------------------------------
# Charts 2 & 3 — TTFT / ITL percentiles vs concurrency
# ---------------------------------------------------------------------------


def _latency_vs_concurrency(
    records: list[RunRecord],
    *,
    prefix: str,
    y_label: str,
    title: str,
    out_path: Path,
    hw_label: str,
) -> bool:
    """Shared implementation for TTFT and ITL percentile charts."""
    attr_map = {pct_key: f"{prefix}_{pct_key}" for pct_key, _, _ in _PCTILE_STYLES}
    usable = [
        r
        for r in records
        if r.concurrency is not None
        and any(getattr(r, attr_map[pct_key]) is not None for pct_key, _, _ in _PCTILE_STYLES)
    ]
    if not usable:
        print(
            f"[{prefix}-vs-concurrency] skipping '{hw_label}': "
            f"need workload.concurrency + summary.{prefix}_p*"
        )
        return False

    # by_arm[arm][pct_key][concurrency] = [values across repeats]
    by_arm: dict[str, dict[str, dict[float, list[float]]]] = {}
    for r in usable:
        arm_d = by_arm.setdefault(r.arm, {})
        for pct_key, _, _ in _PCTILE_STYLES:
            v: float | None = getattr(r, attr_map[pct_key])
            if v is not None:
                arm_d.setdefault(pct_key, {}).setdefault(float(r.concurrency), []).append(v)

    fig, ax = plt.subplots(figsize=(9, 5))
    for arm in ARM_ORDER:
        arm_d = by_arm.get(arm)
        if not arm_d:
            continue
        color = _ARM_COLORS.get(arm, "gray")
        base_label = _ARM_LABELS.get(arm, arm)
        for pct_key, pct_label, ls in _PCTILE_STYLES:
            ys_by_x = arm_d.get(pct_key)
            if not ys_by_x:
                continue
            xs, meds, lo, hi = _median_band(ys_by_x)
            ax.plot(
                xs,
                meds,
                marker="o",
                color=color,
                linestyle=ls,
                label=f"{base_label} {pct_label}",
            )
            if len(xs) > 1:
                ax.fill_between(xs, lo, hi, color=color, alpha=0.10)

    ax.set_xlabel("Concurrency (in-flight requests)")
    ax.set_ylabel(y_label)
    ax.set_title(f"{title}\n{hw_label}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(visible=True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[{prefix}-vs-concurrency] wrote {out_path}")
    return True


def plot_ttft_vs_concurrency(
    records: list[RunRecord],
    out_dir: Path,
    hw_label: str,
) -> bool:
    return _latency_vs_concurrency(
        records,
        prefix="ttft",
        y_label="TTFT (s)",
        title="Time-to-first-token P50 / P95 / P99 vs concurrency",
        out_path=out_dir / "ttft_vs_concurrency.png",
        hw_label=hw_label,
    )


def plot_itl_vs_concurrency(
    records: list[RunRecord],
    out_dir: Path,
    hw_label: str,
) -> bool:
    return _latency_vs_concurrency(
        records,
        prefix="itl",
        y_label="ITL (s)",
        title="Inter-token latency P50 / P95 / P99 vs concurrency",
        out_path=out_dir / "itl_vs_concurrency.png",
        hw_label=hw_label,
    )


# ---------------------------------------------------------------------------
# Chart 4 — TTFT vs shared-prefix fraction, cache on vs off
# ---------------------------------------------------------------------------


def plot_ttft_vs_prefix_fraction(
    records: list[RunRecord],
    out_dir: Path,
    hw_label: str,
) -> bool:
    def _x_value(r: RunRecord) -> float | None:
        if r.prefix_fraction is not None:
            return r.prefix_fraction
        # Fall back to the measured hit rate as an imperfect proxy when the
        # sweep did not record the input fraction explicitly.
        if r.prefix_cache_on and r.prefix_hit_rate is not None:
            return r.prefix_hit_rate
        return None

    on_ys: dict[float, list[float]] = {}
    off_ys: dict[float, list[float]] = {}
    for r in records:
        if r.ttft_p50 is None:
            continue
        x = _x_value(r)
        if x is None:
            continue
        target = on_ys if r.prefix_cache_on else off_ys
        target.setdefault(x, []).append(r.ttft_p50)

    if not on_ys and not off_ys:
        print(
            f"[ttft-vs-prefix-fraction] skipping '{hw_label}': "
            "need workload.prefix_fraction (or workload.prefix_cache.hit_rate) "
            "+ summary.ttft_p50"
        )
        return False

    using_explicit_fraction = any(r.prefix_fraction is not None for r in records)
    x_label = (
        "Shared-prefix fraction"
        if using_explicit_fraction
        else "Prefix cache hit rate (proxy for fraction)"
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    if on_ys:
        xs, meds, lo, hi = _median_band(on_ys)
        ax.plot(xs, meds, marker="o", color="#2ca02c", label="Cache ON")
        if len(xs) > 1:
            ax.fill_between(xs, lo, hi, color="#2ca02c", alpha=0.15)
    if off_ys:
        xs, meds, lo, hi = _median_band(off_ys)
        ax.plot(xs, meds, marker="s", color="#d62728", label="Cache OFF")
        if len(xs) > 1:
            ax.fill_between(xs, lo, hi, color="#d62728", alpha=0.15)

    ax.set_xlabel(x_label)
    ax.set_ylabel("TTFT P50 (s)")
    ax.set_title(f"TTFT vs shared-prefix fraction — cache on vs off\n{hw_label}")
    ax.legend()
    ax.grid(visible=True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    dest = out_dir / "ttft_vs_prefix_fraction.png"
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    print(f"[ttft-vs-prefix-fraction] wrote {dest}")
    return True


# ---------------------------------------------------------------------------
# Chart 5 — KV cache utilization, paged vs contiguous
# ---------------------------------------------------------------------------


def plot_kv_utilization(
    records: list[RunRecord],
    out_dir: Path,
    hw_label: str,
) -> bool:
    kv_recs = [r for r in records if r.kv_utilization is not None]
    if not kv_recs:
        print(
            f"[kv-utilization] skipping '{hw_label}': "
            "need summary.kv_utilization or workload.kv_stats.mean_utilization"
        )
        return False

    paged_recs = [r for r in kv_recs if r.arm in ("ps-paged", "ps-continuous")]
    cont_recs = [r for r in kv_recs if r.arm == "ps-contiguous"]

    if not paged_recs and not cont_recs:
        print(
            f"[kv-utilization] skipping '{hw_label}': "
            "no ps-paged / ps-continuous or ps-contiguous records with kv_utilization"
        )
        return False

    def _series(recs: list[RunRecord]) -> dict[float, list[float]]:
        d: dict[float, list[float]] = {}
        for r in recs:
            x = float(r.max_tokens or 0)
            if x > 0 and r.kv_utilization is not None:
                d.setdefault(x, []).append(r.kv_utilization)
        return d

    fig, ax = plt.subplots(figsize=(8, 5))
    if cont_recs:
        series = _series(cont_recs)
        if series:
            xs, meds, lo, hi = _median_band(series)
            ax.plot(xs, meds, marker="o", color="#1f77b4", label="Contiguous")
            if len(xs) > 1:
                ax.fill_between(xs, lo, hi, color="#1f77b4", alpha=0.15)
    if paged_recs:
        series = _series(paged_recs)
        if series:
            xs, meds, lo, hi = _median_band(series)
            ax.plot(xs, meds, marker="s", color="#9467bd", label="Paged")
            if len(xs) > 1:
                ax.fill_between(xs, lo, hi, color="#9467bd", alpha=0.15)

    ax.set_xlabel("max_tokens per request")
    ax.set_ylabel("KV cache utilization")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"KV cache utilization — paged vs contiguous\n{hw_label}")
    ax.legend()
    ax.grid(visible=True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    dest = out_dir / "kv_utilization.png"
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    print(f"[kv-utilization] wrote {dest}")
    return True


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _load_records(results_dir: Path) -> list[RunRecord]:
    paths = sorted(results_dir.glob("*.json"))
    if not paths:
        logger.warning("no *.json files found in %s", results_dir)
    records = []
    for p in paths:
        rec = _parse_result(p)
        if rec is not None:
            records.append(rec)
    return records


def run(results_dir: Path, out_dir: Path) -> None:
    """Generate all charts from every ``*.json`` file under ``results_dir``.

    Charts are grouped by (gpu, model, dataset) so that different hardware
    contexts produce separate sets of figures.  Any chart whose required inputs
    are absent is skipped with a printed message; the others still render.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    records = _load_records(results_dir)
    if not records:
        print(f"No usable result files in {results_dir} — nothing to plot.")
        return

    groups: dict[tuple[str, str, str], list[RunRecord]] = defaultdict(list)
    for r in records:
        groups[(r.gpu, r.model, r.dataset)].append(r)

    for (gpu, model, dataset), grp in groups.items():
        hw_label = _hw_title(gpu, model, dataset)
        grp_out = out_dir / _hw_safe(gpu, model, dataset)
        grp_out.mkdir(parents=True, exist_ok=True)
        print(f"\n── {hw_label} ──")
        plot_throughput_vs_concurrency(grp, grp_out, hw_label)
        plot_ttft_vs_concurrency(grp, grp_out, hw_label)
        plot_itl_vs_concurrency(grp, grp_out, hw_label)
        plot_ttft_vs_prefix_fraction(grp, grp_out, hw_label)
        plot_kv_utilization(grp, grp_out, hw_label)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate all README charts from committed result JSON files. "
            "Each chart is skipped (with a message) when its inputs are absent — "
            "a partial sweep still plots what is available."
        )
    )
    parser.add_argument(
        "--results",
        required=True,
        type=Path,
        help="Directory containing *.json result files.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output directory for figures (created if absent).",
    )
    args = parser.parse_args(argv)

    if not args.results.is_dir():
        print(f"error: {args.results} is not a directory", file=sys.stderr)
        return 1

    run(args.results, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
