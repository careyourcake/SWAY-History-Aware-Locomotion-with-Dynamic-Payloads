"""Aggregate evaluation CSVs and create paper-ready result plots."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def bootstrap_ci(values, seed=0, draws=2000):
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(values, (draws, len(values)), replace=True), axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    rows = []
    for path in args.inputs:
        with Path(path).open(newline="", encoding="utf-8") as stream:
            rows.extend(csv.DictReader(stream))
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["scenario"], row["mass"], row["length"])].append(row)
    summary = []
    for key, group in sorted(grouped.items()):
        record = dict(zip(("method", "scenario", "mass", "length"), key))
        for metric in (
            "success", "distance", "velocity_rmse", "yaw_rmse", "posture_error",
            "fallen", "timeout", "mechanical_energy", "cot", "peak_torque", "inference_ms",
        ):
            values = [float(row[metric]) for row in group if row[metric] not in ("", "nan")]
            low, high = bootstrap_ci(values) if values else (float("nan"), float("nan"))
            record.update({f"{metric}_mean": float(np.mean(values)) if values else float("nan"), f"{metric}_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, f"{metric}_ci_low": low, f"{metric}_ci_high": high})
        summary.append(record)
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Wrote summary.csv; install matplotlib to create plots."); return
    for axis, prefix in (("mass", "mass_"), ("length", "length_")):
        fig, ax = plt.subplots(figsize=(7, 4))
        methods = sorted({row["method"] for row in summary})
        for method in methods:
            selected = sorted((row for row in summary if row["method"] == method and row["scenario"].startswith(prefix)), key=lambda r: float(r[axis]))
            if selected: ax.plot([float(r[axis]) for r in selected], [r["success_mean"] for r in selected], marker="o", label=method)
        ax.set(xlabel=axis, ylabel="10 m success rate", ylim=(-0.05, 1.05)); ax.grid(alpha=0.3); ax.legend(); fig.tight_layout(); fig.savefig(output_dir / f"success_vs_{axis}.png", dpi=200); plt.close(fig)


if __name__ == "__main__": main()
