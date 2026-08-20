"""Resumable runner for the capability-gated C0-C3 curriculum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml

from b2_mjx.curriculum import load_curriculum, stage_config
from b2_mjx.evaluation.gates import detect_policy_collapse, evaluate_gate, read_csv


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def best_checkpoint(run_dir: Path) -> Path:
    manifest = run_dir / "best_checkpoint.json"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing best checkpoint manifest: {manifest}")
    checkpoint = Path(json.loads(manifest.read_text(encoding="utf-8"))["checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing selected checkpoint: {checkpoint}")
    return checkpoint.resolve()


def latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoint_root = run_dir / "checkpoints"
    if not checkpoint_root.exists():
        return None
    candidates = [path for path in checkpoint_root.iterdir() if path.name.isdigit()]
    return max(candidates, key=lambda path: int(path.name)).resolve() if candidates else None


def run_curriculum(spec_path: Path, method: str, seed: int, output_root: Path) -> dict[str, Any]:
    spec = load_curriculum(spec_path)
    root = output_root.resolve() / method / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "curriculum_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "method": method, "seed": seed, "curriculum_config": str(spec_path.resolve()),
            "total_timesteps": int(spec["total_timesteps"]), "status": "running", "stages": [],
        }
    previous_checkpoint: Path | None = None
    completed = {record["name"]: record for record in manifest["stages"] if record.get("passed")}
    for index, stage in enumerate(spec["stages"]):
        stage_name = stage["name"]
        run_dir = root / stage_name
        if stage_name in completed:
            previous_checkpoint = best_checkpoint(run_dir)
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        generated = run_dir / "stage_config.yaml"
        config = stage_config(spec, index, method, seed)
        config["curriculum"]["source_checkpoint"] = str(previous_checkpoint) if previous_checkpoint else None
        generated.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        record = {
            "index": index, "name": stage_name,
            "stage_timesteps": int(stage["stage_timesteps"]),
            "cumulative_timesteps": sum(int(item["stage_timesteps"]) for item in spec["stages"][:index + 1]),
            "scenario": stage["scenario"], "run_dir": str(run_dir),
            "difficulty": stage["environment"],
            "source_checkpoint": str(previous_checkpoint) if previous_checkpoint else None,
            "stage_config": str(generated.resolve()), "passed": False,
        }
        manifest["stages"] = [r for r in manifest["stages"] if r.get("name") != stage_name] + [record]
        write_json(manifest_path, manifest)
        if not (run_dir / "best_checkpoint.json").exists():
            command = [sys.executable, "-m", "b2_mjx.train", "--config", str(generated), "--run-dir", str(run_dir)]
            resume_checkpoint = latest_checkpoint(run_dir) or previous_checkpoint
            if resume_checkpoint:
                command.extend(("--resume", str(resume_checkpoint)))
            run(command)
        selected = best_checkpoint(run_dir)
        evaluation_path = run_dir / "evaluation" / "gate_metrics.csv"
        if not evaluation_path.exists():
            run([
                sys.executable, "-m", "b2_mjx.evaluate", "--checkpoint", str(run_dir),
                "--scenario", stage["scenario"], "--episodes", str(spec["episodes_per_gate"]),
                "--output", str(evaluation_path),
            ])
        gate = evaluate_gate(
            read_csv(evaluation_path), expected_episodes=int(spec["episodes_per_gate"]),
            min_success_rate=float(spec["gate"]["min_success_rate"]),
            max_velocity_rmse=float(spec["gate"]["max_velocity_rmse"]),
            max_fall_rate=float(spec["gate"]["max_fall_rate"]),
        )
        collapse = detect_policy_collapse(read_csv(run_dir / "metrics.csv"))
        record.update(best_checkpoint=str(selected), evaluation=str(evaluation_path), gate=gate, collapse=collapse, passed=gate["passed"])
        write_json(run_dir / "gate_result.json", gate)
        manifest["stages"] = [r if r.get("name") != stage_name else record for r in manifest["stages"]]
        manifest["status"] = "running" if gate["passed"] else "failed"
        write_json(manifest_path, manifest)
        if not gate["passed"]:
            raise RuntimeError(f"Curriculum gate failed at {stage_name}: {'; '.join(gate['failures'])}")
        previous_checkpoint = selected
    manifest["status"] = "complete"
    manifest["final_checkpoint"] = str(previous_checkpoint)
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=("mlp_dynamic", "stack5_dynamic", "gru_dynamic"))
    parser.add_argument("seed", type=int)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--curriculum", type=Path, default=Path("configs/curriculum.yaml"))
    args = parser.parse_args()
    run_curriculum(args.curriculum, args.method, args.seed, args.output_root)


if __name__ == "__main__":
    main()
