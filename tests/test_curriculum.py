import csv
import json
from pathlib import Path

import pytest

from b2_mjx.curriculum import load_curriculum, stage_config
from b2_mjx.evaluation.gates import detect_policy_collapse, evaluate_gate
import b2_mjx.run_curriculum as runner


def evaluation_rows(success=1.0, rmse=0.2, fallen=0.0, steps=625.0):
    return [{
        "success": str(success), "steps": str(steps), "distance": "10.0",
        "velocity_rmse": str(rmse), "fallen": str(fallen),
    } for _ in range(100)]


def test_curriculum_is_monotonic_and_budget_balanced():
    spec = load_curriculum("configs/curriculum.yaml")
    assert [stage["stage_timesteps"] for stage in spec["stages"]] == [25_000_000] * 4
    assert sum(stage["stage_timesteps"] for stage in spec["stages"]) == spec["total_timesteps"]
    assert stage_config(spec, 3, "gru_dynamic", 2)["training"]["num_timesteps"] == 25_000_000


def test_success_early_termination_does_not_fail_length_gate():
    result = evaluate_gate(evaluation_rows(steps=625.0))
    assert result["passed"]
    assert result["mean_steps"] == 625.0


def test_gate_rejects_missing_and_nonfinite_episodes():
    rows = evaluation_rows()[:-1]
    rows[0]["velocity_rmse"] = "nan"
    result = evaluate_gate(rows)
    assert not result["passed"]
    assert any("episodes" in failure for failure in result["failures"])
    assert any("non-finite" in failure for failure in result["failures"])


def test_policy_collapse_requires_two_consecutive_regressions():
    rows = [
        {"step": "1", "eval/episode_velocity_squared_error": "100", "eval/avg_episode_length": "100"},
        {"step": "2", "eval/episode_velocity_squared_error": "130", "eval/avg_episode_length": "100"},
        {"step": "3", "eval/episode_velocity_squared_error": "140", "eval/avg_episode_length": "100"},
    ]
    result = detect_policy_collapse(rows)
    assert result["collapsed"]
    assert result["collapse_step"] == 3


def test_runner_stops_after_failed_c0_gate(tmp_path, monkeypatch):
    calls = []

    def fake_run(command):
        calls.append(command)
        if "b2_mjx.train" in command:
            run_dir = Path(command[command.index("--run-dir") + 1])
            checkpoint = run_dir / "checkpoints" / "000025000000"
            checkpoint.mkdir(parents=True)
            (run_dir / "best_checkpoint.json").write_text(json.dumps({"checkpoint": str(checkpoint)}))
            with (run_dir / "metrics.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["step", "eval/episode_velocity_squared_error", "eval/avg_episode_length"])
                writer.writeheader(); writer.writerow({"step": 25_000_000, "eval/episode_velocity_squared_error": 100, "eval/avg_episode_length": 100})
        else:
            output = Path(command[command.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(evaluation_rows()[0]))
                writer.writeheader(); writer.writerows(evaluation_rows(success=0.0, rmse=1.0, fallen=1.0))

    monkeypatch.setattr(runner, "run", fake_run)
    with pytest.raises(RuntimeError, match="c0_locomotion"):
        runner.run_curriculum(Path("configs/curriculum.yaml"), "mlp_dynamic", 0, tmp_path)
    manifest = json.loads(next(tmp_path.rglob("curriculum_manifest.json")).read_text())
    assert manifest["status"] == "failed"
    assert [stage["name"] for stage in manifest["stages"]] == ["c0_locomotion"]
    assert len(calls) == 2
