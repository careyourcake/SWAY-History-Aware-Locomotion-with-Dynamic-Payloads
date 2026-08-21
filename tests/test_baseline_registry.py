from pathlib import Path

import yaml


def test_baseline_registry_marks_unimplemented_methods():
    spec = yaml.safe_load(Path("configs/baselines.yaml").read_text())
    assert "baselines" in spec
    assert spec["baselines"]["mule"]["status"] == "reproduce_before_claim"
    assert spec["baselines"]["mule"]["implementation"] == "not_implemented"


def test_custom_stages_are_explicitly_diagnostics():
    spec = yaml.safe_load(Path("configs/diagnostics.yaml").read_text())
    assert len(spec["diagnostics"]) == 4
    assert all("purpose" in entry for entry in spec["diagnostics"])
