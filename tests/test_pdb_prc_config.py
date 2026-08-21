from b2_mjx.config import load_config


def test_pdb_prc_policy_config_is_valid():
    config = load_config("configs/pdb_prc_smoke.yaml")
    assert config["policy"]["type"] == "pdb_prc"
    assert config["policy"]["gate_temperature"] > 0
