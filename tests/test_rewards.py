import jax.numpy as jp

from b2_mjx.envs.b2_payload import posture_reward_terms


def test_posture_rewards_peak_at_target():
    good = posture_reward_terms(jp.array(-1.0), jp.array(0.60), jp.array(0.0), jp.array(True))
    bad = posture_reward_terms(jp.array(-0.7), jp.array(0.40), jp.array(2.0), jp.array(True))
    assert all(float(a) > float(b) for a, b in zip(good, bad))


def test_stand_reward_is_zero_for_nonstanding_command():
    _, _, stand = posture_reward_terms(jp.array(-1.0), jp.array(0.60), jp.array(0.0), jp.array(False))
    assert float(stand) == 0.0
