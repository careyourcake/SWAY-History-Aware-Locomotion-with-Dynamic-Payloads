import pytest

from b2_mjx.evaluation.metrics import EpisodeTotals


def test_ten_step_episode_accounting():
    totals = EpisodeTotals(start_x=1.0)
    for step in range(1, 11):
        totals.update(base_x=1.0 + 0.02 * step, mechanical_power=5.0, dt=0.02)
    assert totals.steps == 10
    assert totals.distance == pytest.approx(0.2)
    assert totals.mechanical_energy == pytest.approx(1.0)
