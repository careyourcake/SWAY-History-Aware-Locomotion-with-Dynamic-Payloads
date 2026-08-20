from copy import deepcopy

import jax
import jax.numpy as jp

from b2_mjx.envs.b2_payload import B2PayloadEnv, TORQUE_LIMITS, default_environment_config


def small_config():
    config = deepcopy(default_environment_config())
    config["episode_length"] = 10
    config["randomization"].update(observation_noise=0.0, action_delay_probability=0.0, push_probability=0.0)
    return config


def test_reset_step_action_and_torque_limits():
    env = B2PayloadEnv(small_config())
    state = env.reset(jax.random.PRNGKey(0))
    assert state.obs["state"].shape == (45,)
    torque = env._torque(state.pipeline_state, jp.ones(env.action_size) * 100.0)
    assert bool(jp.all(jp.abs(torque) <= TORQUE_LIMITS))
    next_state = env.step(state, jp.zeros(env.action_size))
    assert bool(jp.all(jp.isfinite(next_state.pipeline_state.q)))


def test_payload_torque_is_restoring_without_velocity():
    env = B2PayloadEnv(small_config())
    state = env.reset(jax.random.PRNGKey(1))
    q = state.pipeline_state.q.at[env.payload_qpos_ids].set(jp.array([0.1, -0.1]))
    qd = state.pipeline_state.qd.at[env.payload_qvel_ids].set(0.0)
    pipeline = state.pipeline_state.replace(q=q, qd=qd)
    torque = env._payload_torque(pipeline)
    assert bool(jp.all(torque * q[env.payload_qpos_ids] < 0.0))


def test_set_command_updates_all_policy_views():
    env = B2PayloadEnv(small_config())
    state = env.set_command(env.reset(jax.random.PRNGKey(2)), jp.array([0.8, 0.0, 0.0]))
    assert bool(jp.allclose(state.obs["state"][6:9], jp.array([0.8, 0.0, 0.0])))
    assert bool(jp.allclose(state.info["history"][:, 6:9], jp.array([0.8, 0.0, 0.0])))
