"""Render a deterministic checkpoint rollout to MP4."""

from __future__ import annotations

import argparse
from copy import deepcopy

import imageio.v2 as imageio
import jax
import jax.numpy as jp
import mujoco

from b2_mjx.envs import B2PayloadEnv
from b2_mjx.evaluation.policy import load_checkpoint_policy
from b2_mjx.evaluation.scenarios import SCENARIOS
from b2_mjx.envs.randomization import payload_adjusted_system


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, default="mass_3")
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()
    config, run_dir, policy = load_checkpoint_policy(args.checkpoint)
    scenario = SCENARIOS[args.scenario]
    env_config = deepcopy(config["environment"])
    env_config["command_resample_steps"] = env_config["episode_length"] + 1
    env = B2PayloadEnv(env_config)
    payload = jp.array([scenario["mass"], 0.0, 0.0, 0.30])
    env.sys = payload_adjusted_system(
        env.sys, payload,
        dynamic=bool(env_config["dynamic_payload"]["enabled"]),
        length=scenario["length"],
    )
    key = jax.random.PRNGKey(config["seed"])
    state = env.reset(key)
    state = env.set_command(state, jp.array(scenario["command"]))
    step = jax.jit(env.step)
    try:
        renderer = mujoco.Renderer(env.mj_model, height=720, width=1280)
    except Exception as exc:
        raise RuntimeError(
            "MuJoCo could not create an offscreen OpenGL context. On a Linux "
            "headless host install EGL support and run with MUJOCO_GL=egl; on "
            "macOS run from an interactive desktop session."
        ) from exc
    data = mujoco.MjData(env.mj_model)
    frames = []
    count = int(args.seconds / env.dt)
    for _ in range(count):
        key, action_key = jax.random.split(key)
        action, _ = policy(state.obs, action_key)
        state = step(state, action)
        jax.block_until_ready(state.pipeline_state.q)
        data.qpos[:] = state.pipeline_state.q
        data.qvel[:] = state.pipeline_state.qd
        mujoco.mj_forward(env.mj_model, data)
        renderer.update_scene(data)
        frames.append(renderer.render())
        if float(state.done):
            break
    renderer.close()
    output = run_dir / "videos" / f"{args.scenario}.mp4"
    imageio.mimsave(output, frames, fps=round(1.0 / env.dt), codec="libx264")
    print(f"Wrote {len(frames)} frames to {output}")


if __name__ == "__main__":
    main()
