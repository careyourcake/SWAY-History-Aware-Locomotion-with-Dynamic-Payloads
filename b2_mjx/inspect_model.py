"""Print and validate the packaged B2 model."""

from __future__ import annotations

import json

from b2_mjx.envs.b2_payload import B2PayloadEnv, FOOT_SITES, MOTOR_NAMES, TORQUE_LIMITS


def main() -> None:
    env = B2PayloadEnv()
    summary = {
        "model": "b2_payload_mjx",
        "nq": env.mj_model.nq,
        "nv": env.mj_model.nv,
        "nu": env.mj_model.nu,
        "physics_dt": float(env.sys.opt.timestep),
        "control_dt": env.dt,
        "motors": list(MOTOR_NAMES),
        "torque_limits_nm": list(map(float, TORQUE_LIMITS)),
        "foot_sites": list(FOOT_SITES),
        "observation_size": env.observation_size["state"],
        "privileged_observation_size": env.privileged_observation_size,
        "action_size": env.action_size,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
