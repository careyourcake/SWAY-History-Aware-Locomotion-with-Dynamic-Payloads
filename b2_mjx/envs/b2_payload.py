"""B2 locomotion with a hidden, low-order dynamic payload."""

from __future__ import annotations

from importlib.resources import files
from typing import Any, Mapping

import jax
import jax.numpy as jp
import mujoco
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf

from b2_mjx.envs.math import projected_gravity, quat_rotate_inverse


MOTOR_NAMES = (
    "FR_hip", "FR_thigh", "FR_calf", "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf", "RL_hip", "RL_thigh", "RL_calf",
)
PAYLOAD_ACTUATORS = ("payload_roll_torque", "payload_pitch_torque")
PAYLOAD_JOINTS = ("payload_roll_joint", "payload_pitch_joint")
FOOT_SITES = ("FR_foot_site", "FL_foot_site", "RR_foot_site", "RL_foot_site")
TORQUE_LIMITS = jp.array([200.0, 200.0, 300.0] * 4)
ACTION_SCALE = jp.array([0.25, 0.45, 0.55] * 4)
KP = jp.array([100.0, 120.0, 150.0] * 4)
KD = jp.array([3.0, 4.0, 5.0] * 4)
STATE_SIZE = 45
PRIVILEGED_SIZE = 63


def posture_reward_terms(gravity_z, base_height, joint_error, standing):
    """Bounded rewards whose maxima represent the desired posture."""
    upright = jp.exp(-jp.square(gravity_z + 1.0) / 0.05)
    height = jp.exp(-jp.square(base_height - 0.60) / 0.02)
    stand = standing.astype(jp.float32) * jp.exp(-joint_error / 0.25)
    return upright, height, stand


def default_environment_config() -> dict[str, Any]:
    return {
        "episode_length": 1000,
        "command_resample_steps": 250,
        "history_length": 50,
        "history_stride": 2,
        "command": {"forward": [-0.5, 1.5], "lateral": [-0.4, 0.4], "yaw": [-0.8, 0.8], "standing_probability": 0.1},
        "payload": {"mass": [0.0, 20.0], "x": [-0.2, 0.2], "y": [-0.12, 0.12], "z": [0.08, 0.25]},
        "dynamic_payload": {
            "enabled": True, "mass": [1.0, 5.0], "length": [0.2, 0.2],
            "linear_stiffness": 15.0, "cubic_stiffness": 25.0, "damping": 2.5,
            "angle_limit_deg": 20.0, "initial_angle_deg": [-8.0, 8.0],
            "initial_velocity": [-0.5, 0.5],
        },
        "randomization": {
            "friction": [0.5, 1.25], "motor_strength": [0.9, 1.1], "kp_scale": [0.9, 1.1],
            "kd_scale": [0.9, 1.1], "observation_noise": 0.02, "action_delay_probability": 0.15,
            "push_probability": 0.015, "push_velocity": 0.75,
        },
        "reward": {
            "tracking_linear": 4.0, "tracking_yaw": 1.0, "vertical_velocity": -1.0,
            "roll_pitch_rate": -0.05, "upright": 0.5, "height": 0.3, "torque": -2e-6,
            "power": -2e-7, "action_rate": -0.002, "foot_slip": -0.05,
            "collision": -1.0, "alive": 0.0, "stand": 0.2,
        },
    }


class B2PayloadEnv(PipelineEnv):
    """Flat-ground B2 locomotion with hidden dynamic or static payloads."""

    def __init__(self, config: Mapping[str, Any] | None = None):
        self.config = dict(config or default_environment_config())
        model_path = files("b2_mjx.assets").joinpath("unitree_b2/scene_mjx.xml")
        sys = mjcf.load(str(model_path))
        super().__init__(sys=sys, backend="mjx", n_frames=10)
        self._mj_model = mujoco.MjModel.from_xml_path(str(model_path))
        self._validate_model()
        self._motor_qpos_ids = jp.array([self._joint_qpos(name) for name in MOTOR_NAMES])
        self._motor_qvel_ids = jp.array([self._joint_dof(name) for name in MOTOR_NAMES])
        self._payload_qpos_ids = jp.array([self._named_joint_qpos(name) for name in PAYLOAD_JOINTS])
        self._payload_qvel_ids = jp.array([self._named_joint_dof(name) for name in PAYLOAD_JOINTS])
        self._payload_body_id = int(self._mj_model.body("payload_pitch_frame").id)
        self._base_body_id = int(self._mj_model.body("base_link").id)
        self._roll_frame_id = int(self._mj_model.body("payload_roll_frame").id)
        self._foot_site_ids = jp.array([self._site_id(name) for name in FOOT_SITES])
        self._foot_body_ids = jp.array([self._mj_model.site_bodyid[int(i)] - 1 for i in self._foot_site_ids])
        self._default_q = jp.array(self._mj_model.key("home").qpos)
        self._default_joint_q = self._default_q[self._motor_qpos_ids]
        self._nominal_base_mass = jp.array(self._mj_model.body_mass[self._base_body_id])
        self._reward_weights = self.config["reward"]
        self._episode_length = int(self.config["episode_length"])
        self._command_resample_steps = int(self.config["command_resample_steps"])
        self._history_length = int(self.config.get("history_length", 50))
        self._history_stride = int(self.config.get("history_stride", 2))
        self._dynamic = bool(self.config["dynamic_payload"]["enabled"])

    @property
    def action_size(self) -> int:
        return 12

    @property
    def observation_size(self):
        return {"state": STATE_SIZE, "history": self._history_length * STATE_SIZE, "stack": 5 * STATE_SIZE, "privileged_state": PRIVILEGED_SIZE}

    @property
    def privileged_observation_size(self) -> int:
        return PRIVILEGED_SIZE

    @property
    def dt(self) -> float:
        return float(self.sys.opt.timestep) * self._n_frames

    @property
    def mj_model(self):
        return self._mj_model

    @property
    def default_q(self):
        return self._default_q

    @property
    def payload_qpos_ids(self):
        return self._payload_qpos_ids

    @property
    def payload_qvel_ids(self):
        return self._payload_qvel_ids

    def _joint_qpos(self, actuator_name: str) -> int:
        joint_id = int(self._mj_model.actuator_trnid[self._mj_model.actuator(actuator_name).id, 0])
        return int(self._mj_model.jnt_qposadr[joint_id])

    def _joint_dof(self, actuator_name: str) -> int:
        joint_id = int(self._mj_model.actuator_trnid[self._mj_model.actuator(actuator_name).id, 0])
        return int(self._mj_model.jnt_dofadr[joint_id])

    def _named_joint_qpos(self, name: str) -> int:
        return int(self._mj_model.jnt_qposadr[self._mj_model.joint(name).id])

    def _named_joint_dof(self, name: str) -> int:
        return int(self._mj_model.jnt_dofadr[self._mj_model.joint(name).id])

    def _site_id(self, name: str) -> int:
        site_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id < 0:
            raise ValueError(f"Missing site {name}")
        return site_id

    def _validate_model(self) -> None:
        if (self._mj_model.nq, self._mj_model.nv, self._mj_model.nu) != (21, 20, 14):
            raise ValueError("B2 dynamic-payload model must have nq=21, nv=20, nu=14")
        names = tuple(self._mj_model.actuator(i).name for i in range(self._mj_model.nu))
        if names != MOTOR_NAMES + PAYLOAD_ACTUATORS:
            raise ValueError(f"Unexpected actuator order: {names}")
        if not bool(jp.allclose(jp.array(self._mj_model.actuator_ctrlrange[:12, 1]), TORQUE_LIMITS)):
            raise ValueError("Unexpected leg torque limits")
        for site in FOOT_SITES:
            self._site_id(site)

    def _sample_command(self, rng):
        cfg = self.config["command"]
        rng, command_key, stand_key = jax.random.split(rng, 3)
        command = jax.random.uniform(command_key, (3,), minval=jp.array([cfg["forward"][0], cfg["lateral"][0], cfg["yaw"][0]]), maxval=jp.array([cfg["forward"][1], cfg["lateral"][1], cfg["yaw"][1]]))
        standing = jax.random.uniform(stand_key) < float(cfg["standing_probability"])
        return rng, jp.where(standing, jp.zeros(3), command)

    def _payload_parameters(self):
        cfg = self.config["dynamic_payload"]
        if self._dynamic:
            mass = self.sys.body_mass[self._payload_body_id]
            length = -self.sys.body_ipos[self._payload_body_id, 2]
        else:
            mass = jp.maximum(self.sys.body_mass[self._base_body_id] - self._nominal_base_mass, 0.0)
            length = jp.zeros(())
        return jp.array([mass, length, float(cfg["linear_stiffness"]), float(cfg["cubic_stiffness"]), float(cfg["damping"])])

    def reset(self, rng: jax.Array) -> State:
        rng, command = self._sample_command(rng)
        rng, q_key, qd_key, gains_key, angle_key, velocity_key = jax.random.split(rng, 6)
        q = self._default_q.at[:3].add(jax.random.uniform(q_key, (3,), minval=-0.015, maxval=0.015))
        qd = jax.random.uniform(qd_key, (self.sys.nv,), minval=-0.05, maxval=0.05)
        dyn = self.config["dynamic_payload"]
        angle_range = jp.deg2rad(jp.array(dyn["initial_angle_deg"]))
        angles = jax.random.uniform(angle_key, (2,), minval=angle_range[0], maxval=angle_range[1])
        velocities = jax.random.uniform(velocity_key, (2,), minval=float(dyn["initial_velocity"][0]), maxval=float(dyn["initial_velocity"][1]))
        q = q.at[self._payload_qpos_ids].set(jp.where(self._dynamic, angles, jp.zeros(2)))
        qd = qd.at[self._payload_qvel_ids].set(jp.where(self._dynamic, velocities, jp.zeros(2)))
        pipeline_state = self.pipeline_init(q, qd)
        rand_cfg = self.config["randomization"]
        gains = jax.random.uniform(gains_key, (2,), minval=jp.array([rand_cfg["kp_scale"][0], rand_cfg["kd_scale"][0]]), maxval=jp.array([rand_cfg["kp_scale"][1], rand_cfg["kd_scale"][1]]))
        info = {
            "rng": rng, "step": jp.zeros((), dtype=jp.int32), "command": command,
            "last_action": jp.zeros(12), "previous_action": jp.zeros(12), "last_torque": jp.zeros(12),
            "kp_scale": gains[0], "kd_scale": gains[1], "payload": self._payload_parameters(), "timeout": jp.zeros(()),
        }
        obs = self._get_obs(pipeline_state, info, add_noise=True)
        info["history"] = jp.repeat(obs[None, :], self._history_length, axis=0)
        info["recent"] = jp.repeat(obs[None, :], 5, axis=0)
        privileged = self._get_privileged_obs(pipeline_state, info, obs)
        info["privileged_obs"] = privileged
        metrics = {name: jp.zeros(()) for name in self._reward_weights}
        metrics.update({
            "x_velocity": jp.zeros(()), "y_velocity": jp.zeros(()), "yaw_rate": jp.zeros(()),
            "distance": jp.zeros(()), "velocity_squared_error": jp.zeros(()),
            "yaw_squared_error": jp.zeros(()), "posture_error": jp.zeros(()),
            "fallen": jp.zeros(()), "timeout": jp.zeros(()),
        })
        observations = {"state": obs, "history": info["history"].reshape(-1), "stack": info["recent"].reshape(-1), "privileged_state": privileged}
        return State(pipeline_state, observations, jp.zeros(()), jp.zeros(()), metrics, info)

    def _get_obs(self, pipeline_state, info, add_noise: bool):
        q, qd = pipeline_state.q, pipeline_state.qd
        obs = jp.concatenate((qd[3:6], projected_gravity(q[3:7]), info["command"], q[self._motor_qpos_ids] - self._default_joint_q, qd[self._motor_qvel_ids], info["last_action"]))
        if add_noise:
            rng, key = jax.random.split(info["rng"])
            info["rng"] = rng
            obs = obs + float(self.config["randomization"]["observation_noise"]) * jax.random.normal(key, obs.shape)
        return obs

    def _get_privileged_obs(self, pipeline_state, info, obs):
        root_velocity = quat_rotate_inverse(pipeline_state.q[3:7], pipeline_state.qd[:3])
        foot_contact = (pipeline_state.x.pos[self._foot_body_ids, 2] < 0.045).astype(jp.float32)
        return jp.concatenate((obs, root_velocity, foot_contact, info["payload"], pipeline_state.q[self._payload_qpos_ids], pipeline_state.qd[self._payload_qvel_ids], info["kp_scale"][None], info["kd_scale"][None]))

    def set_command(self, state: State, command: jax.Array) -> State:
        """Set a deterministic command and synchronize every observation view."""
        info = dict(state.info)
        info["command"] = jp.asarray(command)
        obs = self._get_obs(state.pipeline_state, info, add_noise=False)
        info["history"] = jp.repeat(obs[None, :], self._history_length, axis=0)
        info["recent"] = jp.repeat(obs[None, :], 5, axis=0)
        info["privileged_obs"] = self._get_privileged_obs(state.pipeline_state, info, obs)
        observations = {
            "state": obs, "history": info["history"].reshape(-1),
            "stack": info["recent"].reshape(-1), "privileged_state": info["privileged_obs"],
        }
        return state.replace(obs=observations, info=info)

    def _torque(self, pipeline_state, action, kp_scale=1.0, kd_scale=1.0):
        joint_q = pipeline_state.q[self._motor_qpos_ids]
        joint_qd = pipeline_state.qd[self._motor_qvel_ids]
        target_q = self._default_joint_q + ACTION_SCALE * action
        torque = (KP * kp_scale) * (target_q - joint_q) - (KD * kd_scale) * joint_qd
        return jp.clip(torque, -TORQUE_LIMITS, TORQUE_LIMITS)

    def _payload_torque(self, pipeline_state):
        cfg = self.config["dynamic_payload"]
        angle = pipeline_state.q[self._payload_qpos_ids]
        velocity = pipeline_state.qd[self._payload_qvel_ids]
        torque = -float(cfg["linear_stiffness"]) * angle - float(cfg["cubic_stiffness"]) * angle ** 3 - float(cfg["damping"]) * velocity
        return jp.where(self._dynamic, jp.clip(torque, -500.0, 500.0), jp.zeros(2))

    def _physics_step(self, pipeline_state, leg_torque):
        def substep(state, _):
            controls = jp.concatenate((leg_torque, self._payload_torque(state)))
            return self._pipeline.step(self.sys, state, controls, self._debug), None
        return jax.lax.scan(substep, pipeline_state, (), self._n_frames)[0]

    def _foot_contact_and_slip(self, pipeline_state):
        pos, vel = pipeline_state.x.pos[self._foot_body_ids], pipeline_state.xd.vel[self._foot_body_ids]
        contact = pos[:, 2] < 0.045
        return contact, jp.sum(jp.where(contact[:, None], vel[:, :2] ** 2, 0.0))

    def _reward_terms(self, old_state, pipeline_state, action, torque, command):
        q, qd = pipeline_state.q, pipeline_state.qd
        root_velocity = quat_rotate_inverse(q[3:7], qd[:3])
        gravity = projected_gravity(q[3:7])
        _, slip = self._foot_contact_and_slip(pipeline_state)
        linear_error = jp.sum((root_velocity[:2] - command[:2]) ** 2)
        yaw_error = (qd[5] - command[2]) ** 2
        standing = jp.linalg.norm(command) < 0.05
        joint_error = jp.sum((q[self._motor_qpos_ids] - self._default_joint_q) ** 2)
        upright_reward, height_reward, stand_reward = posture_reward_terms(
            gravity[2], q[2], joint_error, standing
        )
        terms = {
            "tracking_linear": jp.exp(-linear_error / 0.25), "tracking_yaw": jp.exp(-yaw_error / 0.25),
            "vertical_velocity": qd[2] ** 2, "roll_pitch_rate": jp.sum(qd[3:5] ** 2),
            "upright": upright_reward, "height": height_reward,
            "torque": jp.sum(torque ** 2), "power": jp.sum(jp.abs(torque * qd[self._motor_qvel_ids])),
            "action_rate": jp.sum((action - old_state.info["last_action"]) ** 2), "foot_slip": slip,
            "collision": (q[2] < 0.30).astype(jp.float32), "alive": jp.ones(()),
            "stand": stand_reward,
        }
        return terms, root_velocity

    def step(self, state: State, action: jax.Array) -> State:
        action = jp.clip(action, -1.0, 1.0)
        rng, delay_key, push_key, command_key = jax.random.split(state.info["rng"], 4)
        cfg = self.config["randomization"]
        delayed = jax.random.uniform(delay_key) < float(cfg["action_delay_probability"])
        applied = jp.where(delayed, state.info["last_action"], action)
        torque = self._torque(state.pipeline_state, applied, state.info["kp_scale"], state.info["kd_scale"])
        pipeline_state = self._physics_step(state.pipeline_state, torque)
        push = jax.random.uniform(push_key) < float(cfg["push_probability"])
        push_velocity = jax.random.uniform(push_key, (2,), minval=-float(cfg["push_velocity"]), maxval=float(cfg["push_velocity"]))
        pipeline_state = pipeline_state.replace(qd=pipeline_state.qd.at[:2].add(jp.where(push, push_velocity, jp.zeros(2))))
        step = state.info["step"] + 1
        resample = step % self._command_resample_steps == 0
        _, new_command = self._sample_command(command_key)
        command = jp.where(resample, new_command, state.info["command"])
        reward_terms, root_velocity = self._reward_terms(state, pipeline_state, applied, torque, command)
        reward = sum(float(self._reward_weights[name]) * value for name, value in reward_terms.items()) * self.dt
        gravity = projected_gravity(pipeline_state.q[3:7])
        finite = jp.all(jp.isfinite(pipeline_state.q)) & jp.all(jp.isfinite(pipeline_state.qd))
        fallen = (pipeline_state.q[2] < 0.30) | (gravity[2] > -0.45)
        timeout = step >= self._episode_length
        done = (~finite) | fallen | timeout
        info = dict(state.info)
        info.update(rng=rng, step=step, command=command, previous_action=state.info["last_action"], last_action=applied, last_torque=torque, timeout=timeout.astype(jp.float32))
        obs = self._get_obs(pipeline_state, info, add_noise=True)
        recent = jp.concatenate((state.info["recent"][1:], obs[None, :]), axis=0)
        shifted_history = jp.concatenate((state.info["history"][1:], obs[None, :]), axis=0)
        history = jp.where((step % self._history_stride == 0), shifted_history, state.info["history"])
        info.update(recent=recent, history=history)
        info["privileged_obs"] = self._get_privileged_obs(pipeline_state, info, obs)
        metrics = dict(state.metrics)
        metrics.update(reward_terms)
        step_displacement = pipeline_state.q[0] - state.pipeline_state.q[0]
        metrics.update(
            x_velocity=root_velocity[0], y_velocity=root_velocity[1], yaw_rate=pipeline_state.qd[5],
            # Brax's episode wrapper performs the accumulation.  Keeping a
            # cumulative value here would make episode distance grow twice.
            distance=step_displacement,
            velocity_squared_error=jp.sum((root_velocity[:2] - command[:2]) ** 2),
            yaw_squared_error=jp.square(pipeline_state.qd[5] - command[2]),
            posture_error=jp.square(gravity[2] + 1.0) + jp.square(pipeline_state.q[2] - 0.60),
            fallen=fallen.astype(jp.float32), timeout=timeout.astype(jp.float32),
        )
        observations = {"state": obs, "history": history.reshape(-1), "stack": recent.reshape(-1), "privileged_state": info["privileged_obs"]}
        return state.replace(pipeline_state=pipeline_state, obs=observations, reward=reward, done=done.astype(jp.float32), metrics=metrics, info=info)
