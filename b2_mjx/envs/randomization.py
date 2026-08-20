"""Per-environment physical parameter randomization for Brax/MJX."""

from __future__ import annotations

import jax
import jax.numpy as jp


MIN_DYNAMIC_MASS = 1e-3


def _uniform(keys, low: float, high: float):
    return jax.vmap(lambda key: jax.random.uniform(key, (), minval=low, maxval=high))(keys)


def _body_id(sys, name: str) -> int:
    return int(sys.mj_model.body(name).id)


def _geom_id(sys, name: str) -> int:
    return int(sys.mj_model.geom(name).id)


def _dynamic_system(sys, mass, length):
    body_id = _body_id(sys, "payload_pitch_frame")
    geom_id = _geom_id(sys, "dynamic_payload_geom")
    body_mass = sys.body_mass.at[body_id].set(mass)
    # Preserve the terminal sphere's radius while scaling its inertia with mass.
    body_inertia = sys.body_inertia.at[body_id].set(jp.maximum(mass, MIN_DYNAMIC_MASS) * (2.0 / 5.0) * 0.08 ** 2)
    body_ipos = sys.body_ipos.at[body_id, 2].set(-length)
    geom_pos = sys.geom_pos.at[geom_id, 2].set(-length)
    return sys.tree_replace({"body_mass": body_mass, "body_inertia": body_inertia, "body_ipos": body_ipos, "geom_pos": geom_pos})


def payload_adjusted_system(sys, payload, *, dynamic: bool = False, length: float = 0.2):
    """Returns a system configured with one payload.

    Dynamic payloads use the pendulum body.  Static payloads merge mass and
    parallel-axis inertia into the base and leave a negligible pendulum mass.
    """
    mass, px, py, pz = payload
    if dynamic:
        configured = _dynamic_system(sys, jp.maximum(mass, MIN_DYNAMIC_MASS), length)
        roll_frame = _body_id(sys, "payload_roll_frame")
        body_pos = configured.body_pos.at[roll_frame].set(jp.array([px, py, pz]))
        return configured.tree_replace({"body_pos": body_pos})

    base_id = _body_id(sys, "base_link")
    base_mass = sys.body_mass[base_id]
    new_mass = base_mass + mass
    base_com = sys.body_ipos[base_id]
    payload_pos = jp.array([px, py, pz])
    new_com = (base_mass * base_com + mass * payload_pos) / jp.maximum(new_mass, 1e-6)
    displacement = payload_pos - new_com
    payload_inertia = mass * jp.array([
        displacement[1] ** 2 + displacement[2] ** 2,
        displacement[0] ** 2 + displacement[2] ** 2,
        displacement[0] ** 2 + displacement[1] ** 2,
    ])
    configured = _dynamic_system(sys, jp.array(MIN_DYNAMIC_MASS), length)
    return configured.tree_replace({
        "body_mass": configured.body_mass.at[base_id].set(new_mass),
        "body_ipos": configured.body_ipos.at[base_id].set(new_com),
        "body_inertia": configured.body_inertia.at[base_id].add(payload_inertia),
    })


def make_domain_randomization(config):
    payload = config["payload"]
    dynamic_cfg = config["dynamic_payload"]
    randomization = config["randomization"]
    dynamic = bool(dynamic_cfg["enabled"])

    def randomize(sys, rng):
        keys = jax.vmap(lambda key: jax.random.split(key, 8))(rng)
        mass = _uniform(keys[:, 0], *(dynamic_cfg["mass"] if dynamic else payload["mass"]))
        px = _uniform(keys[:, 1], *payload["x"])
        py = _uniform(keys[:, 2], *payload["y"])
        pz = _uniform(keys[:, 3], *payload["z"])
        length = _uniform(keys[:, 4], *dynamic_cfg["length"])
        friction = _uniform(keys[:, 5], *randomization["friction"])
        motor_strength = _uniform(keys[:, 6], *randomization["motor_strength"])

        count = rng.shape[0]
        body_mass = jp.repeat(sys.body_mass[None, :], count, axis=0)
        body_pos = jp.repeat(sys.body_pos[None, :, :], count, axis=0)
        body_ipos = jp.repeat(sys.body_ipos[None, :, :], count, axis=0)
        body_inertia = jp.repeat(sys.body_inertia[None, :, :], count, axis=0)
        base_id = _body_id(sys, "base_link")
        payload_id = _body_id(sys, "payload_pitch_frame")
        roll_id = _body_id(sys, "payload_roll_frame")
        payload_geom_id = _geom_id(sys, "dynamic_payload_geom")
        geom_pos = jp.repeat(sys.geom_pos[None, :, :], count, axis=0)

        if dynamic:
            body_mass = body_mass.at[:, payload_id].set(mass)
            sphere_inertia = mass[:, None] * (2.0 / 5.0) * 0.08 ** 2
            body_inertia = body_inertia.at[:, payload_id].set(jp.repeat(sphere_inertia, 3, axis=1))
            body_ipos = body_ipos.at[:, payload_id, 2].set(-length)
            geom_pos = geom_pos.at[:, payload_geom_id, 2].set(-length)
            body_pos = body_pos.at[:, roll_id].set(jp.stack((px, py, pz), axis=-1))
        else:
            nominal_mass = sys.body_mass[base_id]
            new_mass = nominal_mass + mass
            base_com = sys.body_ipos[base_id]
            payload_pos = jp.stack((px, py, pz), axis=-1)
            new_com = (nominal_mass * base_com + mass[:, None] * payload_pos) / jp.maximum(new_mass[:, None], 1e-6)
            displacement = payload_pos - new_com
            payload_inertia = mass[:, None] * jp.stack((
                displacement[:, 1] ** 2 + displacement[:, 2] ** 2,
                displacement[:, 0] ** 2 + displacement[:, 2] ** 2,
                displacement[:, 0] ** 2 + displacement[:, 1] ** 2,
            ), axis=-1)
            body_mass = body_mass.at[:, base_id].set(new_mass)
            body_ipos = body_ipos.at[:, base_id].set(new_com)
            body_inertia = body_inertia.at[:, base_id].add(payload_inertia)
            body_mass = body_mass.at[:, payload_id].set(MIN_DYNAMIC_MASS)
            body_inertia = body_inertia.at[:, payload_id].set(MIN_DYNAMIC_MASS * (2.0 / 5.0) * 0.08 ** 2)

        floor_id = _geom_id(sys, "floor")
        geom_friction = jp.repeat(sys.geom_friction[None, :, :], count, axis=0)
        geom_friction = geom_friction.at[:, floor_id, 0].set(friction)
        actuator_gear = jp.repeat(sys.actuator_gear[None, :, :], count, axis=0)
        actuator_gear = actuator_gear.at[:, :12].multiply(motor_strength[:, None, None])
        randomized = sys.tree_replace({
            "body_mass": body_mass,
            "body_pos": body_pos,
            "body_ipos": body_ipos,
            "body_inertia": body_inertia,
            "geom_pos": geom_pos,
            "geom_friction": geom_friction,
            "actuator_gear": actuator_gear,
        })
        in_axes = jax.tree_util.tree_map(lambda _: None, sys).tree_replace({
            "body_mass": 0,
            "body_pos": 0,
            "body_ipos": 0,
            "body_inertia": 0,
            "geom_pos": 0,
            "geom_friction": 0,
            "actuator_gear": 0,
        })
        return randomized, in_axes

    return randomize
