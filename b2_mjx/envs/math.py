"""Small quaternion helpers using MuJoCo's wxyz convention."""

from __future__ import annotations

import jax.numpy as jp


def quat_conjugate(q):
    return jp.concatenate((q[..., :1], -q[..., 1:]), axis=-1)


def quat_rotate(q, v):
    q_vec = q[..., 1:]
    uv = jp.cross(q_vec, v)
    uuv = jp.cross(q_vec, uv)
    return v + 2.0 * (q[..., :1] * uv + uuv)


def quat_rotate_inverse(q, v):
    return quat_rotate(quat_conjugate(q), v)


def projected_gravity(q):
    return quat_rotate_inverse(q, jp.array([0.0, 0.0, -1.0]))

