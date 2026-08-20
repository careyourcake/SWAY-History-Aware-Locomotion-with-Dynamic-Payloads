"""Unitree B2 payload locomotion with MuJoCo MJX.

The environment stack is imported lazily so configuration, evaluation-gate,
and manifest tools can run on analysis machines without JAX/MuJoCo installed.
"""

__all__ = ["ENV_NAME", "B2PayloadEnv", "get_environment"]


def __getattr__(name):
    if name in __all__:
        from b2_mjx import envs
        return getattr(envs, name)
    raise AttributeError(name)
