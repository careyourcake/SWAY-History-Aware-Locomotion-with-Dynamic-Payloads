"""Environment registry."""

from b2_mjx.envs.b2_payload import B2PayloadEnv

ENV_NAME = "B2PayloadVelocity-v0"


def get_environment(name: str = ENV_NAME, **kwargs) -> B2PayloadEnv:
    if name != ENV_NAME:
        raise KeyError(f"Unknown environment {name!r}; available: {ENV_NAME}")
    return B2PayloadEnv(**kwargs)


__all__ = ["ENV_NAME", "B2PayloadEnv", "get_environment"]

