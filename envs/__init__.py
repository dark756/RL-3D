"""Gymnasium environments (M2). Importing this package registers "MarsLander-v0" (`gymnasium.make`)."""

from gymnasium.envs.registration import register

from .mars_lander_env import OBS_NAMES, MarsLanderEnv  # noqa: F401

register(id="MarsLander-v0", entry_point="envs.mars_lander_env:MarsLanderEnv")
