"""The PPO agent as a controller (M4): a trained policy flying the env's 22 observations, like the classical one.

`PpoPolicy` loads a checkpoint folder written by `scripts/train.py` (`model.zip` + `vecnormalize.pkl`) and turns
a raw observation into an action: the policy's mean (no exploration noise). `scripts/evaluate.py` flies it with
`--policy ppo:FOLDER` on the same fixed test sets as the classical controller.

Observation scaling (VecNormalize, brief D9): the mean and variance of each observation are measured once on
classical-controller landings over the full start range (`calibrate_obs_scaling`) and then kept fixed
(`FixedMeanStd`). VecNormalize's running average would follow the curriculum: after millions of low starts,
heights of 200-500 m would land 10+ standard deviations out and be clipped. Rewards are scaled as usual.
"""

from __future__ import annotations

import functools
import pickle
from pathlib import Path

import numpy as np
from stable_baselines3.common.running_mean_std import RunningMeanStd

from lander.config import Config

CALIBRATION_FIRST_SEED = 800_000     # calibration episodes: seeds 800,000+ (validation 900,000+, test sets 1,000,000+)
VAR_FLOOR = 1e-4                     # smallest variance used to scale an observation


class FixedMeanStd(RunningMeanStd):
    """VecNormalize's observation statistics, set once by `calibrate_obs_scaling` and then left alone."""

    def update(self, arr: np.ndarray) -> None:
        """Ignore new samples: the scaling stays as calibrated."""


def calibrate_obs_scaling(cfg: Config, episodes: int = 40, first_seed: int = CALIBRATION_FIRST_SEED) -> FixedMeanStd:
    """Mean and variance of every observation over classical-controller landings from the full start range.

    Level 1 (D19), every other episode a storm; seeds below 1,000,000 and apart from the validation seeds.
    """
    from controllers.classical import ClassicalController
    from envs.mars_lander_env import MarsLanderEnv

    env, ctl = MarsLanderEnv(cfg), ClassicalController(cfg)
    rows = []
    for k in range(episodes):
        obs, _ = env.reset(seed=first_seed + k, options={"storm": k % 2 == 1, "level": 1.0})
        ctl.reset()
        rows.append(obs)
        done = False
        while not done:
            obs, _, done, _, _ = env.step(ctl(obs))
            rows.append(obs)
    data = np.asarray(rows, dtype=np.float64)
    rms = FixedMeanStd(shape=data.shape[1:])
    rms.mean, rms.var, rms.count = data.mean(axis=0), np.maximum(data.var(axis=0), VAR_FLOOR), float(len(data))
    return rms


class PpoPolicy:
    """A trained checkpoint as a controller: raw observation -> action (the policy's mean)."""

    def __init__(self, folder: str | Path):
        import torch
        from stable_baselines3 import PPO

        torch.set_num_threads(1)     # identical results every time, and parallel evaluations don't compete
        folder = Path(folder)
        self.folder = folder
        self.model = PPO.load(folder / "model.zip", device="cpu")
        with open(folder / "vecnormalize.pkl", "rb") as f:
            self.norm = pickle.load(f)

    def reset(self) -> None:
        """Start of an episode (the policy keeps no state)."""

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(self.norm.normalize_obs(np.asarray(obs, dtype=np.float32)), deterministic=True)
        return np.clip(action, -1.0, 1.0).astype(np.float32)


@functools.lru_cache(maxsize=8)
def load_ppo_policy(folder: str) -> PpoPolicy:
    """`PpoPolicy` for a checkpoint folder, loaded once per process."""
    return PpoPolicy(folder)
