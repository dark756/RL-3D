"""`EpisodeLogger`: a Gymnasium wrapper that writes episodes of a `MarsLanderEnv` to disk (`lander.episode_log`).

    env = EpisodeLogger(MarsLanderEnv(), "runs/episodes", trajectory=True)

Each finished episode becomes <out_dir>/<prefix>_<number>.json + .npz; `scripts.replay` re-runs one.
"""

from __future__ import annotations

import os
from pathlib import Path

import gymnasium as gym
import numpy as np

from lander.episode_log import episode_record, trajectory_row, write_episode


class EpisodeLogger(gym.Wrapper):
    """Log every `every`-th episode: the actions always, the true trajectory if `trajectory` is set.

    The default file prefix is the process id, so parallel environments do not overwrite each other.
    """

    def __init__(self, env: gym.Env, out_dir: str | Path = "runs/episodes", trajectory: bool = False,
                 every: int = 1, prefix: str | None = None):
        super().__init__(env)
        self.out_dir, self.trajectory, self.every = Path(out_dir), trajectory, max(1, int(every))
        self.prefix = prefix or f"p{os.getpid()}"
        self.count = 0
        self.last_path: Path | None = None
        self._logging = False

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.count += 1
        self._logging = (self.count - 1) % self.every == 0
        if self._logging:
            self._seed, self._actions, self._rewards = seed, [], []
            self._trajectory = [trajectory_row(self.env.unwrapped.last_result.telemetry)] if self.trajectory else None
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self._logging:
            self._actions.append(np.array(action, copy=True))
            self._rewards.append(float(reward))
            if self._trajectory is not None:
                self._trajectory.append(trajectory_row(self.env.unwrapped.last_result.telemetry))
            if terminated or truncated:
                self._write(info)
        return obs, reward, terminated, truncated, info

    def _write(self, info: dict) -> None:
        env = self.env.unwrapped
        arrays = {"actions": np.asarray(self._actions), "rewards": np.asarray(self._rewards, dtype=np.float64)}
        if self._trajectory is not None:
            arrays["trajectory"] = np.asarray(self._trajectory)
        record = episode_record(env.sim.cfg, self._seed, env.episode, info.get("outcome"), sum(self._rewards),
                                len(self._actions))
        self.last_path = write_episode(self.out_dir / f"{self.prefix}_{self.count:06d}", record, arrays)
        self._logging = False
