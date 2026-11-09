"""Run the training environment with a policy, print each outcome, and optionally log the episodes.

    python -m scripts.run_episodes --policy classical --episodes 5 --log runs/episodes --trajectory
    python -m scripts.run_episodes --policy descend --gentle --episodes 5
    python -m scripts.replay runs/episodes/<file>.json --watch

Policies (`POLICIES`):
- classical: the M3 controller (`controllers/classical.py`), flying the same observations as the agent.
- random: uniform random actions, engines-off included, so it cuts the engines early or tumbles.
- descend: valve settings for 95 % of the weight and no attitude control; engines off once a leg touches.
  It only lands from gentle low starts: --gentle sets 5-10 m up, 1-2 m/s down, calm, flat and nearly level.
- ppo:FOLDER: a trained PPO agent, from a checkpoint folder written by scripts/train.py
  (e.g. ppo:runs/train/<run>/seed0/best); its mean action, no exploration noise.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np

from controllers.classical import ClassicalController
from envs.episode_logger import EpisodeLogger
from envs.mars_lander_env import OBS_NAMES, MarsLanderEnv
from scripts.common import add_config_args, config_from_args

POLICIES = ("classical", "random", "descend")

GENTLE = {"EPISODE_START_HEIGHT_M": "5,10", "EPISODE_START_DESCENT_MPS": "1,2", "EPISODE_START_SIDEWAYS_MPS": "0,0.3",
          "EPISODE_START_OFFSET_M": "0,3", "EPISODE_START_TILT_DEG": "0,1", "EPISODE_START_RATE_DPS": "0.2",
          "EPISODE_STORM_FRACTION": "0", "MARS_TERRAIN_SLOPE_MAX_DEG": "0"}
_LEGS = slice(OBS_NAMES.index("leg_1"), OBS_NAMES.index("leg_3") + 1)


def descend(env, obs: np.ndarray) -> np.ndarray:
    """Valves for 95 % of the current weight (torque-free trim), engines off once any leg has touched."""
    sim = env.unwrapped.sim
    weight = sim.vehicle.mass.total(sim.vehicle.propellant) * sim.cfg.mars.gravity_mps2
    valves = np.array(sim.hover_trim(total_thrust=0.95 * weight))
    return np.concatenate([2.0 * valves - 1.0, [1.0 if obs[_LEGS].any() else -1.0]]).astype(np.float32)


def make_policy(name: str, env, seed: int = 0) -> Callable[[np.ndarray], np.ndarray]:
    """A fresh policy for one episode of `env`: `policy(obs)` -> action.

    The classical controller is built from the env's base config (`.env` plus env-wide overrides), so it knows
    the nominal vehicle; per-episode overrides change only the simulated world.
    """
    if name == "classical":
        return ClassicalController(env.unwrapped.base_cfg)
    if name == "random":
        rng = np.random.default_rng(seed)
        return lambda obs: rng.uniform(-1.0, 1.0, 4).astype(np.float32)
    if name == "descend":
        return lambda obs: descend(env, obs)
    if name.startswith("ppo:"):
        from controllers.ppo import load_ppo_policy     # imports Stable-Baselines3 only when needed
        return load_ppo_policy(name[4:])
    raise ValueError(f"unknown policy {name!r}; choose from {POLICIES} or ppo:FOLDER")


def policy_arg(name: str) -> str:
    """argparse type for --policy: one of `POLICIES`, or ppo:FOLDER with a checkpoint in FOLDER."""
    if name in POLICIES:
        return name
    if name.startswith("ppo:"):
        if not (Path(name[4:]) / "model.zip").is_file():
            raise argparse.ArgumentTypeError(f"no model.zip in {name[4:]} (a checkpoint folder from scripts.train)")
        return name
    raise argparse.ArgumentTypeError(f"choose from {', '.join(POLICIES)} or ppo:FOLDER")


def main(argv: list[str] | None = None) -> list[dict]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="classical", type=policy_arg,
                    help=f"{', '.join(POLICIES)} or ppo:FOLDER (default classical)")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0, help="episode k uses seed + k")
    ap.add_argument("--gentle", action="store_true", help="gentle low starts (see above)")
    ap.add_argument("--log", default=None, metavar="DIR", help="write each episode to DIR (JSON + NPZ)")
    ap.add_argument("--trajectory", action="store_true", help="also log the true trajectory (for replay checks)")
    add_config_args(ap)
    args = ap.parse_args(argv)

    env = MarsLanderEnv(config=config_from_args(args), overrides=GENTLE if args.gentle else None)
    if args.log:
        env = EpisodeLogger(env, args.log, trajectory=args.trajectory)
    outcomes = []
    for k in range(args.episodes):
        obs, info = env.reset(seed=args.seed + k)
        policy = make_policy(args.policy, env, args.seed + k)
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(policy(obs))
            done = terminated or truncated
        o = info["outcome"]
        outcomes.append(o)
        where = f"  -> {env.last_path}" if args.log else ""
        print(f"episode {k}: {o['result']:<14} {o['reason']:<20} reward {o['reward']:8.2f}  "
              f"t {o['time_s']:6.2f} s  {o['distance_m']:6.1f} m from the target{where}")
    counts = Counter(o["result"] for o in outcomes)
    print(f"{len(outcomes)} episodes: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    return outcomes


if __name__ == "__main__":
    main()
