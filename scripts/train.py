"""Train the PPO agent (M4; brief D9, D23, D29-D31): seeds in parallel, a curriculum, checkpoints, resume.

    python -m scripts.train                                  # seeds 0-4 in parallel for 8 h -> runs/train/<date-time>
    python -m scripts.train --resume runs/train/<run>        # 8 more hours from each seed's last checkpoint
    python -m scripts.train --seeds 0 --hours 1 --out runs/train/pilot
    python -m scripts.training_report runs/train/<run>       # status table and learning curves

Each seed is one process; its envs step in --envs-per-seed worker processes (5 seeds x 2 envs was the fastest
layout measured on the 8-core laptop). Per seed, <run>/seed<k>/ holds:
- tb/: TensorBoard logs (`tensorboard --logdir <run>`);
- curriculum.csv: after every update, the level and the success rate there (the training curve);
- validation.csv: every --validate-every steps, the policy's mean action on fixed validation episodes
  (level 1, half calm, half storm, seeds 900,000+): success, safe landings, reward;
- checkpoints/latest (every --checkpoint-every steps and at the end; --resume carries on from it),
  checkpoints/step_<N> (kept every --keep-every steps) and best/ (highest validation success so far).

Curriculum (D23, D29): episodes start at level 0 (CURRICULUM_START_*). Once the last
CURRICULUM_WINDOW_EPISODES episodes started at the current level reach CURRICULUM_PROMOTE_SUCCESS, the level
rises by CURRICULUM_STEP, up to 1 (D19's full range), where training goes on. The untrained policy holds the
valves at hover trim and keeps the engines lit (D23, PPO_INIT_CUT_PROB). Observation scaling is calibrated once
per run on classical-controller landings and then fixed (`controllers/ppo.py`). Seeds: training below 800,000,
calibration 800,000+, validation 900,000+, the test sets 1,000,000+.

While training runs the laptop is kept awake (the screen may still turn off; keep it plugged in, lid open) and
the processes run at below-normal priority, so other programs come first. Ctrl+C saves every seed and stops.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import multiprocessing as mp
import os
import shutil
import sys
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

from lander.config import Config, load_config, parse_set_args
from lander.episode_log import code_version, versions

VALIDATION_SETS = {"calm": (900_000, False), "storm": (950_000, True)}   # first seed, storm
ENV_SEED_STRIDE = 10_000     # seed k's training envs start from 10,000 (k + 1) + 1,000 x (session - 1) + worker


@dataclass
class RunSettings:
    """How a run trains; written to <run>/run.json and reused by --resume."""

    seeds: list[int]
    envs_per_seed: int = 2
    subproc: bool = True
    checkpoint_every: int = 500_000
    keep_every: int = 5_000_000
    validate_every: int = 1_000_000
    validate_episodes: int = 100
    calibration_episodes: int = 40
    overrides: dict = field(default_factory=dict)


@dataclass
class SeedState:
    """Where one seed's training stands; saved with every checkpoint (state.json)."""

    timesteps: int = 0
    hours: float = 0.0
    sessions: int = 0
    level: float = 0.0
    window: list = field(default_factory=list)      # successes (1/0) of recent episodes started at `level`
    episodes: int = 0
    episodes_at_level: int = 0
    last_validation: int = 0
    best_validation: float = -1.0


# ---------------------------------------------------------------------------------------------- helpers
def parse_seeds(text: str) -> list[int]:
    """'0-4' -> [0, 1, 2, 3, 4]; '0,2,7' -> [0, 2, 7]."""
    out: list[int] = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi) + 1)) if hi else [int(lo)]
    if not out or min(out) < 0 or max(out) >= 79 or len(set(out)) != len(out):
        raise ValueError(f"seeds must be distinct whole numbers 0-78, got '{text}'")   # keeps env seeds < 800,000
    return out


def run_config(settings: RunSettings) -> Config:
    """`.env` plus the run's --set overrides."""
    return load_config(overrides=settings.overrides or None)


def keep_awake() -> bool:
    """Ask Windows not to sleep while this process runs (the screen may still turn off); False if it refused."""
    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes
    set_state = ctypes.WinDLL("kernel32").SetThreadExecutionState
    set_state.argtypes, set_state.restype = [wintypes.DWORD], wintypes.DWORD
    return set_state(0x80000000 | 0x00000001) != 0                      # ES_CONTINUOUS | ES_SYSTEM_REQUIRED


def lower_priority() -> bool:
    """Below-normal priority for this process and every process it starts, so other programs come first."""
    if sys.platform != "win32":
        os.nice(5)
        return True
    import ctypes
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE                # a 64-bit handle, not ctypes' default int
    kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.SetPriorityClass.restype = wintypes.BOOL
    return bool(kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x00004000))   # BELOW_NORMAL_PRIORITY_CLASS


def append_csv(path: Path, row: dict[str, Any]) -> None:
    """Append one row (writing the header first if the file is new)."""
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def config_changes(run: dict, cfg: Config) -> list[str]:
    """Facts whose value differs from the one recorded when the run started (or that are new / gone)."""
    then = {k: v["value"] for k, v in run["config"].items()}
    now = {k: v["value"] for k, v in cfg.to_log_dict().items()}
    return sorted(k for k in then.keys() | now.keys() if then.get(k) != now.get(k))


# ---------------------------------------------------------------------------------------------- the model
def make_model(cfg: Config, venv, seed: int, tb_dir: Path):
    """A fresh PPO agent with the PPO_* settings, starting with the engines on at hover trim (D23)."""
    from stable_baselines3 import PPO

    p = cfg.ppo
    arch = [int(n) for n in p.net_arch]
    model = PPO("MlpPolicy", venv, learning_rate=p.learning_rate, n_steps=int(p.steps_per_update) // venv.num_envs,
                batch_size=int(p.minibatch), n_epochs=int(p.epochs), gamma=p.gamma, gae_lambda=p.gae_lambda,
                clip_range=p.clip, target_kl=p.target_kl, vf_coef=p.value_coef, max_grad_norm=p.max_grad_norm,
                policy_kwargs=dict(net_arch=dict(pi=arch, vf=arch), log_std_init=math.log(p.init_std)),
                seed=seed, device="cpu", tensorboard_log=str(tb_dir), verbose=0)
    start_engines_on(model, cfg)
    return model


def start_engines_on(model, cfg: Config) -> None:
    """D23: the untrained policy's mean action holds the valves at hover trim (start mass) and keeps the engines lit.

    The engines-off output starts so far below 0 that exploration noise (PPO_INIT_STD) cuts the engines with
    probability PPO_INIT_CUT_PROB per control step; the small initial weights leave the mean at these biases.
    """
    import torch

    from lander.sim import LanderSim

    valves = LanderSim(cfg).hover_trim(cfg.mass.start_propellant_kg)
    cut_mean = NormalDist().inv_cdf(cfg.ppo.init_cut_prob) * cfg.ppo.init_std
    with torch.no_grad():
        model.policy.action_net.bias.copy_(torch.tensor([2.0 * v - 1.0 for v in valves] + [cut_mean]))


def save_checkpoint(folder: Path, model, venv, state: SeedState) -> None:
    """model.zip + vecnormalize.pkl + state.json, written to a temporary folder first and then swapped in."""
    tmp = folder.with_name(folder.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    state.timesteps = int(model.num_timesteps)
    model.save(tmp / "model.zip")
    venv.save(str(tmp / "vecnormalize.pkl"))
    (tmp / "state.json").write_text(json.dumps(asdict(state), indent=1), encoding="utf-8")
    shutil.rmtree(folder, ignore_errors=True)
    tmp.rename(folder)


def validate(model, venv, env, episodes: int) -> dict[str, float]:
    """The policy's mean action on the validation episodes (level 1): success, safe landings, mean reward."""
    outcomes: dict[str, list[dict]] = {}
    for name, (first, storm) in VALIDATION_SETS.items():
        outcomes[name] = []
        for k in range(episodes // 2):
            obs, _ = env.reset(seed=first + k, options={"storm": storm, "level": 1.0})
            done = False
            while not done:
                action, _ = model.predict(venv.normalize_obs(obs), deterministic=True)
                obs, _, done, _, info = env.step(action)
            outcomes[name].append(info["outcome"])
    every = outcomes["calm"] + outcomes["storm"]
    return {"success": float(np.mean([o["success"] for o in every])),
            "success_calm": float(np.mean([o["success"] for o in outcomes["calm"]])),
            "success_storm": float(np.mean([o["success"] for o in outcomes["storm"]])),
            "safe": float(np.mean([o["safe"] for o in every])), "reward": float(np.mean([o["reward"] for o in every]))}


# ---------------------------------------------------------------------------------------------- callbacks
def _callbacks():
    """The callback classes (built on first use, so importing this module does not import Stable-Baselines3)."""
    from stable_baselines3.common.callbacks import BaseCallback

    class Curriculum(BaseCallback):
        """Raises the curriculum level (D29) and writes the training curve (curriculum.csv, TensorBoard)."""

        def __init__(self, cfg: Config, state: SeedState, seed_dir: Path, seed: int, clock):
            super().__init__()
            self.cu, self.state, self.seed, self.clock = cfg.curriculum, state, seed, clock
            self.window = deque(state.window, maxlen=int(cfg.curriculum.window_episodes))
            self.recent: deque = deque(maxlen=100)          # (success, reward, steps) of episodes at any level
            self.reasons: deque = deque(maxlen=100)         # how those episodes ended
            self.csv = seed_dir / "curriculum.csv"

        def _on_step(self) -> bool:
            for info in self.locals["infos"]:
                outcome = info.get("outcome")
                if outcome is None:
                    continue
                self.state.episodes += 1
                self.recent.append((outcome["success"], outcome["reward"], info.get("episode", {}).get("l", 0)))
                self.reasons.append(outcome.get("reason") or outcome.get("result", "?"))
                if abs(info["level"] - self.state.level) < 1e-9:
                    self.window.append(1 if outcome["success"] else 0)
                    self.state.episodes_at_level += 1
            full = len(self.window) == self.window.maxlen
            if full and self.state.level < 1.0 and np.mean(self.window) >= self.cu.promote_success:
                self.state.level = min(1.0, round(self.state.level + self.cu.step, 6))
                self.window.clear()
                self.state.episodes_at_level = 0
                self.training_env.env_method("set_curriculum_level", self.state.level)
                print(f"[seed {self.seed}] {self.num_timesteps / 1e6:6.2f} M steps, {self.clock():5.2f} h: "
                      f"curriculum level -> {self.state.level:.1f}", flush=True)
            return True

        def _on_rollout_end(self) -> None:
            self.state.window = list(self.window)
            rate = float(np.mean(self.window)) if self.window else math.nan
            recent = np.array(self.recent, dtype=float) if self.recent else np.full((1, 3), math.nan)
            row = {"timesteps": self.num_timesteps, "hours": round(self.clock(), 4), "level": self.state.level,
                   "episodes": self.state.episodes, "episodes_at_level": self.state.episodes_at_level,
                   "success_at_level": rate, "recent_success": float(np.mean(recent[:, 0])),
                   "recent_reward": float(np.mean(recent[:, 1])), "recent_length": float(np.mean(recent[:, 2])),
                   "recent_endings": " ".join(f"{k}:{v}" for k, v in Counter(self.reasons).most_common(4))}
            for key in ("level", "success_at_level", "recent_success", "recent_reward", "recent_length"):
                self.logger.record(f"curriculum/{key}", row[key])
            append_csv(self.csv, row)

    class Validation(BaseCallback):
        """Every `validate_every` steps: flies the validation episodes; keeps best/ (validation.csv, TensorBoard)."""

        def __init__(self, cfg: Config, settings: RunSettings, state: SeedState, seed_dir: Path, seed: int, clock):
            super().__init__()
            self.cfg, self.settings, self.state, self.seed, self.clock = cfg, settings, state, seed, clock
            self.seed_dir, self.env = seed_dir, None

        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> None:
            if self.num_timesteps - self.state.last_validation < self.settings.validate_every:
                return
            from envs.mars_lander_env import MarsLanderEnv
            self.env = self.env or MarsLanderEnv(overrides=self.settings.overrides or None)
            self.state.last_validation = self.num_timesteps
            res = validate(self.model, self.training_env, self.env, self.settings.validate_episodes)
            for key, value in res.items():
                self.logger.record(f"validation/{key}", value)
            append_csv(self.seed_dir / "validation.csv",
                       {"timesteps": self.num_timesteps, "hours": round(self.clock(), 4), "level": self.state.level,
                        **res})
            best = res["success"] > self.state.best_validation
            if best:
                self.state.best_validation = res["success"]
                save_checkpoint(self.seed_dir / "best", self.model, self.training_env, self.state)
            print(f"[seed {self.seed}] {self.num_timesteps / 1e6:6.2f} M steps, {self.clock():5.2f} h: validation "
                  f"success {res['success']:.0%} (calm {res['success_calm']:.0%}, storm {res['success_storm']:.0%}), "
                  f"safe {res['safe']:.0%}, reward {res['reward']:6.1f}{'  (best so far)' if best else ''}", flush=True)

    class Keeper(BaseCallback):
        """Saves checkpoints, prints a status line every 10 minutes, and stops at the deadline."""

        def __init__(self, settings: RunSettings, state: SeedState, seed_dir: Path, seed: int, clock, deadline):
            super().__init__()
            self.settings, self.state, self.seed, self.clock, self.deadline = settings, state, seed, clock, deadline
            self.folder = seed_dir / "checkpoints"
            self.last_save, self.last_print = state.timesteps, time.time()

        def _on_step(self) -> bool:
            return time.time() < self.deadline

        def _on_rollout_end(self) -> None:
            if self.num_timesteps - self.last_save >= self.settings.checkpoint_every:
                self.state.hours = self.clock()
                save_checkpoint(self.folder / "latest", self.model, self.training_env, self.state)
                if self.num_timesteps // self.settings.keep_every > self.last_save // self.settings.keep_every:
                    shutil.copytree(self.folder / "latest", self.folder / f"step_{self.num_timesteps}",
                                    dirs_exist_ok=True)
                self.last_save = self.num_timesteps
            if time.time() - self.last_print > 600:
                self.last_print = time.time()
                rate = np.mean(self.state.window) if self.state.window else math.nan
                print(f"[seed {self.seed}] {self.num_timesteps / 1e6:6.2f} M steps, {self.clock():5.2f} h: level "
                      f"{self.state.level:.1f}, success at this level {rate:.0%} over the last "
                      f"{len(self.state.window)} episodes", flush=True)

    return Curriculum, Validation, Keeper


# ---------------------------------------------------------------------------------------------- one seed
def train_seed(run_dir: str | Path, seed: int, hours: float | None = None, steps: float | None = None) -> SeedState:
    """Train (or keep training) one seed of a run until `hours` pass or `steps` more steps are done."""
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

    from controllers.ppo import FixedMeanStd
    from envs.mars_lander_env import MarsLanderEnv

    torch.set_num_threads(1)
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    settings = RunSettings(**run["settings"])
    cfg = run_config(settings)
    seed_dir = run_dir / f"seed{seed}"
    latest = seed_dir / "checkpoints" / "latest"
    resumed = (latest / "model.zip").is_file()
    state = SeedState(**json.loads((latest / "state.json").read_text())) if resumed else SeedState()
    state.sessions += 1
    start, hours_before = time.time(), state.hours
    deadline = start + 3600.0 * hours if hours else math.inf

    def clock() -> float:
        return hours_before + (time.time() - start) / 3600.0

    venv = make_vec_env(MarsLanderEnv, n_envs=settings.envs_per_seed, env_kwargs={"overrides": settings.overrides or None},
                        seed=ENV_SEED_STRIDE * (seed + 1) + 1000 * (state.sessions - 1),
                        vec_env_cls=SubprocVecEnv if settings.subproc else DummyVecEnv)
    if resumed:
        venv = VecNormalize.load(str(latest / "vecnormalize.pkl"), venv)
        venv.training = True
        model = PPO.load(latest / "model.zip", env=venv, device="cpu", tensorboard_log=str(seed_dir / "tb"))
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, gamma=cfg.ppo.gamma)
        scaling = json.loads((run_dir / "obs_scaling.json").read_text(encoding="utf-8"))
        venv.obs_rms = FixedMeanStd(shape=(len(scaling["mean"]),))
        venv.obs_rms.mean, venv.obs_rms.var = np.array(scaling["mean"]), np.array(scaling["var"])
        venv.obs_rms.count = float(scaling["count"])
        model = make_model(cfg, venv, seed, seed_dir / "tb")
    venv.env_method("set_curriculum_level", state.level)
    curriculum, validation, keeper = _callbacks()
    callbacks = CallbackList([curriculum(cfg, state, seed_dir, seed, clock),
                              validation(cfg, settings, state, seed_dir, seed, clock),
                              keeper(settings, state, seed_dir, seed, clock, deadline)])
    print(f"[seed {seed}] {'resuming at' if resumed else 'starting'} {state.timesteps / 1e6:.2f} M steps, "
          f"level {state.level:.1f}", flush=True)
    try:
        model.learn(total_timesteps=int(steps) if steps else 10 ** 12, callback=callbacks,
                    reset_num_timesteps=not resumed, tb_log_name="ppo")
    except KeyboardInterrupt:
        print(f"[seed {seed}] stopped by Ctrl+C", flush=True)
    state.hours = clock()
    save_checkpoint(latest, model, venv, state)
    try:
        venv.close()
    except (OSError, EOFError, BrokenPipeError):
        pass
    print(f"[seed {seed}] saved at {state.timesteps / 1e6:.2f} M steps after {state.hours:.2f} h, "
          f"level {state.level:.1f}", flush=True)
    return state


# ---------------------------------------------------------------------------------------------- the run
def start_run(run_dir: Path, settings: RunSettings) -> None:
    """New run folder: run.json (settings, config, code version) and obs_scaling.json (the calibration)."""
    from controllers.ppo import calibrate_obs_scaling

    cfg = run_config(settings)
    run_dir.mkdir(parents=True, exist_ok=False)
    run = {"created": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "settings": asdict(settings),
           "code": code_version(), "versions": versions(), "config": cfg.to_log_dict()}
    (run_dir / "run.json").write_text(json.dumps(run, indent=1), encoding="utf-8")
    print(f"calibrating the observation scaling on {settings.calibration_episodes} classical landings ...", flush=True)
    rms = calibrate_obs_scaling(cfg, settings.calibration_episodes)
    scaling = {"mean": rms.mean.tolist(), "var": rms.var.tolist(), "count": rms.count}
    (run_dir / "obs_scaling.json").write_text(json.dumps(scaling, indent=1), encoding="utf-8")


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="0-4", help="e.g. 0-4 or 0,2 (new runs; default 0-4)")
    ap.add_argument("--hours", type=float, default=8.0, help="stop every seed after this long (default 8)")
    ap.add_argument("--steps", type=float, default=None, help="also stop each seed after this many more steps")
    ap.add_argument("--out", default=None, help="run folder (default runs/train/<date-time>)")
    ap.add_argument("--resume", default=None, metavar="RUN", help="carry on with an earlier run's seeds")
    ap.add_argument("--envs-per-seed", type=int, default=2)
    ap.add_argument("--in-process", action="store_true", help="step each seed's envs in its own process")
    ap.add_argument("--checkpoint-every", type=float, default=5e5, help="steps between checkpoints (default 5e5)")
    ap.add_argument("--keep-every", type=float, default=5e6, help="steps between kept checkpoints (default 5e6)")
    ap.add_argument("--validate-every", type=float, default=1e6, help="steps between validations (default 1e6)")
    ap.add_argument("--validate-episodes", type=int, default=100, help="validation episodes, half calm, half storm")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a fact for the run")
    ap.add_argument("--normal-priority", action="store_true", help="don't lower the priority")
    ap.add_argument("--allow-sleep", action="store_true", help="let the laptop sleep (training pauses)")
    args = ap.parse_args(argv)

    if args.resume:
        run_dir = Path(args.resume)
        run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        settings = RunSettings(**run["settings"])
        changed = config_changes(run, run_config(settings))
        today = [k for k in changed if not k.startswith("PPO_")]
        kept = [k for k in changed if k.startswith("PPO_")]
        if today:
            print(f"warning: these facts changed since the run started (training uses today's values): "
                  f"{', '.join(today)}", flush=True)
        if kept:
            print(f"warning: these PPO facts changed since the run started, but each checkpoint keeps the run's own "
                  f"(start a new run to use them): {', '.join(kept)}", flush=True)
    else:
        settings = RunSettings(seeds=parse_seeds(args.seeds), envs_per_seed=args.envs_per_seed,
                               subproc=not args.in_process, checkpoint_every=int(args.checkpoint_every),
                               keep_every=int(args.keep_every), validate_every=int(args.validate_every),
                               validate_episodes=args.validate_episodes, overrides=parse_set_args(args.set))
        run_config(settings)                                          # fail fast on a bad --set
        run_dir = Path(args.out or f"runs/train/{dt.datetime.now():%Y%m%d-%H%M}")
        start_run(run_dir, settings)
    if not args.normal_priority and not lower_priority():
        print("warning: could not lower the priority", flush=True)
    if not args.allow_sleep and not keep_awake():
        print("warning: could not keep the computer awake; it may sleep and pause training", flush=True)
    print(f"training seeds {settings.seeds} in {run_dir} for {args.hours:g} h"
          + (f" or {args.steps:.3g} steps" if args.steps else "") + "; Ctrl+C saves and stops", flush=True)
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=train_seed, args=(str(run_dir), s, args.hours, args.steps), name=f"seed{s}")
             for s in settings.seeds]
    for p in procs:
        p.start()
    try:
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        print("Ctrl+C: waiting for every seed to save ...", flush=True)
        for p in procs:
            p.join(timeout=300)
    from scripts.training_report import print_report
    print_report(run_dir)
    return run_dir


if __name__ == "__main__":
    main()
