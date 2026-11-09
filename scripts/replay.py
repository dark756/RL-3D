"""Replay a logged episode (D10): rebuild its exact config, reset with the logged start, send the logged
actions, and check that the result matches the log. Can also watch it or export it as a video.

    python -m scripts.replay runs/episodes/p1234_000001.json
    python -m scripts.replay runs/episodes/p1234_000001.json --video runs/videos/replay.mp4
    python -m scripts.replay runs/episodes/p1234_000001.json --watch --speed 0.5

A replay is bit-identical only on the machine and MuJoCo version that recorded it (D10).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from envs.mars_lander_env import MarsLanderEnv, action_to_command
from lander.episode_log import code_version, config_from_log, keys_missing_from_log, read_episode, trajectory_row, versions


@dataclass
class ReplayCheck:
    """How a replay compares with its log."""

    steps: int
    logged_steps: int
    outcome: dict | None
    logged_outcome: dict | None
    total_reward: float
    logged_return: float
    max_state_error: float | None      # largest |difference| from the logged trajectory (None: none logged)

    @property
    def identical(self) -> bool:
        """Same number of steps, outcome, return and (if logged) trajectory, to the last bit."""
        return (self.steps == self.logged_steps and self.outcome == self.logged_outcome
                and self.total_reward == self.logged_return and self.max_state_error in (None, 0.0))


def environment_warnings(record: dict) -> list[str]:
    """Differences between the logging run and this one that can break bit-identical replay."""
    out = []
    now = versions()
    for key in ("mujoco", "numpy", "python", "platform"):
        if record["versions"].get(key) != now.get(key):
            out.append(f"{key}: logged {record['versions'].get(key)}, now {now.get(key)}")
    code = code_version()
    if record["code"]["commit"] != code["commit"]:
        out.append(f"code: logged commit {record['code']['commit'][:10]}, now {code['commit'][:10]}")
    if record["code"].get("dirty") or code.get("dirty"):
        out.append("code: uncommitted changes when logging or now, so the code may differ")
    missing = keys_missing_from_log(record)
    if missing:
        out.append(f"config: not in the log (added to .env later), taken from today's .env: {', '.join(missing)}")
    return out


def replay(path: str | Path, on_step: Callable[[MarsLanderEnv], None] | None = None) -> ReplayCheck:
    """Re-run a logged episode and compare; `on_step(env)` runs after the reset and after every step."""
    record, arrays = read_episode(path)
    env = MarsLanderEnv(config=config_from_log(record))
    try:
        env.reset(seed=record["seed"], options=record["episode"])
        traj = arrays.get("trajectory")
        errors: list[float] = []

        def compare(i: int) -> None:
            if traj is not None and i < len(traj):
                errors.append(float(np.max(np.abs(trajectory_row(env.last_result.telemetry) - traj[i]))))

        compare(0)
        if on_step:
            on_step(env)
        total, info, steps = 0.0, {}, 0
        for action in arrays["actions"]:
            _, reward, terminated, truncated, info = env.step(action)
            total += reward
            steps += 1
            compare(steps)
            if on_step:
                on_step(env)
            if terminated or truncated:
                break
        return ReplayCheck(steps, int(record["steps"]), info.get("outcome"), record["outcome"], total,
                           float(record["return"]), max(errors) if errors else None)
    finally:
        env.close()


def watch_replay(path: str | Path, speed: float = 1.0) -> None:
    """Show a logged episode in the live viewer (Space pauses, Enter restarts)."""
    from lander.viewer import watch

    record, arrays = read_episode(path)
    env = MarsLanderEnv(config=config_from_log(record))
    actions, state = arrays["actions"], {"i": 0}

    def reset_fn():
        state["i"] = 0
        env.reset(seed=record["seed"], options=record["episode"])
        return env.last_result

    def policy(sim):
        action = actions[min(state["i"], len(actions) - 1)]
        state["i"] += 1
        return action_to_command(action)

    watch(env.sim, policy, reset_fn, speed=speed, hold_after_done_s=0.0,
          done_fn=lambda sim, res: state["i"] >= len(actions))


def main(argv: list[str] | None = None) -> ReplayCheck | None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="episode JSON written by EpisodeLogger")
    ap.add_argument("--video", default=None, help="also write an MP4 of the replay here")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--camera", default="chase", choices=["chase", "ground", "padcam"])
    ap.add_argument("--watch", action="store_true", help="show it in the live viewer instead of checking it")
    ap.add_argument("--speed", type=float, default=1.0, help="viewer playback speed")
    args = ap.parse_args(argv)

    record, _ = read_episode(args.log)
    for warning in environment_warnings(record):
        print("warning:", warning)
    if args.watch:
        watch_replay(args.log, args.speed)
        return None
    video = {}

    def on_step(env: MarsLanderEnv) -> None:
        if args.video:
            if "writer" not in video:
                from lander.video import VideoWriter
                video["writer"] = VideoWriter(env.sim, args.video, args.width, args.height, args.camera)
            video["writer"].add(env.last_result)

    try:
        check = replay(args.log, on_step)
    finally:
        if "writer" in video:
            video["writer"].close()
    out = check.outcome or {}
    result = " ".join(str(x) for x in (out.get("result"), out.get("reason")) if x)
    print(f"replayed {check.steps} of {check.logged_steps} steps: {result}, "
          f"return {check.total_reward:.3f} (logged {check.logged_return:.3f})")
    if check.max_state_error is not None:
        print(f"largest difference from the logged trajectory: {check.max_state_error:.3g}")
    print("MATCH: identical to the log" if check.identical else "DIFFERENT from the log")
    if "writer" in video:
        print(f"wrote {video['writer'].count} frames to {args.video}")
    return check


if __name__ == "__main__":
    main()
