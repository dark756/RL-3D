"""Episode logs and replay (brief sections 6 and 9, D10, D22)."""

from __future__ import annotations

import json
from importlib import metadata

import numpy as np

from envs.episode_logger import EpisodeLogger
from envs.mars_lander_env import MarsLanderEnv
from lander.episode_log import TRAJECTORY_COLUMNS, code_version, config_from_log, keys_missing_from_log, read_episode
from scripts.replay import environment_warnings, replay
from scripts.run_episodes import GENTLE, descend


def _episode(env, policy):
    obs, _ = env.reset(seed=5)
    done = False
    while not done:
        obs, _, terminated, truncated, _ = env.step(policy(env, obs))
        done = terminated or truncated
    return env.last_path


def test_logged_landing_replays_bit_identically(tmp_path, cfg):
    env = EpisodeLogger(MarsLanderEnv(config=cfg, overrides=GENTLE), tmp_path, trajectory=True)
    path = _episode(env, descend)
    record, arrays = read_episode(path)
    assert record["outcome"]["result"] == "landed" and record["seed"] == 5
    assert record["versions"]["mujoco"] == metadata.version("mujoco")
    assert record["code"] == code_version() and record["episode"]["sim_seed"] == env.unwrapped.episode["sim_seed"]
    assert record["config"]["EPISODE_START_HEIGHT_M"]["status"] == "override"          # overrides are recorded
    assert arrays["actions"].shape == (record["steps"], 4) and arrays["actions"].dtype == np.float32
    assert arrays["trajectory"].shape == (record["steps"] + 1, len(TRAJECTORY_COLUMNS))
    assert config_from_log(record).params == env.unwrapped.sim.cfg.params                # exact config back
    check = replay(path)
    assert check.identical and check.max_state_error == 0.0


def test_failed_episode_replays_and_tampering_is_caught(tmp_path, cfg):
    rng = np.random.default_rng(1)
    env = EpisodeLogger(MarsLanderEnv(config=cfg), tmp_path, trajectory=True)
    path = _episode(env, lambda e, o: np.append(rng.uniform(-1.0, 1.0, 3), -1.0).astype(np.float32))  # tumbles
    assert read_episode(path)[0]["outcome"]["result"] == "failed"
    assert replay(path).identical
    record, arrays = read_episode(path)
    arrays["actions"][0, 0] = np.nextafter(arrays["actions"][0, 0], np.float32(2.0))  # one action, one bit
    np.savez_compressed(path.parent / record["arrays"], **arrays)
    assert not replay(path).identical


def test_log_older_than_a_new_fact_still_replays(tmp_path, cfg):
    env = EpisodeLogger(MarsLanderEnv(config=cfg, overrides=GENTLE), tmp_path, trajectory=True)
    path = _episode(env, descend)
    record, _ = read_episode(path)
    record["config"] = {k: v for k, v in record["config"].items() if not k.startswith("CTRL_")}   # as written at M2
    path.write_text(json.dumps(record), encoding="utf-8")
    assert keys_missing_from_log(record) == sorted(k for k in cfg.params if k.startswith("CTRL_"))
    assert config_from_log(record).ctrl == cfg.ctrl
    assert any("CTRL_BRAKE_MPS2" in w for w in environment_warnings(record)) and replay(path).identical


def test_logger_every_nth_episode(tmp_path, cfg):
    env = EpisodeLogger(MarsLanderEnv(config=cfg, overrides=GENTLE), tmp_path, every=2, prefix="run")
    for _ in range(3):
        _episode(env, descend)
    assert sorted(p.name for p in tmp_path.glob("*.json")) == ["run_000001.json", "run_000003.json"]


def test_code_version(tmp_path):
    assert code_version(str(tmp_path)) == {"commit": "unversioned", "dirty": None}      # not a git repo
    here = code_version()
    assert here["commit"] == "unversioned" or (len(here["commit"]) == 40 and isinstance(here["dirty"], bool))
